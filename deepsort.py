import os
import cv2
import time
import argparse
import torch
import warnings
import json
import numpy as np
import csv


from detector import build_detector
from deep_sort import build_tracker
from track_utils.draw import draw_boxes
from track_utils.parser import get_config
from track_utils.log import get_logger
from PIL import Image


class PersonTracker(object):
    def __init__(self, cfg, args):
        self.args = args
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        self.detector = build_detector(cfg, use_cuda=use_cuda, segment=self.args.segment)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            # TODO save masks

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        with open('coco_classes.json', 'r') as f:
            idx_to_class = json.load(f)
        image_folder = self.args.save_path

        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith(".jpg")],
            key=lambda x: int(x.split(".")[0])
        )

        idx_frame = -1
        for image_file in image_files:
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()

            image_path = os.path.join(image_folder, image_file)
            pil_image = Image.open(image_path)
            pil_image = pil_image.convert("RGB")

            ori_im = cv2.cvtColor(
                np.array(pil_image),
                cv2.COLOR_RGB2BGR
                )
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            if self.args.segment:
                bbox_xywh, cls_conf, cls_ids, seg_masks = self.detector(im)
                
                # Handle case when seg_masks is 2D (H, W) instead of expected 3D (N, H, W)
                if len(seg_masks.shape) == 2:
                    seg_masks = seg_masks.reshape(1, *seg_masks.shape)  # Add batch dimension
                
                # select person class (modify mask creation)
                mask = cls_ids == 0
                
                # Check if any persons detected
                if not mask.any():
                    # Generate empty CSV for this frame
                    csv_name = f"{idx_frame}.csv"
                    csv_path = os.path.join(self.args.save_path, csv_name)
                    with open(csv_path, mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([])  # Just write an empty row
                    
                    end = time.time()
                    self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: 0, tracking numbers: 0" \
                                     .format(end - start, 1 / (end - start)))
                    continue

                # Apply mask to all arrays
                bbox_xywh = bbox_xywh[mask]
                cls_conf = cls_conf[mask]
                cls_ids = cls_ids[mask]
                
                # Ensure mask has correct shape for indexing seg_masks
                mask_idx = np.where(mask)[0]  # Get indices where mask is True
                seg_masks = seg_masks[mask_idx]

            else:
                bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # Handle case when no detections at all
            if len(cls_ids) == 0:
                # Generate empty CSV for this frame
                csv_name = f"{idx_frame}.csv"
                csv_path = os.path.join(self.args.save_path, csv_name)
                with open(csv_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([])  # Just write an empty row
                
                end = time.time()
                self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: 0, tracking numbers: 0" \
                                 .format(end - start, 1 / (end - start)))
                continue

            # select person class
            mask = cls_ids == 0
            bbox_xywh = bbox_xywh[mask]
            cls_conf = cls_conf[mask]
            cls_ids = cls_ids[mask]

            # If no persons detected after filtering
            if len(cls_ids) == 0:
                # Generate empty CSV for this frame
                csv_name = f"{idx_frame}.csv"
                csv_path = os.path.join(self.args.save_path, csv_name)
                with open(csv_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([])  # Just write an empty row
                
                end = time.time()
                self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: 0, tracking numbers: 0" \
                                 .format(end - start, 1 / (end - start)))
                continue

            # Continue with normal processing only if persons were detected
            if self.args.segment:
                seg_masks = seg_masks[mask]
                outputs, mask_outputs = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im, seg_masks)
            else:
                outputs, _ = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)

            # draw boxes for visualization
            if idx_frame in [0, 1]:

                columns = []
                rows = list(zip(*columns))
                csv_name = f"{idx_frame}.csv"
                csv_path = os.path.join(self.args.save_path, csv_name)
                with open(csv_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(rows)
                print(f"CSV file '{csv_name}' has been generated successfully.")

            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                cls = outputs[:, -2]
                names = [idx_to_class[str(label)] for label in cls]

                ori_im = draw_boxes(ori_im, bbox_xyxy, names, identities, None if not self.args.segment else mask_outputs)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities, cls))

                # csv
                columns = []
                H, W = mask_outputs[0].shape

                for i, identity in enumerate(identities):
                    column = [identity]
                    for x in range(H):
                        for y in range(W):
                            column.append(1 if mask_outputs[i][x][y]>0.7 else 0)
                    columns.append(column)

                rows = list(zip(*columns))

                csv_name = f"{idx_frame}.csv"
                csv_path = os.path.join(self.args.save_path, csv_name)
                with open(csv_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(rows)

                print(f"CSV file '{csv_name}' has been generated successfully.")

            end = time.time()

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/mask_rcnn.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--no_segment", dest='segment', action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="./data/jxl_s_10")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    return parser.parse_args()

def process_folders(base_path, cfg, args):
    """Recursively process all image folders"""
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            print(f"Processing folder: {item_path}")
            # Create a new args object for this folder
            folder_args = argparse.Namespace(**vars(args))
            folder_args.save_path = item_path  # Set save path to current folder
            
            # Process this folder
            with PersonTracker(cfg, folder_args) as vdo_trk:
                vdo_trk.run()
            
            # Recursively process subfolders
            process_folders(item_path, cfg, args)

# Modify main
if __name__ == "__main__":
    args = parse_args()
    
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    # Start recursive processing from save_path
    process_folders(args.save_path, cfg, args)