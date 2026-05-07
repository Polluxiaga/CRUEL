"""Run Mask R-CNN + DeepSORT tracking and export per-frame person mask CSVs."""

import os
import cv2
import time
import argparse
import torch
import warnings
import json
import numpy as np
import csv
from PIL import Image

from detector import build_detector
from deep_sort import build_tracker
from tracking_utils.draw import draw_boxes
from tracking_utils.parser import get_config
from tracking_utils.log import get_logger

class PersonTracker(object):
    """Track pedestrians in trajectory image folders and write flattened masks."""
    def __init__(self, cfg, args):
        self.args = args
        self.logger = get_logger("root")

        # Hardware acceleration setup
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which may be very slow!", UserWarning)

        # Initialize detector and tracker
        self.detector = build_detector(cfg, use_cuda=use_cuda, segment=self.args.segment)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            self.logger.info(f"Save results to {self.args.save_path}")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def _generate_empty_csv(self, frame_idx):
        """Helper to create an empty CSV file for frames with no relevant detections."""
        csv_path = os.path.join(self.args.save_path, f"{frame_idx}.csv")
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([])

    def run(self):
        results = []

        # Load class mapping
        with open('coco_classes.json', 'r') as f:
            idx_to_class = json.load(f)

        image_folder = self.args.save_path
        # Gather and sort image files numerically
        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith(".jpg")],
            key=lambda x: int(x.split(".")[0])
        )

        for idx_frame, image_file in enumerate(image_files):
            # Frame skipping logic
            if idx_frame % self.args.frame_interval != 0:
                continue

            start_time = time.time()

            # Load and convert image
            image_path = os.path.join(image_folder, image_file)
            pil_image = Image.open(image_path).convert("RGB")

            # Convert for OpenCV (BGR) and Tracker (RGB)
            ori_im = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # Step 1: Detection
            if self.args.segment:
                bbox_xywh, cls_conf, cls_ids, seg_masks = self.detector(im)
                # Ensure seg_masks is 3D (N, H, W)
                if seg_masks is not None and len(seg_masks.shape) == 2:
                    seg_masks = seg_masks.reshape(1, *seg_masks.shape)
            else:
                bbox_xywh, cls_conf, cls_ids = self.detector(im)
                seg_masks = None

            # Step 2: Filter for 'person' class (ID 0)
            mask = (cls_ids == 0)

            # Check if any person was detected
            if len(cls_ids) == 0 or not mask.any():
                self._generate_empty_csv(idx_frame)
                end_time = time.time()
                self.logger.info(f"frame {idx_frame} - time: {end_time-start_time:.3f}s, "
                                 f"fps: {1/(end_time-start_time):.3f}, detections: 0")
                continue

            # Apply filter mask to all detection arrays
            bbox_xywh = bbox_xywh[mask]
            cls_conf = cls_conf[mask]
            cls_ids = cls_ids[mask]
            if seg_masks is not None:
                seg_masks = seg_masks[mask]

            # Step 3: Tracking update
            if self.args.segment:
                outputs, mask_outputs = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im, seg_masks)
            else:
                outputs, _ = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)
                mask_outputs = None

            # Specific handling for frames 0 and 1 as per original requirement
            if idx_frame in [0, 1]:
                self._generate_empty_csv(idx_frame)
                print(f"CSV file '{idx_frame}.csv' has been generated successfully.")

            # Step 4: Process tracking results
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                cls_indices = outputs[:, -2]
                names = [idx_to_class[str(label)] for label in cls_indices]

                # Draw visualization
                ori_im = draw_boxes(ori_im, bbox_xyxy, names, identities,
                                    mask_outputs if self.args.segment else None)

                # Store tracking results
                bbox_tlwh = [self.deepsort._xyxy_to_tlwh(x) for x in bbox_xyxy]
                results.append((idx_frame - 1, bbox_tlwh, identities, cls_indices))

                # Generate binary mask CSV. Downstream Gaze2Nav steps require
                # segmentation masks, so detection-only runs emit an empty CSV.
                if not self.args.segment or mask_outputs is None:
                    self._generate_empty_csv(idx_frame)
                    print(f"CSV file '{idx_frame}.csv' has been generated successfully.")
                else:
                    # Each column represents a tracked ID, each row a pixel state.
                    columns = []
                    for i, identity in enumerate(identities):
                        # Flatten mask and threshold at 0.7 (exact logic as original loop)
                        mask_flat = (mask_outputs[i].flatten() > 0.7).astype(int)
                        column = [identity] + mask_flat.tolist()
                        columns.append(column)

                    # Transpose columns to rows for CSV writing
                    rows = list(zip(*columns))
                    csv_path = os.path.join(self.args.save_path, f"{idx_frame}.csv")
                    with open(csv_path, mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(rows)
                    print(f"CSV file '{idx_frame}.csv' has been generated successfully.")
            else:
                self._generate_empty_csv(idx_frame)
                print(f"CSV file '{idx_frame}.csv' has been generated successfully.")

            # Log performance metrics
            end_time = time.time()
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}"
                             .format(end_time - start_time, 1 / (end_time - start_time),
                                     bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/mask_rcnn.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--no_segment", dest='segment', action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="./data")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()


def process_folders(base_path, cfg, args):
    """Recursively search for and process folders containing images."""
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            print(f"Processing folder: {item_path}")

            # Create isolated args for this specific subdirectory
            folder_args = argparse.Namespace(**vars(args))
            folder_args.save_path = item_path

            # Initialize tracker within context manager
            with PersonTracker(cfg, folder_args) as vdo_trk:
                vdo_trk.run()

            # Recurse into subfolders
            process_folders(item_path, cfg, args)


if __name__ == "__main__":
    args = parse_args()

    # Load and merge configurations
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    # Start processing from the root save_path
    process_folders(args.save_path, cfg, args)
