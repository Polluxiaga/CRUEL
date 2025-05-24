from .maskrcnn import Mask_RCNN

__all__ = ['build_detector']


def build_detector(cfg, use_cuda, segment=False):
    return Mask_RCNN(segment, num_classes=cfg.MASKRCNN.NUM_CLASSES, box_thresh=cfg.MASKRCNN.BOX_THRESH,
                     label_json_path=cfg.MASKRCNN.LABEL, weight_path=cfg.MASKRCNN.WEIGHT)