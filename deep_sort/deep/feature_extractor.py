"""ReID feature extractor used by the DeepSORT wrapper."""

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net


class Extractor(object):
    """Load the ReID network and produce appearance embeddings for crops."""

    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        # self.net = resnet18(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict if 'net_dict' not in state_dict else state_dict['net_dict'], strict=False)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """Resize, scale, tensorize, and normalize crops for the ReID network."""

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        """Return one appearance embedding per input crop."""
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()
