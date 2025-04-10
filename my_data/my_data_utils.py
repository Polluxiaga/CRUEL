import numpy as np
import os
from PIL import Image
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
import io
from typing import Union

VISUALIZATION_IMAGE_SIZE = (224, 128)
IMAGE_ASPECT_RATIO = (
    7 / 4
)  # all images are centered cropped to a 7:4 aspect ratio in training


def ts2np(tensor):
    """Convert tensor, list of tensors, or numpy array to numpy array.
    
    Args:
        tensor: torch.Tensor, list of torch.Tensor, or np.ndarray
        
    Returns:
        np.ndarray: Converted numpy array
    """
    if isinstance(tensor, list):
        return [ts2np(t) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f"Unsupported type for ts2np: {type(tensor)}")


def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    data_ext = {
        "image": ".jpg",
        # add more data types here
    }
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")


def img_path_to_data(
    path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
    ) -> torch.Tensor:
    """
    Load an image from a path and crop it and transform it to a tensor
    Args:
        path (str): path to the image
        image_resize_size (Tuple[int, int]): size to resize the image to
    Returns:
        torch.Tensor: resized image as tensor
    """
    img = Image.open(path)
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img