import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import io
from typing import Union


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
    path: Union[str, io.BytesIO]
) -> torch.Tensor:
    """
    Load an image from a path and transform it to a tensor.
    Args:
        path (str): path to the image
    Returns:
        torch.Tensor: image as tensor
    """
    img = Image.open(path)
    tensor = TF.to_tensor(img)
    return tensor