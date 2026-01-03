"""
Preprocessing utilities for ML pipelines
Common data transformations and helpers
"""

import numpy as np
from typing import Tuple, Optional, Union


def normalize_array(arr: np.ndarray, 
                    mean: Union[float, np.ndarray] = 0.0,
                    std: Union[float, np.ndarray] = 1.0) -> np.ndarray:
    """
    Normalize array with mean and standard deviation
    
    Args:
        arr: Input array
        mean: Mean value(s) to subtract
        std: Standard deviation to divide by
    
    Returns:
        Normalized array
    """
    return (arr - mean) / std


def denormalize_array(arr: np.ndarray,
                     mean: Union[float, np.ndarray] = 0.0,
                     std: Union[float, np.ndarray] = 1.0) -> np.ndarray:
    """
    Reverse normalization
    
    Args:
        arr: Normalized array
        mean: Mean value(s) that were subtracted
        std: Standard deviation that was divided by
    
    Returns:
        Denormalized array
    """
    return (arr * std) + mean


def to_channel_first(image: np.ndarray) -> np.ndarray:
    """
    Convert HWC to CHW format
    
    Args:
        image: Image in HWC format (Height, Width, Channels)
    
    Returns:
        Image in CHW format (Channels, Height, Width)
    """
    if len(image.shape) == 2:
        return image
    return np.transpose(image, (2, 0, 1))


def to_channel_last(image: np.ndarray) -> np.ndarray:
    """
    Convert CHW to HWC format
    
    Args:
        image: Image in CHW format (Channels, Height, Width)
    
    Returns:
        Image in HWC format (Height, Width, Channels)
    """
    if len(image.shape) == 2:
        return image
    return np.transpose(image, (1, 2, 0))


def pad_to_square(image: np.ndarray, 
                  fill_value: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to square shape
    
    Args:
        image: Input image
        fill_value: Value to use for padding
    
    Returns:
        Tuple of (padded_image, (pad_top, pad_left))
    """
    h, w = image.shape[:2]
    size = max(h, w)
    
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    
    if len(image.shape) == 3:
        padded = np.full((size, size, image.shape[2]), fill_value, dtype=image.dtype)
        padded[pad_h:pad_h+h, pad_w:pad_w+w, :] = image
    else:
        padded = np.full((size, size), fill_value, dtype=image.dtype)
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
    
    return padded, (pad_h, pad_w)


def letterbox_resize(image: np.ndarray, 
                    target_size: Tuple[int, int],
                    fill_value: int = 114) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image maintaining aspect ratio with padding (letterbox)
    
    Args:
        image: Input image
        target_size: Target (width, height)
        fill_value: Padding fill value
    
    Returns:
        Tuple of (resized_image, scale_factor, (pad_top, pad_left))
    """
    import cv2
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale to fit
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    if len(image.shape) == 3:
        padded = np.full((target_h, target_w, image.shape[2]), fill_value, dtype=image.dtype)
    else:
        padded = np.full((target_h, target_w), fill_value, dtype=image.dtype)
    
    # Place resized image in center
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    
    if len(image.shape) == 3:
        padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = resized
    else:
        padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    
    return padded, scale, (pad_top, pad_left)


def compute_stats(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std across dataset
    
    Args:
        images: Array of images (N, H, W, C) or (N, C, H, W)
    
    Returns:
        Tuple of (mean, std) per channel
    """
    if images.ndim == 4:
        # Assume NCHW or NHWC
        if images.shape[1] <= 4:  # NCHW
            axis = (0, 2, 3)
        else:  # NHWC
            axis = (0, 1, 2)
        
        mean = np.mean(images, axis=axis)
        std = np.std(images, axis=axis)
    else:
        mean = np.mean(images)
        std = np.std(images)
    
    return mean, std


def clip_boxes_to_image(boxes: np.ndarray, 
                       image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Clip bounding boxes to image boundaries
    
    Args:
        boxes: Array of boxes (N, 4) in format [x, y, w, h]
        image_shape: (height, width)
    
    Returns:
        Clipped boxes
    """
    h, w = image_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)  # x
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)  # y
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - boxes[:, 0])  # width
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - boxes[:, 1])  # height
    return boxes


def scale_boxes(boxes: np.ndarray, 
               scale_x: float, 
               scale_y: float) -> np.ndarray:
    """
    Scale bounding boxes
    
    Args:
        boxes: Array of boxes (N, 4) in format [x, y, w, h]
        scale_x: Horizontal scale factor
        scale_y: Vertical scale factor
    
    Returns:
        Scaled boxes
    """
    boxes[:, 0] *= scale_x  # x
    boxes[:, 1] *= scale_y  # y
    boxes[:, 2] *= scale_x  # width
    boxes[:, 3] *= scale_y  # height
    return boxes


def random_crop(image: np.ndarray, 
               crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Random crop from image
    
    Args:
        image: Input image
        crop_size: (height, width) of crop
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    if h <= crop_h or w <= crop_w:
        return image
    
    top = np.random.randint(0, h - crop_h)
    left = np.random.randint(0, w - crop_w)
    
    if len(image.shape) == 3:
        return image[top:top+crop_h, left:left+crop_w, :]
    else:
        return image[top:top+crop_h, left:left+crop_w]


def center_crop(image: np.ndarray, 
               crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Center crop from image
    
    Args:
        image: Input image
        crop_size: (height, width) of crop
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    if h <= crop_h or w <= crop_w:
        return image
    
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    
    if len(image.shape) == 3:
        return image[top:top+crop_h, left:left+crop_w, :]
    else:
        return image[top:top+crop_h, left:left+crop_w]
