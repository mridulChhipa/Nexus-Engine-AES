"""
Pre-built ML pipelines for common tasks
"""

import numpy as np
from typing import List, Tuple, Optional
from .image_nodes import (
    ImageResizeNode, 
    ImageNormalizeNode, 
    ColorSpaceNode,
    ResizeMethod,
    ColorSpace,
    ImageMetadata,
    TensorMetadata
)


class ImageNetPipeline:
    """
    Standard ImageNet preprocessing pipeline
    - Resize to 224x224
    - RGB format
    - Normalize with ImageNet mean/std
    """
    
    def __init__(self, input_size: int = 224):
        self.resize = ImageResizeNode("resize", input_size, input_size, ResizeMethod.BILINEAR)
        self.normalize = ImageNormalizeNode(
            "normalize",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            scale=1.0/255.0
        )
        self.input_size = input_size
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, TensorMetadata]:
        """Process image through pipeline"""
        # Resize
        resized, _ = self.resize.process(image)
        
        # Normalize
        tensor, meta = self.normalize.process(resized)
        
        return tensor, meta


class YOLOPipeline:
    """
    YOLO object detection preprocessing pipeline
    - Letterbox resize to square
    - RGB format
    - Scale to [0, 1]
    """
    
    def __init__(self, input_size: int = 640):
        self.input_size = input_size
        self.name = "YOLOPipeline"
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Process image through pipeline"""
        import cv2
        from .preprocessing import letterbox_resize
        
        # Letterbox resize
        resized, scale, (pad_top, pad_left) = letterbox_resize(
            image, 
            (self.input_size, self.input_size),
            fill_value=114
        )
        
        # Convert to float and scale [0, 255] -> [0, 1]
        tensor = resized.astype(np.float32) / 255.0
        
        # HWC to CHW
        if len(tensor.shape) == 3:
            tensor = np.transpose(tensor, (2, 0, 1))
        
        # Add batch dimension
        tensor = np.expand_dims(tensor, 0)
        
        metadata = {
            'shape': tensor.shape,
            'scale': scale,
            'pad_top': pad_top,
            'pad_left': pad_left,
            'original_shape': image.shape
        }
        
        return tensor, metadata


class SegmentationPipeline:
    """
    Semantic segmentation preprocessing pipeline
    - Resize
    - Normalize
    - Convert to tensor
    """
    
    def __init__(self, input_size: Tuple[int, int] = (512, 512),
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):
        self.resize = ImageResizeNode("resize", input_size[0], input_size[1])
        self.normalize = ImageNormalizeNode("normalize", mean=mean, std=std)
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, TensorMetadata]:
        """Process image through pipeline"""
        resized, _ = self.resize.process(image)
        tensor, meta = self.normalize.process(resized)
        return tensor, meta


class GrayscalePipeline:
    """
    Grayscale image preprocessing pipeline
    - Convert to grayscale
    - Resize
    - Normalize
    """
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        self.colorspace = ColorSpaceNode("to_gray", ColorSpace.RGB, ColorSpace.GRAY)
        self.resize = ImageResizeNode("resize", input_size[0], input_size[1])
        self.normalize = ImageNormalizeNode(
            "normalize",
            mean=[0.5],
            std=[0.5],
            scale=1.0/255.0
        )
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, TensorMetadata]:
        """Process image through pipeline"""
        gray, _ = self.colorspace.process(image)
        resized, _ = self.resize.process(gray)
        tensor, meta = self.normalize.process(resized)
        return tensor, meta


class BatchProcessor:
    """
    Process multiple images through a pipeline
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def __call__(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List]:
        """
        Process batch of images
        
        Args:
            images: List of images
        
        Returns:
            Tuple of (batched_tensor, list_of_metadata)
        """
        tensors = []
        metadatas = []
        
        for img in images:
            tensor, meta = self.pipeline(img)
            tensors.append(tensor)
            metadatas.append(meta)
        
        # Stack tensors into batch
        batched = np.concatenate(tensors, axis=0)
        
        return batched, metadatas


# Example usage
if __name__ == "__main__":
    print("Testing pre-built pipelines...")
    
    # Create test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # ImageNet pipeline
    imagenet = ImageNetPipeline(224)
    tensor, meta = imagenet(test_img)
    print(f"ImageNet output: {tensor.shape}, {meta}")
    
    # YOLO pipeline
    yolo = YOLOPipeline(640)
    tensor, meta = yolo(test_img)
    print(f"YOLO output: {tensor.shape}, metadata: {meta}")
    
    # Batch processing
    batch_processor = BatchProcessor(ImageNetPipeline(224))
    images = [test_img, test_img, test_img]
    batched, metas = batch_processor(images)
    print(f"Batch output: {batched.shape}")
