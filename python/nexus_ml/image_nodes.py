"""
Image Processing Nodes using OpenCV in Python
Compatible with Nexus Engine RawBuffer through NumPy
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from enum import Enum


class ResizeMethod(Enum):
    """Image resize interpolation methods"""
    NEAREST = cv2.INTER_NEAREST      # Fast, blocky
    BILINEAR = cv2.INTER_LINEAR      # Balanced (default)
    BICUBIC = cv2.INTER_CUBIC        # Smooth, slower
    AREA = cv2.INTER_AREA            # Best for downscaling
    LANCZOS = cv2.INTER_LANCZOS4     # High quality


class ColorSpace(Enum):
    """Color space conversion options"""
    RGB = "RGB"
    BGR = "BGR"
    GRAY = "GRAY"
    HSV = "HSV"
    YUV = "YUV"
    LAB = "LAB"


class ImageMetadata:
    """Image metadata structure matching C++ ImageMetaData"""
    def __init__(self, width: int, height: int, channels: int, format: str = "RGB"):
        self.width = width
        self.height = height
        self.channels = channels
        self.format = format
    
    def total_size(self) -> int:
        return self.width * self.height * self.channels
    
    def __repr__(self):
        return f"ImageMetadata({self.width}x{self.height}x{self.channels}, {self.format})"


class TensorMetadata:
    """Tensor metadata for normalized data"""
    def __init__(self, shape: List[int], dtype: str = "float32", layout: str = "NCHW"):
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
    
    def total_elements(self) -> int:
        return int(np.prod(self.shape))
    
    def __repr__(self):
        return f"TensorMetadata(shape={self.shape}, dtype={self.dtype}, layout={self.layout})"


class BoundingBox:
    """Bounding box with confidence and class"""
    def __init__(self, x: float, y: float, width: float, height: float, 
                 confidence: float, class_id: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.class_id = class_id
    
    def area(self) -> float:
        return self.width * self.height
    
    def __repr__(self):
        return (f"BBox(x={self.x:.1f}, y={self.y:.1f}, w={self.width:.1f}, h={self.height:.1f}, "
                f"conf={self.confidence:.2f}, cls={self.class_id})")


class ImageResizeNode:
    """
    Resize images to target dimensions
    
    Args:
        name: Node name
        width: Target width
        height: Target height
        method: Resize interpolation method
    """
    
    def __init__(self, name: str, width: int, height: int, 
                 method: ResizeMethod = ResizeMethod.BILINEAR):
        self.name = name
        self.target_width = width
        self.target_height = height
        self.method = method
        self.node_id = id(self)
        print(f"Node [{name}] created (ImageResize)")
    
    def process(self, image: np.ndarray, metadata: Optional[ImageMetadata] = None) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Resize image to target dimensions
        
        Args:
            image: Input image as numpy array (HxWxC or HxW)
            metadata: Optional image metadata
        
        Returns:
            Tuple of (resized_image, updated_metadata)
        """
        if metadata is None:
            # Infer metadata from image
            if len(image.shape) == 3:
                h, w, c = image.shape
                metadata = ImageMetadata(w, h, c, "RGB")
            else:
                h, w = image.shape
                metadata = ImageMetadata(w, h, 1, "GRAY")
        
        # Resize
        resized = cv2.resize(image, (self.target_width, self.target_height), 
                           interpolation=self.method.value)
        
        # Update metadata
        output_meta = ImageMetadata(
            self.target_width, 
            self.target_height,
            metadata.channels,
            metadata.format
        )
        
        print(f"[{self.name}] Resized from {metadata.width}x{metadata.height} to "
              f"{self.target_width}x{self.target_height}")
        
        return resized, output_meta


class ImageNormalizeNode:
    """
    Normalize image for neural network input
    Converts to float, scales, and applies mean/std normalization
    
    Args:
        name: Node name
        mean: Per-channel mean values (e.g., ImageNet: [0.485, 0.456, 0.406])
        std: Per-channel std values (e.g., ImageNet: [0.229, 0.224, 0.225])
        scale: Scaling factor (default: 1/255.0 to convert [0,255] to [0,1])
    """
    
    def __init__(self, name: str,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 scale: float = 1.0/255.0):
        self.name = name
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.scale = scale
        self.node_id = id(self)
        print(f"Node [{name}] created (ImageNormalize)")
    
    def process(self, image: np.ndarray, metadata: Optional[ImageMetadata] = None) -> Tuple[np.ndarray, TensorMetadata]:
        """
        Normalize image to tensor format
        
        Args:
            image: Input image as uint8 numpy array
            metadata: Optional image metadata
        
        Returns:
            Tuple of (normalized_tensor, tensor_metadata)
        """
        # Convert to float and scale [0, 255] -> [0, 1]
        float_img = image.astype(np.float32) * self.scale
        
        # Apply normalization: (x - mean) / std
        if len(float_img.shape) == 3 and float_img.shape[2] == 3:
            # RGB/BGR image
            normalized = (float_img - self.mean) / self.std
        elif len(float_img.shape) == 2 or float_img.shape[2] == 1:
            # Grayscale
            normalized = (float_img - self.mean[0]) / self.std[0]
        else:
            normalized = float_img
        
        # Convert to NCHW format (batch, channels, height, width)
        if len(normalized.shape) == 3:
            # HWC -> CHW
            normalized = np.transpose(normalized, (2, 0, 1))
            c, h, w = normalized.shape
        else:
            # HW -> 1HW
            h, w = normalized.shape
            normalized = np.expand_dims(normalized, 0)
            c = 1
        
        # Add batch dimension: CHW -> NCHW
        normalized = np.expand_dims(normalized, 0)
        
        # Create tensor metadata
        tensor_meta = TensorMetadata(
            shape=[1, c, h, w],
            dtype="float32",
            layout="NCHW"
        )
        
        print(f"[{self.name}] Normalized to tensor shape {normalized.shape}")
        
        return normalized, tensor_meta


class ColorSpaceNode:
    """
    Convert between color spaces
    
    Args:
        name: Node name
        source: Source color space
        target: Target color space
    """
    
    def __init__(self, name: str, source: ColorSpace, target: ColorSpace):
        self.name = name
        self.source = source
        self.target = target
        self.node_id = id(self)
        print(f"Node [{name}] created (ColorSpace: {source.value} -> {target.value})")
    
    def _get_conversion_code(self) -> Optional[int]:
        """Get OpenCV conversion code"""
        conversions = {
            (ColorSpace.RGB, ColorSpace.BGR): cv2.COLOR_RGB2BGR,
            (ColorSpace.RGB, ColorSpace.GRAY): cv2.COLOR_RGB2GRAY,
            (ColorSpace.RGB, ColorSpace.HSV): cv2.COLOR_RGB2HSV,
            (ColorSpace.RGB, ColorSpace.YUV): cv2.COLOR_RGB2YUV,
            (ColorSpace.RGB, ColorSpace.LAB): cv2.COLOR_RGB2Lab,
            
            (ColorSpace.BGR, ColorSpace.RGB): cv2.COLOR_BGR2RGB,
            (ColorSpace.BGR, ColorSpace.GRAY): cv2.COLOR_BGR2GRAY,
            (ColorSpace.BGR, ColorSpace.HSV): cv2.COLOR_BGR2HSV,
            (ColorSpace.BGR, ColorSpace.YUV): cv2.COLOR_BGR2YUV,
            (ColorSpace.BGR, ColorSpace.LAB): cv2.COLOR_BGR2Lab,
            
            (ColorSpace.GRAY, ColorSpace.RGB): cv2.COLOR_GRAY2RGB,
            (ColorSpace.GRAY, ColorSpace.BGR): cv2.COLOR_GRAY2BGR,
            
            (ColorSpace.HSV, ColorSpace.RGB): cv2.COLOR_HSV2RGB,
            (ColorSpace.HSV, ColorSpace.BGR): cv2.COLOR_HSV2BGR,
            
            (ColorSpace.YUV, ColorSpace.RGB): cv2.COLOR_YUV2RGB,
            (ColorSpace.YUV, ColorSpace.BGR): cv2.COLOR_YUV2BGR,
            
            (ColorSpace.LAB, ColorSpace.RGB): cv2.COLOR_Lab2RGB,
            (ColorSpace.LAB, ColorSpace.BGR): cv2.COLOR_Lab2BGR,
        }
        return conversions.get((self.source, self.target))
    
    def process(self, image: np.ndarray, metadata: Optional[ImageMetadata] = None) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Convert color space
        
        Args:
            image: Input image
            metadata: Optional image metadata
        
        Returns:
            Tuple of (converted_image, updated_metadata)
        """
        # No conversion needed
        if self.source == self.target:
            return image, metadata
        
        conversion_code = self._get_conversion_code()
        if conversion_code is None:
            raise ValueError(f"Unsupported conversion: {self.source.value} -> {self.target.value}")
        
        # Convert
        converted = cv2.cvtColor(image, conversion_code)
        
        # Update metadata
        if metadata is None:
            h, w = image.shape[:2]
            channels = 1 if len(converted.shape) == 2 else converted.shape[2]
            metadata = ImageMetadata(w, h, channels, self.source.value)
        
        output_channels = 1 if len(converted.shape) == 2 else converted.shape[2]
        output_meta = ImageMetadata(
            metadata.width,
            metadata.height,
            output_channels,
            self.target.value
        )
        
        print(f"[{self.name}] Converted {self.source.value} -> {self.target.value}")
        
        return converted, output_meta


class BoundingBoxNode:
    """
    Filter and process bounding boxes with Non-Maximum Suppression
    
    Args:
        name: Node name
        conf_threshold: Minimum confidence to keep box (default: 0.5)
        nms_threshold: IoU threshold for NMS (default: 0.4)
    """
    
    def __init__(self, name: str, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.name = name
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.node_id = id(self)
        print(f"Node [{name}] created (BoundingBox NMS)")
    
    @staticmethod
    def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = box1.area() + box2.area() - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def apply_nms(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Apply Non-Maximum Suppression"""
        if not boxes:
            return []
        
        # Sort by confidence (descending)
        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, box in enumerate(boxes):
            if i in suppressed:
                continue
            
            keep.append(box)
            
            # Suppress overlapping boxes of the same class
            for j in range(i + 1, len(boxes)):
                if j in suppressed:
                    continue
                
                if box.class_id == boxes[j].class_id:
                    iou = self.calculate_iou(box, boxes[j])
                    if iou > self.nms_threshold:
                        suppressed.add(j)
        
        return keep
    
    def process(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Filter boxes by confidence and apply NMS
        
        Args:
            boxes: List of bounding boxes
        
        Returns:
            Filtered list of bounding boxes
        """
        # Filter by confidence
        filtered = [box for box in boxes if box.confidence >= self.conf_threshold]
        
        print(f"[{self.name}] Filtered {len(boxes)} boxes -> {len(filtered)} above threshold")
        
        # Apply NMS
        result = self.apply_nms(filtered)
        
        print(f"[{self.name}] After NMS: {len(result)} boxes kept")
        
        return result
    
    @staticmethod
    def draw_boxes(image: np.ndarray, boxes: List[BoundingBox], 
                   class_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            boxes: List of bounding boxes to draw
            class_names: Optional list of class names
        
        Returns:
            Image with drawn boxes
        """
        output = image.copy()
        
        for box in boxes:
            # Draw rectangle
            x1, y1 = int(box.x), int(box.y)
            x2, y2 = int(box.x + box.width), int(box.y + box.height)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Class {box.class_id}" if class_names is None else class_names[box.class_id]
            label += f" {box.confidence:.2f}"
            
            cv2.putText(output, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output
