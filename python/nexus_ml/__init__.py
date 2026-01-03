"""
Nexus ML - Python-based ML extensions for Nexus Engine
Provides image processing, ML inference, and preprocessing utilities
"""

from .image_nodes import (
    ImageResizeNode,
    ImageNormalizeNode,
    ColorSpaceNode,
    BoundingBoxNode,
    ResizeMethod,
    ColorSpace,
    ImageMetadata,
    TensorMetadata,
    BoundingBox
)

from .pipelines import (
    ImageNetPipeline,
    YOLOPipeline,
    SegmentationPipeline,
    GrayscalePipeline,
    BatchProcessor
)

__all__ = [
    # Image nodes
    'ImageResizeNode',
    'ImageNormalizeNode', 
    'ColorSpaceNode',
    'BoundingBoxNode',
    # Enums and metadata
    'ResizeMethod',
    'ColorSpace',
    'ImageMetadata',
    'TensorMetadata',
    'BoundingBox',
    # Pipelines
    'ImageNetPipeline',
    'YOLOPipeline',
    'SegmentationPipeline',
    'GrayscalePipeline',
    'BatchProcessor'
]
