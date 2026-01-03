# Nexus ML - Python ML Extensions

Python-based machine learning and image processing extensions for Nexus Engine.

## Quick Start

### Installation

```bash
cd python
pip install -r requirements.txt
```

### Run Interactive CLI

```bash
python interactive_cli.py
```

## Features

### 🎨 Interactive Command-Line Interface
- Colorful menu-driven interface
- Load and process images
- Multiple ML preprocessing pipelines
- Batch processing support
- Save processed results

### 🔧 Image Processing Nodes

**4 Core Nodes:**
1. **ImageResizeNode** - 5 interpolation methods (Nearest, Bilinear, Bicubic, Area, Lanczos)
2. **ImageNormalizeNode** - Neural network preprocessing with mean/std normalization
3. **ColorSpaceNode** - Convert between RGB, BGR, GRAY, HSV, YUV, LAB
4. **BoundingBoxNode** - NMS filtering, IoU calculation, confidence thresholding

### 📦 Pre-built Pipelines

- **ImageNetPipeline** - ResNet/VGG/MobileNet preprocessing (224×224, normalized)
- **YOLOPipeline** - Object detection preprocessing (640×640, letterbox)
- **GrayscalePipeline** - Grayscale conversion and normalization
- **BatchProcessor** - Process multiple images

### 🛠️ CLI Menu Options

- **1** - Load Image (browse or enter path)
- **2** - ImageNet Pipeline (classification models)
- **3** - YOLO Pipeline (object detection)
- **4** - Grayscale Conversion
- **5** - Custom Processing (edge detection, color transform, etc.)
- **6** - Batch Process Directory
- **7** - View Image Info
- **8** - Save Processed Image
- **9** - View Processing History
- **0** - Exit

## Usage Examples

### Basic Image Processing

```python
import numpy as np
from nexus_ml import ImageResizeNode, ImageNormalizeNode

# Load image
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Resize
resize_node = ImageResizeNode("resize", 224, 224)
resized, metadata = resize_node.process(image)

# Normalize
normalize_node = ImageNormalizeNode("normalize")
tensor, tensor_meta = normalize_node.process(resized)

print(f"Output tensor: {tensor.shape}")  # (1, 3, 224, 224)
```

### Using Pre-built Pipelines

```python
from nexus_ml import ImageNetPipeline, YOLOPipeline

# ImageNet preprocessing
imagenet = ImageNetPipeline(224)
tensor, meta = imagenet(image)

# YOLO preprocessing
yolo = YOLOPipeline(640)
tensor, meta = yolo(image)
```

### Batch Processing

```python
from nexus_ml import ImageNetPipeline, BatchProcessor

pipeline = ImageNetPipeline(224)
batch_processor = BatchProcessor(pipeline)

images = [image1, image2, image3]
batched_tensor, metadatas = batch_processor(images)

print(f"Batch shape: {batched_tensor.shape}")  # (3, 3, 224, 224)
```

### Bounding Box Processing

```python
from nexus_ml import BoundingBoxNode, BoundingBox

# Create detections
boxes = [
    BoundingBox(50, 50, 100, 100, 0.95, 0),  # x, y, w, h, confidence, class
    BoundingBox(55, 55, 95, 95, 0.85, 0),    # Overlapping
]

# Apply NMS filtering
nms = BoundingBoxNode("nms", conf_threshold=0.5, nms_threshold=0.4)
filtered = nms.process(boxes)

# Draw on image
result = BoundingBoxNode.draw_boxes(image, filtered)
```

## Project Structure

```
python/
├── nexus_ml/              # Main package
│   ├── __init__.py        # Package exports
│   ├── image_nodes.py     # 4 image processing nodes
│   ├── preprocessing.py   # Utility functions
│   └── pipelines.py       # Pre-built pipelines
├── interactive_cli.py     # CLI application
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Dependencies

- `numpy` - Array operations
- `opencv-python` - Image processing
- `pillow` - Optional, for additional image format support

## Integration with C++ Nexus Engine

The Python ML nodes work alongside the C++ Nexus Engine:

**C++ Side (include/ml/):**
- `MLTypes.hpp` - Metadata structures (ImageMetaData, TensorMetaData)
- `MLNode.hpp` - Base class for ML nodes

**Python Side:**
- Image processing implementations
- ML preprocessing pipelines
- Can be called from C++ via pybind11 (future integration)

## Use Cases

✅ **Image Classification** - Preprocess for ResNet, VGG, MobileNet
✅ **Object Detection** - Prepare images for YOLO models
✅ **Medical Imaging** - Grayscale processing and normalization
✅ **Batch Processing** - Process entire directories
✅ **Custom Pipelines** - Chain multiple transformations

## Troubleshooting

**Import errors:**
```bash
pip install --upgrade numpy opencv-python
```

**Numpy MINGW warnings (Windows):**
```bash
pip install numpy --only-binary :all: --force-reinstall
```

**Path issues:**
- Use absolute paths: `C:\path\to\image.jpg`
- Or place images in the `python/` directory

---

For more details, see [DEMO_GUIDE.md](DEMO_GUIDE.md) or run `python interactive_cli.py`
