"""
Interactive Secure Image Processing System - CLI Edition
A complete command-line interface showcasing Nexus Engine ML capabilities

Features:
- Interactive menu system
- Load and process images
- Multiple processing pipelines
- Visual text-based feedback
- Export results
- Batch processing support
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from nexus_ml import (
    ImageResizeNode, ImageNormalizeNode, ColorSpaceNode, BoundingBoxNode,
    ImageNetPipeline, YOLOPipeline, GrayscalePipeline,
    ResizeMethod, ColorSpace, BoundingBox
)


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class ImageProcessorCLI:
    def __init__(self):
        self.current_image = None
        self.current_image_path = None
        self.processed_image = None
        self.processing_history = []
        
    def print_header(self):
        """Print application header"""
        print("\n" + "="*70)
        print(f"{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════╗{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}║{Colors.END}   {Colors.BOLD}NEXUS ML - SECURE IMAGE PROCESSOR{Colors.END}                         {Colors.CYAN}║{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}║{Colors.END}   {Colors.GREEN}Interactive Command-Line Interface{Colors.END}                        {Colors.CYAN}║{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚═══════════════════════════════════════════════════════════════════╝{Colors.END}")
        print("="*70 + "\n")
    
    def print_menu(self):
        """Print main menu"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}━━━ MAIN MENU ━━━{Colors.END}")
        print(f"{Colors.GREEN}1.{Colors.END} Load Image")
        print(f"{Colors.GREEN}2.{Colors.END} Process with ImageNet Pipeline")
        print(f"{Colors.GREEN}3.{Colors.END} Process with YOLO Pipeline")
        print(f"{Colors.GREEN}4.{Colors.END} Process to Grayscale")
        print(f"{Colors.GREEN}5.{Colors.END} Apply Custom Pipeline")
        print(f"{Colors.GREEN}6.{Colors.END} Batch Process Directory")
        print(f"{Colors.GREEN}7.{Colors.END} View Image Info")
        print(f"{Colors.GREEN}8.{Colors.END} Save Processed Image")
        print(f"{Colors.GREEN}9.{Colors.END} View Processing History")
        print(f"{Colors.RED}0.{Colors.END} Exit")
        print(f"{Colors.BLUE}━━━━━━━━━━━━━━━{Colors.END}")
    
    def print_status(self, message, status="info"):
        """Print colored status message"""
        color = {
            "info": Colors.CYAN,
            "success": Colors.GREEN,
            "warning": Colors.YELLOW,
            "error": Colors.RED
        }.get(status, Colors.END)
        
        symbol = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✗"
        }.get(status, "•")
        
        print(f"\n{color}{Colors.BOLD}[{symbol}]{Colors.END} {message}")
    
    def print_box(self, title, content, width=68):
        """Print content in a box"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}┌{'─' * (width-2)}┐{Colors.END}")
        print(f"{Colors.CYAN}│{Colors.END} {Colors.BOLD}{title}{Colors.END}{' ' * (width - len(title) - 3)}{Colors.CYAN}│{Colors.END}")
        print(f"{Colors.CYAN}├{'─' * (width-2)}┤{Colors.END}")
        
        for line in content.split('\n'):
            if line:
                padding = width - len(line) - 3
                print(f"{Colors.CYAN}│{Colors.END} {line}{' ' * padding}{Colors.CYAN}│{Colors.END}")
        
        print(f"{Colors.CYAN}└{'─' * (width-2)}┘{Colors.END}")
    
    def load_image(self):
        """Load an image from file"""
        print(f"\n{Colors.YELLOW}Enter image path (or press Enter to browse):{Colors.END}")
        path = input("> ").strip()
        
        if not path:
            # List current directory images
            print(f"\n{Colors.CYAN}Images in current directory:{Colors.END}")
            images = list(Path('.').glob('*.jpg')) + list(Path('.').glob('*.png')) + \
                     list(Path('.').glob('*.jpeg')) + list(Path('.').glob('*.bmp'))
            
            if not images:
                self.print_status("No images found in current directory", "warning")
                return
            
            for i, img in enumerate(images[:20], 1):
                print(f"  {i}. {img.name}")
            
            print(f"\n{Colors.YELLOW}Enter number or full path:{Colors.END}")
            choice = input("> ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(images):
                path = str(images[int(choice) - 1])
            else:
                path = choice
        
        try:
            # Load image
            image = cv2.imread(path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.current_image = image
            self.current_image_path = path
            
            h, w = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            
            info = (f"Filename: {Path(path).name}\n"
                   f"Size: {w} x {h} pixels\n"
                   f"Channels: {channels}\n"
                   f"Dtype: {image.dtype}\n"
                   f"Size: {os.path.getsize(path) / 1024:.1f} KB")
            
            self.print_box("IMAGE LOADED", info)
            self.print_status(f"Loaded: {Path(path).name}", "success")
            
        except Exception as e:
            self.print_status(f"Failed to load image: {str(e)}", "error")
    
    def process_imagenet(self):
        """Process with ImageNet pipeline"""
        if not self._check_image_loaded():
            return
        
        self.print_status("Processing with ImageNet pipeline...", "info")
        
        try:
            pipeline = ImageNetPipeline(224)
            tensor, meta = pipeline(self.current_image)
            
            # Store processed (convert back for saving)
            tensor_chw = tensor[0]
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            denorm = (tensor_chw * std) + mean
            denorm = np.clip(denorm * 255, 0, 255).astype(np.uint8)
            self.processed_image = np.transpose(denorm, (1, 2, 0))
            
            info = (f"Pipeline: ImageNet Classification\n"
                   f"Output Shape: {meta.shape}\n"
                   f"Dtype: {meta.dtype}\n"
                   f"Layout: {meta.layout}\n"
                   f"\n"
                   f"Operations Applied:\n"
                   f"  1. Resized to 224x224\n"
                   f"  2. Normalized (ImageNet stats)\n"
                   f"  3. Converted to NCHW tensor\n"
                   f"\n"
                   f"Ready for: ResNet, VGG, MobileNet")
            
            self.print_box("IMAGENET PROCESSING COMPLETE", info)
            self.processing_history.append(("ImageNet", meta.shape))
            self.print_status("Processing complete!", "success")
            
        except Exception as e:
            self.print_status(f"Processing failed: {str(e)}", "error")
    
    def process_yolo(self):
        """Process with YOLO pipeline"""
        if not self._check_image_loaded():
            return
        
        self.print_status("Processing with YOLO pipeline...", "info")
        
        try:
            pipeline = YOLOPipeline(640)
            tensor, meta = pipeline(self.current_image)
            
            # Store processed
            tensor_chw = tensor[0]
            image_hwc = np.transpose(tensor_chw, (1, 2, 0))
            self.processed_image = np.clip(image_hwc * 255, 0, 255).astype(np.uint8)
            
            info = (f"Pipeline: YOLO Object Detection\n"
                   f"Output Shape: {meta['shape']}\n"
                   f"Scale Factor: {meta['scale']:.3f}\n"
                   f"Padding: {meta['pad_top']}px top, {meta['pad_left']}px left\n"
                   f"\n"
                   f"Operations Applied:\n"
                   f"  1. Letterbox resize to 640x640\n"
                   f"  2. Maintained aspect ratio\n"
                   f"  3. Added padding\n"
                   f"  4. Scaled to [0, 1]\n"
                   f"\n"
                   f"Ready for: YOLOv5, v7, v8")
            
            self.print_box("YOLO PROCESSING COMPLETE", info)
            self.processing_history.append(("YOLO", meta['shape']))
            self.print_status("Processing complete!", "success")
            
        except Exception as e:
            self.print_status(f"Processing failed: {str(e)}", "error")
    
    def process_grayscale(self):
        """Process to grayscale"""
        if not self._check_image_loaded():
            return
        
        self.print_status("Converting to grayscale...", "info")
        
        try:
            pipeline = GrayscalePipeline((224, 224))
            tensor, meta = pipeline(self.current_image)
            
            # Store processed
            tensor_single = tensor[0, 0]
            denorm = (tensor_single * 0.5) + 0.5
            self.processed_image = np.clip(denorm * 255, 0, 255).astype(np.uint8)
            
            info = (f"Pipeline: Grayscale Conversion\n"
                   f"Output Shape: {meta.shape}\n"
                   f"Dtype: {meta.dtype}\n"
                   f"\n"
                   f"Operations Applied:\n"
                   f"  1. RGB to Grayscale\n"
                   f"  2. Resized to 224x224\n"
                   f"  3. Normalized to [-1, 1]\n"
                   f"\n"
                   f"Use cases:\n"
                   f"  • Medical imaging\n"
                   f"  • Document processing\n"
                   f"  • Texture analysis")
            
            self.print_box("GRAYSCALE PROCESSING COMPLETE", info)
            self.processing_history.append(("Grayscale", meta.shape))
            self.print_status("Processing complete!", "success")
            
        except Exception as e:
            self.print_status(f"Processing failed: {str(e)}", "error")
    
    def process_custom(self):
        """Apply custom processing"""
        if not self._check_image_loaded():
            return
        
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Custom Processing Options:{Colors.END}")
        print(f"{Colors.GREEN}1.{Colors.END} Edge Detection")
        print(f"{Colors.GREEN}2.{Colors.END} Color Space Transformation")
        print(f"{Colors.GREEN}3.{Colors.END} Multi-scale Resize")
        print(f"{Colors.GREEN}4.{Colors.END} Blur and Sharpen")
        
        choice = input(f"\n{Colors.YELLOW}Select option (1-4):{Colors.END} ").strip()
        
        try:
            if choice == "1":
                self._apply_edge_detection()
            elif choice == "2":
                self._apply_color_transform()
            elif choice == "3":
                self._apply_multiscale()
            elif choice == "4":
                self._apply_blur_sharpen()
            else:
                self.print_status("Invalid option", "warning")
        
        except Exception as e:
            self.print_status(f"Processing failed: {str(e)}", "error")
    
    def _apply_edge_detection(self):
        """Apply edge detection"""
        self.print_status("Applying edge detection...", "info")
        
        resize_node = ImageResizeNode("resize", 400, 400)
        resized, _ = resize_node.process(self.current_image)
        
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        info = (f"Custom: Edge Detection\n"
               f"Algorithm: Canny\n"
               f"Thresholds: 50, 150\n"
               f"Output: 400x400")
        
        self.print_box("EDGE DETECTION COMPLETE", info)
        self.processing_history.append(("Edge Detection", (400, 400)))
        self.print_status("Edge detection complete!", "success")
    
    def _apply_color_transform(self):
        """Apply color space transformation"""
        self.print_status("Transforming color space...", "info")
        
        color_node = ColorSpaceNode("rgb2hsv", ColorSpace.RGB, ColorSpace.HSV)
        hsv, _ = color_node.process(self.current_image)
        
        # Convert back for display
        self.processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        info = (f"Custom: Color Transform\n"
               f"Conversion: RGB → HSV\n"
               f"Channels: Hue, Saturation, Value")
        
        self.print_box("COLOR TRANSFORM COMPLETE", info)
        self.processing_history.append(("Color Transform", self.processed_image.shape))
        self.print_status("Color transform complete!", "success")
    
    def _apply_multiscale(self):
        """Apply multi-scale resize"""
        self.print_status("Creating multi-scale pyramid...", "info")
        
        scales = [224, 384, 512]
        print(f"\n{Colors.CYAN}Generating {len(scales)} scales...{Colors.END}")
        
        for scale in scales:
            resize_node = ImageResizeNode(f"resize_{scale}", scale, scale)
            resized, _ = resize_node.process(self.current_image)
            print(f"  ✓ {scale}x{scale}")
        
        # Store final scale
        self.processed_image = resized
        
        info = (f"Custom: Multi-scale Pyramid\n"
               f"Scales: {', '.join(map(str, scales))}\n"
               f"Final Output: {scales[-1]}x{scales[-1]}")
        
        self.print_box("MULTI-SCALE COMPLETE", info)
        self.processing_history.append(("Multi-scale", (scales[-1], scales[-1])))
        self.print_status("Multi-scale processing complete!", "success")
    
    def _apply_blur_sharpen(self):
        """Apply blur and sharpen"""
        self.print_status("Applying blur and sharpen...", "info")
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(self.current_image, (15, 15), 0)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(self.current_image, 1.5, blurred, -0.5, 0)
        self.processed_image = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        info = (f"Custom: Blur & Sharpen\n"
               f"Method: Unsharp Mask\n"
               f"Kernel: 15x15 Gaussian")
        
        self.print_box("BLUR & SHARPEN COMPLETE", info)
        self.processing_history.append(("Blur & Sharpen", self.processed_image.shape))
        self.print_status("Processing complete!", "success")
    
    def batch_process(self):
        """Batch process directory"""
        print(f"\n{Colors.YELLOW}Enter directory path (or '.' for current):{Colors.END}")
        dir_path = input("> ").strip() or "."
        
        if not os.path.isdir(dir_path):
            self.print_status("Invalid directory", "error")
            return
        
        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(Path(dir_path).glob(ext))
        
        if not images:
            self.print_status("No images found", "warning")
            return
        
        print(f"\n{Colors.CYAN}Found {len(images)} images{Colors.END}")
        print(f"{Colors.YELLOW}Process all with ImageNet pipeline? (y/n):{Colors.END}")
        
        if input("> ").strip().lower() != 'y':
            return
        
        # Create output directory
        output_dir = Path(dir_path) / "processed"
        output_dir.mkdir(exist_ok=True)
        
        pipeline = ImageNetPipeline(224)
        
        print(f"\n{Colors.CYAN}Processing...{Colors.END}")
        for i, img_path in enumerate(images, 1):
            try:
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                tensor, _ = pipeline(image)
                
                # Save (simplified for batch)
                output_path = output_dir / f"processed_{img_path.name}"
                cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                print(f"  [{i}/{len(images)}] ✓ {img_path.name}")
                
            except Exception as e:
                print(f"  [{i}/{len(images)}] ✗ {img_path.name}: {str(e)}")
        
        self.print_status(f"Batch processing complete! Output: {output_dir}", "success")
    
    def view_info(self):
        """View current image information"""
        if not self._check_image_loaded():
            return
        
        h, w = self.current_image.shape[:2]
        channels = self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1
        
        info = (f"File: {Path(self.current_image_path).name}\n"
               f"Path: {self.current_image_path}\n"
               f"Dimensions: {w} x {h} pixels\n"
               f"Channels: {channels}\n"
               f"Dtype: {self.current_image.dtype}\n"
               f"Memory: {self.current_image.nbytes / 1024:.1f} KB\n"
               f"Min/Max: {self.current_image.min()} / {self.current_image.max()}")
        
        self.print_box("IMAGE INFORMATION", info)
    
    def save_processed(self):
        """Save processed image"""
        if self.processed_image is None:
            self.print_status("No processed image to save", "warning")
            return
        
        print(f"\n{Colors.YELLOW}Enter output path (default: output.png):{Colors.END}")
        path = input("> ").strip() or "output.png"
        
        try:
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr_image)
            
            self.print_status(f"Saved to: {path}", "success")
            
        except Exception as e:
            self.print_status(f"Failed to save: {str(e)}", "error")
    
    def view_history(self):
        """View processing history"""
        if not self.processing_history:
            self.print_status("No processing history", "info")
            return
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}Processing History:{Colors.END}")
        for i, (operation, shape) in enumerate(self.processing_history, 1):
            print(f"  {i}. {operation} → {shape}")
    
    def _check_image_loaded(self):
        """Check if image is loaded"""
        if self.current_image is None:
            self.print_status("Please load an image first (Option 1)", "warning")
            return False
        return True
    
    def run(self):
        """Run the CLI application"""
        self.print_header()
        
        while True:
            self.print_menu()
            
            choice = input(f"\n{Colors.YELLOW}Enter choice (0-9):{Colors.END} ").strip()
            
            if choice == "1":
                self.load_image()
            elif choice == "2":
                self.process_imagenet()
            elif choice == "3":
                self.process_yolo()
            elif choice == "4":
                self.process_grayscale()
            elif choice == "5":
                self.process_custom()
            elif choice == "6":
                self.batch_process()
            elif choice == "7":
                self.view_info()
            elif choice == "8":
                self.save_processed()
            elif choice == "9":
                self.view_history()
            elif choice == "0":
                self.print_status("Thank you for using Nexus ML! Goodbye! 👋", "success")
                break
            else:
                self.print_status("Invalid choice. Please enter 0-9", "warning")
            
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")


def main():
    """Entry point"""
    try:
        app = ImageProcessorCLI()
        app.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user. Exiting...{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {str(e)}{Colors.END}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
