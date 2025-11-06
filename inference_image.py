"""
Siamese YOLO Image Inference Script

This script performs one-shot object detection on a single image or folder of images
using a query image.

Usage:
    # Single image
    python inference_image.py --model runs/detect/train/weights/best.pt \
                              --query path/to/query_image.jpg \
                              --source path/to/test_image.jpg \
                              --output output_folder
    
    # Folder of images
    python inference_image.py --model runs/detect/train/weights/best.pt \
                              --query path/to/query_image.jpg \
                              --source path/to/images_folder/ \
                              --output output_folder
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ultralytics.nn.tasks import SiamDetectionModel
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.plotting import Annotator, colors


class SiamYOLOImageInference:
    """Siamese YOLO inference class for one-shot object detection on images."""
    
    def __init__(self, model_path, device='cuda', conf_threshold=0.25, iou_threshold=0.45, imgsz=640):
        """
        Initialize the Siamese YOLO inference engine.
        
        Args:
            model_path (str): Path to the trained model weights (.pt file)
            device (str): Device to run inference on ('cuda' or 'cpu')
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            imgsz (int): Input image size
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        
        # Load model
        LOGGER.info(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Model names (classes)
        self.names = self.model.names if hasattr(self.model, 'names') else {0: 'object'}
        
        LOGGER.info(f"Model loaded successfully on {self.device}")
        LOGGER.info(f"Classes: {self.names}")
    
    def _load_model(self, model_path):
        """Load the Siamese YOLO model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration if available
        if isinstance(checkpoint, dict):
            # Try to load from 'ema' first (Ultralytics saves best model here)
            if 'ema' in checkpoint and checkpoint['ema'] is not None:
                model = checkpoint['ema']
                if hasattr(model, 'float'):
                    model = model.float()
            # Then try 'model' key
            elif 'model' in checkpoint and checkpoint['model'] is not None:
                model = checkpoint['model']
                if hasattr(model, 'float'):
                    model = model.float()
            else:
                # Try to create model from checkpoint structure
                LOGGER.warning("Checkpoint structure not recognized, attempting to load as state dict")
                cfg = checkpoint.get('model_cfg', 'yolo11n.yaml')
                nc = checkpoint.get('nc', 1)
                model = SiamDetectionModel(cfg=cfg, nc=nc, verbose=False)
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        
        model = model.to(self.device)
        return model
    
    def preprocess_image(self, image, target_size=None):
        """Preprocess image for model input."""
        if target_size is None:
            target_size = self.imgsz
        
        orig_shape = image.shape[:2]
        img = self._letterbox(image, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        return img, orig_shape
    
    def _letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        """Resize image with aspect ratio preservation and padding."""
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img
    
    def postprocess(self, predictions, orig_shape, input_shape):
        """Post-process model predictions."""
        predictions = non_max_suppression(
            predictions,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            agnostic=False,
            max_det=300
        )
        
        detections = []
        for pred in predictions:
            if len(pred):
                pred[:, :4] = ops.scale_boxes(input_shape[2:], pred[:, :4], orig_shape).round()
                detections.append(pred.cpu().numpy())
            else:
                detections.append(np.empty((0, 6)))
        
        return detections[0] if detections else np.empty((0, 6))
    
    def visualize(self, image, detections, query_image=None):
        """Visualize detections on image."""
        # Use thicker lines for better visibility
        annotator = Annotator(image.copy(), line_width=3, example=str(self.names))
        
        # Draw detections with thicker, brighter boxes
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = f'{self.names[int(cls)]} {conf:.2f}'
            color = colors(int(cls), True)
            annotator.box_label([x1, y1, x2, y2], label, color=color)
        
        result = annotator.result()
        
        # Add detection counter at top
        num_detections = len(detections)
        counter_text = f'Detections: {num_detections}'
        # Add background for text
        text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.rectangle(result, (10, 10), (20 + text_size[0], 40 + text_size[1]), (0, 0, 0), -1)
        cv2.putText(result, counter_text, (15, 35 + text_size[1]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Add query image in top-right corner (to avoid detection counter)
        if query_image is not None:
            query_h, query_w = query_image.shape[:2]
            corner_size = min(150, result.shape[0] // 4, result.shape[1] // 4)
            scale = corner_size / max(query_h, query_w)
            new_w, new_h = int(query_w * scale), int(query_h * scale)
            query_resized = cv2.resize(query_image, (new_w, new_h))
            
            # Add bright border for visibility
            query_bordered = cv2.copyMakeBorder(
                query_resized, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 255, 0)
            )
            
            # Overlay on result (top-right corner)
            h, w = query_bordered.shape[:2]
            margin = 10
            x_pos = result.shape[1] - w - margin  # Right side
            y_pos = margin  # Top
            result[y_pos:y_pos+h, x_pos:x_pos+w] = query_bordered
            
            # Add "Query" label with background
            label_text = "Query Object"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = y_pos + h + 25
            cv2.rectangle(result, (x_pos, label_y - text_size[1] - 5), 
                         (x_pos + text_size[0] + 10, label_y + 5), (0, 0, 0), -1)
            cv2.putText(result, label_text, (x_pos + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result
    
    def process_image(self, query_image_path, image_path, output_dir):
        """
        Process a single image with Siamese YOLO detection.
        
        Args:
            query_image_path (str): Path to query image
            image_path (str): Path to input image
            output_dir (Path): Directory to save results
            
        Returns:
            int: Number of detections
        """
        # Load query image (cache it)
        if not hasattr(self, '_query_tensor'):
            LOGGER.info(f"Loading query image from {query_image_path}")
            self.query_image = cv2.imread(str(query_image_path))
            if self.query_image is None:
                raise ValueError(f"Could not load query image: {query_image_path}")
            self._query_tensor, _ = self.preprocess_image(self.query_image)
        
        # Load test image
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning(f"Could not load image: {image_path}")
            return 0
        
        # Preprocess
        image_tensor, orig_shape = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor, support_img=self._query_tensor)
        
        # Postprocess
        detections = self.postprocess(predictions, orig_shape, image_tensor.shape)
        
        # Visualize
        annotated_image = self.visualize(image, detections, self.query_image)
        
        # Save result ONLY if detections found
        num_dets = len(detections)
        if num_dets > 0:
            max_conf = detections[:, 4].max() if len(detections) > 0 else 0
            output_filename = f"{Path(image_path).stem}_dets_{num_dets}_conf_{max_conf:.2f}.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), annotated_image)
            LOGGER.info(f"Saved: {output_filename} ({num_dets} detections, max conf: {max_conf:.2f})")
        
        return num_dets
    
    def process_folder(self, query_image_path, source_path, output_dir):
        """
        Process folder of images with Siamese YOLO detection.
        
        Args:
            query_image_path (str): Path to query image
            source_path (str): Path to folder or single image
            output_dir (str): Directory to save results
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of images
        source_path = Path(source_path)
        if source_path.is_file():
            image_paths = [source_path]
        else:
            # Get all images in folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(source_path.glob(f'*{ext}'))
                image_paths.extend(source_path.glob(f'*{ext.upper()}'))
        
        if not image_paths:
            LOGGER.warning(f"No images found in {source_path}")
            return
        
        LOGGER.info(f"Found {len(image_paths)} images to process")
        
        # Process images
        total_detections = 0
        images_with_detections = 0
        for image_path in tqdm(image_paths, desc="Processing images"):
            detections = self.process_image(query_image_path, image_path, output_dir)
            total_detections += detections
            if detections > 0:
                images_with_detections += 1
        
        LOGGER.info(f"Processing complete!")
        LOGGER.info(f"Processed {len(image_paths)} images")
        LOGGER.info(f"Images with detections: {images_with_detections} ({images_with_detections/len(image_paths)*100:.1f}%)")
        LOGGER.info(f"Total detections: {total_detections}")
        LOGGER.info(f"Average detections per image: {total_detections / len(image_paths):.2f}")
        LOGGER.info(f"Saved {images_with_detections} images (only images with detections)")
        LOGGER.info(f"Results saved to: {output_dir}")


def main():
    """Main function to run inference."""
    parser = argparse.ArgumentParser(description='Siamese YOLO Inference on Images')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--query', type=str, required=True,
                       help='Path to query image (object to detect)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to input image or folder of images')
    parser.add_argument('--output', type=str, default='inference_output',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not Path(args.query).exists():
        raise FileNotFoundError(f"Query image not found: {args.query}")
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Source not found: {args.source}")
    
    # Initialize inference engine
    inference = SiamYOLOImageInference(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz
    )
    
    # Process images
    inference.process_folder(
        query_image_path=args.query,
        source_path=args.source,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
