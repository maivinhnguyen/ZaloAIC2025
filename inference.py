"""
Siamese YOLO Inference Script

This script performs one-shot object detection on video using a query image.
It processes each frame of the video, detects objects similar to the query image,
and saves visualized results to an output folder.

Usage:
    python inference.py --model runs/detect/train/weights/best.pt \
                       --query path/to/query_image.jpg \
                       --video path/to/video.mp4 \
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


class SiamYOLOInference:
    """Siamese YOLO inference class for one-shot object detection on videos."""
    
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
                # Get model config from checkpoint or use default
                cfg = checkpoint.get('model_cfg', 'yolo11n.yaml')
                nc = checkpoint.get('nc', 1)
                model = SiamDetectionModel(cfg=cfg, nc=nc, verbose=False)
                model.load_state_dict(checkpoint)
        else:
            # checkpoint is the model itself
            model = checkpoint
        
        model = model.to(self.device)
        return model
    
    def preprocess_image(self, image, target_size=None):
        """
        Preprocess image for model input.
        
        Args:
            image (np.ndarray): Input image in BGR format (OpenCV format)
            target_size (int): Target size for resizing (default: self.imgsz)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
            tuple: Original image shape (h, w)
        """
        if target_size is None:
            target_size = self.imgsz
        
        # Store original shape
        orig_shape = image.shape[:2]  # (h, w)
        
        # Resize image while maintaining aspect ratio
        img = self._letterbox(image, target_size)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to tensor
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        return img, orig_shape
    
    def _letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        """
        Resize image with aspect ratio preservation and padding.
        
        Args:
            img (np.ndarray): Input image
            new_shape (int): Target size
            color (tuple): Padding color
            
        Returns:
            np.ndarray: Resized and padded image
        """
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img
    
    def postprocess(self, predictions, orig_shape, input_shape):
        """
        Post-process model predictions.
        
        Args:
            predictions (torch.Tensor): Raw model predictions
            orig_shape (tuple): Original image shape (h, w)
            input_shape (tuple): Input tensor shape
            
        Returns:
            np.ndarray: Processed detections [x1, y1, x2, y2, conf, cls]
        """
        # Apply NMS
        predictions = non_max_suppression(
            predictions,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            agnostic=False,
            max_det=300
        )
        
        # Process detections
        detections = []
        for pred in predictions:
            if len(pred):
                # Rescale boxes from input size to original image size
                pred[:, :4] = ops.scale_boxes(input_shape[2:], pred[:, :4], orig_shape).round()
                detections.append(pred.cpu().numpy())
            else:
                detections.append(np.empty((0, 6)))
        
        return detections[0] if detections else np.empty((0, 6))
    
    def visualize(self, image, detections, query_image=None):
        """
        Visualize detections on image.
        
        Args:
            image (np.ndarray): Original image in BGR format
            detections (np.ndarray): Detections array [x1, y1, x2, y2, conf, cls]
            query_image (np.ndarray): Optional query image to show in corner
            
        Returns:
            np.ndarray: Annotated image
        """
        # Use thicker lines for better visibility
        annotator = Annotator(image.copy(), line_width=3, example=str(self.names))
        
        # Draw detections with thicker, brighter boxes
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = f'{self.names[int(cls)]} {conf:.2f}'
            # Use bright color for better visibility
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
        
        # Add query image in top-right corner if provided (to avoid detection counter)
        if query_image is not None:
            # Resize query image to fit in corner
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
    
    def process_video(self, query_image_path, video_path, output_dir, save_video=True, show_progress=True):
        """
        Process video with Siamese YOLO detection.
        
        Args:
            query_image_path (str): Path to query image
            video_path (str): Path to input video
            output_dir (str): Directory to save results
            save_video (bool): Whether to save output video
            show_progress (bool): Whether to show progress bar
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        # Load query image
        LOGGER.info(f"Loading query image from {query_image_path}")
        query_image = cv2.imread(str(query_image_path))
        if query_image is None:
            raise ValueError(f"Could not load query image: {query_image_path}")
        
        # Preprocess query image
        query_tensor, _ = self.preprocess_image(query_image)
        
        # Open video
        LOGGER.info(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        LOGGER.info(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        if save_video:
            output_video_path = output_dir / f"output_{Path(video_path).name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
            LOGGER.info(f"Saving output video to: {output_video_path}")
        
        # Process frames
        frame_count = 0
        detection_count = 0
        frames_with_detections = 0
        
        pbar = tqdm(total=total_frames, desc="Processing frames") if show_progress else None
        
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                frame_tensor, orig_shape = self.preprocess_image(frame)
                
                # Run inference with query and support images
                predictions = self.model(frame_tensor, support_img=query_tensor)
                
                # Postprocess predictions
                detections = self.postprocess(predictions, orig_shape, frame_tensor.shape)
                
                # Count detections
                num_dets_in_frame = len(detections)
                detection_count += num_dets_in_frame
                if num_dets_in_frame > 0:
                    frames_with_detections += 1
                
                # Visualize
                annotated_frame = self.visualize(frame, detections, query_image)
                
                # Save frame to video
                if save_video:
                    out.write(annotated_frame)
                
                # Save ONLY frames with detections above threshold
                if num_dets_in_frame > 0:
                    # Get max confidence of detections in this frame
                    max_conf = detections[:, 4].max() if len(detections) > 0 else 0
                    frame_path = frames_dir / f"frame_{frame_count:06d}_dets_{num_dets_in_frame}_conf_{max_conf:.2f}.jpg"
                    cv2.imwrite(str(frame_path), annotated_frame)
                
                frame_count += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({'detections': detection_count, 'frames_with_dets': frames_with_detections})
        
        # Cleanup
        cap.release()
        if save_video:
            out.release()
        if pbar:
            pbar.close()
        
        LOGGER.info(f"Processing complete!")
        LOGGER.info(f"Processed {frame_count} frames")
        LOGGER.info(f"Frames with detections: {frames_with_detections} ({frames_with_detections/frame_count*100:.1f}%)")
        LOGGER.info(f"Total detections: {detection_count}")
        LOGGER.info(f"Average detections per frame: {detection_count / frame_count:.2f}")
        LOGGER.info(f"Saved {frames_with_detections} frame images (only frames with detections)")
        LOGGER.info(f"Results saved to: {output_dir}")


def main():
    """Main function to run inference."""
    parser = argparse.ArgumentParser(description='Siamese YOLO Inference on Video')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--query', type=str, required=True,
                       help='Path to query image (object to detect)')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
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
    parser.add_argument('--no-save-video', action='store_true',
                       help='Do not save output video (only save frames)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not Path(args.query).exists():
        raise FileNotFoundError(f"Query image not found: {args.query}")
    if not Path(args.video).exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    # Initialize inference engine
    inference = SiamYOLOInference(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz
    )
    
    # Process video
    inference.process_video(
        query_image_path=args.query,
        video_path=args.video,
        output_dir=args.output,
        save_video=not args.no_save_video
    )


if __name__ == '__main__':
    main()
