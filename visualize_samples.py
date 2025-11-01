"""
Script to visualize training samples with bounding boxes from a dataset.
Takes data.yaml path as input and displays annotated images.
"""

import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from ultralytics.data.dataset import SiamDataset
from ultralytics.utils import LOGGER


def visualize_sample(sample, output_path=None, show=True):
    """
    Visualize a single sample with bounding boxes.
    
    Args:
        sample (dict): Sample from dataset with 'img', 'bboxes', 'cls', etc.
        output_path (str): Path to save the visualization
        show (bool): Whether to display the image
    """
    # Extract image and labels
    img_tensor = sample.get('img')
    bboxes = sample.get('bboxes')
    cls_labels = sample.get('cls')
    
    # Convert tensor to numpy if needed
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.numpy()
    else:
        img = img_tensor
    
    # Handle different image formats
    if img.ndim == 3:
        if img.shape[0] == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[2] in [1, 3, 4]:  # HWC format
            pass
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
    
    # Denormalize if needed
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    # Convert to BGR for OpenCV
    if img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    
    # Draw bounding boxes
    if bboxes is not None and len(bboxes) > 0:
        if isinstance(bboxes, torch.Tensor):
            bboxes_np = bboxes.numpy()
        else:
            bboxes_np = bboxes
        
        if isinstance(cls_labels, torch.Tensor):
            cls_np = cls_labels.numpy()
        else:
            cls_np = cls_labels
        
        h, w = img_bgr.shape[:2]
        
        for i, bbox in enumerate(bboxes_np):
            # Assume bbox is in xywh format (normalized)
            x_center, y_center, box_w, box_h = bbox
            
            # Denormalize
            x_center = int(x_center * w)
            y_center = int(y_center * h)
            box_w = int(box_w * w)
            box_h = int(box_h * h)
            
            # Convert to corner coordinates
            x1 = x_center - box_w // 2
            y1 = y_center - box_h // 2
            x2 = x_center + box_w // 2
            y2 = y_center + box_h // 2
            
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Draw rectangle
            color = (0, 255, 0)  # Green for bboxes
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            
            # Draw class label
            if len(cls_np) > i:
                cls_id = int(cls_np[i, 0]) if cls_np[i].ndim > 0 else int(cls_np[i])
                text = f"Class {cls_id}"
                cv2.putText(img_bgr, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
        
        print(f"✓ Drew {len(bboxes_np)} bounding boxes")
    else:
        print("✗ No bounding boxes found in sample")
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(str(output_path), img_bgr)
        print(f"✓ Saved visualization to {output_path}")
    
    return img_bgr


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Siamese dataset samples with bounding boxes'
    )
    parser.add_argument('data', type=str, help='Path to data.yaml file')
    parser.add_argument('--output-dir', type=str, default='./sample_visualizations',
                       help='Output directory for saved visualizations')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--save', action='store_true',
                       help='Save visualizations to disk')
    
    args = parser.parse_args()
    
    # Load data config
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")
    
    print(f"Loading data from: {data_path}")
    
    # Import YAML loader
    try:
        import yaml
        with open(data_path) as f:
            data_dict = yaml.safe_load(f)
    except Exception as e:
        LOGGER.error(f"Failed to load data.yaml: {e}")
        return
    
    # Build dataset
    print("Building dataset...")
    train_path = data_dict.get('train')
    if not train_path:
        raise ValueError("'train' key not found in data.yaml")
    
    # Handle relative paths
    if not Path(train_path).is_absolute():
        train_path = data_path.parent / train_path
    
    try:
        dataset = SiamDataset(
            img_path=str(train_path),
            imgsz=640,
            batch_size=1,
            augment=False,  # No augmentation for visualization
            cache=False,
            data=data_dict,
            task='detect'
        )
    except Exception as e:
        LOGGER.error(f"Failed to build dataset: {e}")
        print(f"Error details: {e}")
        return
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Create output directory
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {output_dir}")
    
    # Visualize samples
    num_samples = min(args.num_samples, len(dataset))
    print(f"\nVisualizing {num_samples} samples...\n")
    
    for idx in range(num_samples):
        print(f"--- Sample {idx + 1}/{num_samples} ---")
        try:
            sample = dataset[idx]
            
            # Get image info
            if 'query_img' in sample:
                print(f"Query image shape: {sample['query_img'].shape}")
            if 'img' in sample:
                print(f"Image shape: {sample['img'].shape}")
            
            # Check for bboxes
            bboxes = sample.get('bboxes')
            if bboxes is not None:
                if isinstance(bboxes, torch.Tensor):
                    print(f"Bboxes: {bboxes.shape} - {len(bboxes)} boxes found")
                else:
                    print(f"Bboxes: {len(bboxes)} boxes found")
            else:
                print("Bboxes: None")
            
            cls_labels = sample.get('cls')
            if cls_labels is not None:
                if isinstance(cls_labels, torch.Tensor):
                    print(f"Classes: {cls_labels.shape}")
                else:
                    print(f"Classes: {len(cls_labels)} labels found")
            
            # Visualize
            if args.save:
                output_path = output_dir / f"sample_{idx:03d}_query.jpg"
            else:
                output_path = None
            
            # Use query_img if available, otherwise use img
            if 'query_img' in sample:
                sample_to_plot = sample.copy()
                sample_to_plot['img'] = sample['query_img']
            else:
                sample_to_plot = sample
            
            img_with_boxes = visualize_sample(sample_to_plot, output_path=output_path, show=False)
            
            # Visualize support image if available
            if 'support_img' in sample:
                print(f"Support image shape: {sample['support_img'].shape}")
                support_sample = {
                    'img': sample['support_img'],
                    'bboxes': None,  # Support images don't have boxes
                    'cls': None
                }
                if args.save:
                    output_path = output_dir / f"sample_{idx:03d}_support.jpg"
                else:
                    output_path = None
                visualize_sample(support_sample, output_path=output_path, show=False)
            
            print()
        
        except Exception as e:
            print(f"✗ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("Done!")
    if args.save:
        print(f"Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
