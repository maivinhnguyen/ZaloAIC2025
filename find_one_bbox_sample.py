"""
Visualize a specific sample that has bboxes.
"""

from pathlib import Path
import yaml
import torch
from ultralytics.data.dataset import SiamDataset
import cv2
import numpy as np

data_path = Path("dataset/data.yaml")
with open(data_path) as f:
    data_dict = yaml.safe_load(f)

train_path = data_dict.get('train')
if not Path(train_path).is_absolute():
    train_path = data_path.parent / train_path

# Build dataset (using augment=False for visualization)
dataset = SiamDataset(
    img_path=str(train_path),
    imgsz=640,
    batch_size=1,
    augment=False,
    cache=False,
    data=data_dict,
    task='detect'
)

print(f"Dataset loaded with {len(dataset)} samples\n")

# Look through labels to find one with bboxes
print("Finding samples with bboxes...\n")

for idx in range(len(dataset.labels)):
    label = dataset.labels[idx]
    bboxes = label.get("bboxes")
    
    if bboxes is not None and len(bboxes) > 0:
        print(f"✓ Found sample {idx} with {len(bboxes)} bboxes!")
        print(f"  im_file: {label.get('im_file')}")
        print(f"  support_file: {label.get('support_file')}")
        print(f"  bboxes: {bboxes}")
        
        # Load and visualize
        sample = dataset[idx]
        img = sample.get('img')
        query_img = sample.get('query_img')
        
        if query_img is not None:
            img = query_img
        
        print(f"  Image shape: {img.shape if hasattr(img, 'shape') else 'unknown'}")
        print(f"  Sample bboxes: {sample.get('bboxes')}")
        print()
        break
else:
    print("No samples with bboxes found in the loaded dataset!")
