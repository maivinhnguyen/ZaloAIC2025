# SiamYOLOv8 Training Data Structure Guide

## 📋 Overview

SiamYOLOv8 requires **triplet data** consisting of:
1. **Query Image** - The image to detect objects in
2. **Support Image** - The reference/template image for one-shot learning
3. **Annotations** - Bounding box labels for the query image

This guide explains how to organize and format your training data.

---

## 📁 Directory Structure

### Recommended Layout

```
dataset/
├── images/
│   ├── train/
│   │   ├── query/          # Query images for training
│   │   │   ├── img001.jpg
│   │   │   ├── img002.jpg
│   │   │   └── ...
│   │   └── support/        # Support/template images for training
│   │       ├── template001.jpg
│   │       ├── template002.jpg
│   │       └── ...
│   ├── val/
│   │   ├── query/          # Query images for validation
│   │   │   ├── img101.jpg
│   │   │   └── ...
│   │   └── support/        # Support images for validation
│   │       ├── template101.jpg
│   │       └── ...
│   └── test/
│       ├── query/          # Query images for testing
│       └── support/        # Support images for testing
│
├── labels/
│   ├── train.txt           # Training triplet labels
│   ├── val.txt             # Validation triplet labels
│   └── test.txt            # Test triplet labels
│
└── data.yaml               # Dataset configuration file
```

### Minimal Layout (Alternative)

```
siam_dataset/
├── train_query/            # Training query images
├── train_support/          # Training support images
├── val_query/              # Validation query images
├── val_support/            # Validation support images
├── train_labels.txt        # Training annotations
└── data.yaml               # Configuration
```

---

## 🏷️ Label Format (Triplet Format)

### Standard Format

Each line in the label file contains:

```
<query_image_path> <support_image_path> <class_id> <x_center> <y_center> <width> <height> [<class_id> <x_center> <y_center> <width> <height> ...]
```

### Parameters

| Field | Description | Range |
|-------|-------------|-------|
| `query_image_path` | Path to query image | Relative or absolute |
| `support_image_path` | Path to support/template image | Relative or absolute |
| `class_id` | Object class ID (0-indexed) | 0 to nc-1 |
| `x_center` | Bounding box center X (normalized) | 0.0 to 1.0 |
| `y_center` | Bounding box center Y (normalized) | 0.0 to 1.0 |
| `width` | Bounding box width (normalized) | 0.0 to 1.0 |
| `height` | Bounding box height (normalized) | 0.0 to 1.0 |

### Examples

**Single object:**
```
images/train/query/img001.jpg images/train/support/template001.jpg 0 0.5 0.5 0.4 0.3
```

**Multiple objects:**
```
images/train/query/img002.jpg images/train/support/template002.jpg 0 0.3 0.4 0.2 0.25 0 0.7 0.6 0.15 0.2
```

**No objects (empty/background):**
```
images/train/query/img003.jpg images/train/support/template003.jpg
```

### Coordinate Conversion

If you have pixel coordinates (x_left, y_top, x_right, y_bottom):

```python
x_center = (x_left + x_right) / (2 * image_width)
y_center = (y_top + y_bottom) / (2 * image_height)
width = (x_right - x_left) / image_width
height = (y_bottom - y_top) / image_height
```

Or with (x_left, y_top, width_pixels, height_pixels):

```python
x_center = (x_left + width_pixels / 2) / image_width
y_center = (y_top + height_pixels / 2) / image_height
width = width_pixels / image_width
height = height_pixels / image_height
```

---

## 📄 Configuration File (data.yaml)

Create a `data.yaml` file for your dataset:

### Minimal Configuration

```yaml
# Dataset root path
path: /path/to/dataset

# Training/validation directories
train: images/train
val: images/val
test: images/test  # Optional

# Number of classes
nc: 1

# Class names
names: {0: 'object'}
```

### Extended Configuration

```yaml
# Dataset paths
path: /home/user/datasets/siamese_coco
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 1

# Class names mapping
names:
  0: 'object'

# Optional: Dataset statistics
train_images: 10000
val_images: 2000
test_images: 1000

# Optional: Data augmentation config
augmentation:
  enable: true
  brightness: 0.1
  contrast: 0.1
  saturation: 0.2
```

---

## 💾 Label File Examples

### Example 1: Simple Single-Object Detection

**train_labels.txt:**
```
images/train/query/01.jpg images/train/support/s01.jpg 0 0.5 0.5 0.4 0.3
images/train/query/02.jpg images/train/support/s02.jpg 0 0.45 0.55 0.35 0.35
images/train/query/03.jpg images/train/support/s03.jpg 0 0.6 0.4 0.3 0.4
images/train/query/04.jpg images/train/support/s04.jpg 0 0.5 0.5 0.4 0.4
images/train/query/05.jpg images/train/support/s05.jpg 0 0.55 0.45 0.32 0.38
```

### Example 2: Multiple Objects

**train_labels.txt:**
```
# Single object
images/train/query/multi01.jpg images/train/support/s_multi01.jpg 0 0.3 0.3 0.2 0.25

# Multiple objects of same class
images/train/query/multi02.jpg images/train/support/s_multi02.jpg 0 0.3 0.3 0.2 0.25 0 0.7 0.7 0.25 0.2

# Multiple objects (more complex scene)
images/train/query/multi03.jpg images/train/support/s_multi03.jpg 0 0.2 0.2 0.15 0.15 0 0.5 0.5 0.2 0.25 0 0.8 0.3 0.15 0.2
```

### Example 3: Mixed with Background Images

**train_labels.txt:**
```
# Images with objects
images/train/query/obj01.jpg images/train/support/s_obj01.jpg 0 0.5 0.5 0.4 0.3
images/train/query/obj02.jpg images/train/support/s_obj02.jpg 0 0.45 0.55 0.35 0.35

# Background/empty images
images/train/query/bg01.jpg images/train/support/s_bg01.jpg
images/train/query/bg02.jpg images/train/support/s_bg02.jpg
```

---

## 🔄 Dataset Splitting

### Train/Val/Test Split Recommendation

```
Total Images: 10,000
├── Train: 8,000 (80%)
├── Val:   1,500 (15%)
└── Test:  500   (5%)
```

### Creating Splits Programmatically

```python
import os
import random
from pathlib import Path

# Configuration
dataset_root = Path("/path/to/dataset")
train_split = 0.8
val_split = 0.15
test_split = 0.05

# Create directories
splits = ['train', 'val', 'test']
for split in splits:
    (dataset_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

# Load all triplet data
all_data = []
with open(dataset_root / 'all_labels.txt', 'r') as f:
    all_data = f.readlines()

# Shuffle and split
random.shuffle(all_data)
n = len(all_data)

train_idx = int(n * train_split)
val_idx = int(n * (train_split + val_split))

train_data = all_data[:train_idx]
val_data = all_data[train_idx:val_idx]
test_data = all_data[val_idx:]

# Write split files
with open(dataset_root / 'labels' / 'train.txt', 'w') as f:
    f.writelines(train_data)

with open(dataset_root / 'labels' / 'val.txt', 'w') as f:
    f.writelines(val_data)

with open(dataset_root / 'labels' / 'test.txt', 'w') as f:
    f.writelines(test_data)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
```

---

## 🛠️ Data Preparation Examples

### Example 1: Converting from COCO Format

```python
import json
import cv2
from pathlib import Path

def coco_to_siam(coco_json, output_file, query_dir, support_dir):
    """Convert COCO format to Siamese triplet format."""
    
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping for image IDs
    images = {img['id']: img for img in coco_data['images']}
    
    # Get image size for normalization
    triplets = []
    
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        image_info = images[image_id]
        
        img_width = image_info['width']
        img_height = image_info['height']
        
        # Get bbox (in COCO format: x_left, y_top, width, height)
        bbox = annotation['bbox']
        x_left, y_top, w, h = bbox
        
        # Convert to normalized center format
        x_center = (x_left + w / 2) / img_width
        y_center = (y_top + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        # Clamp to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        class_id = annotation['category_id']
        
        # Create triplet
        query_img = f"{query_dir}/{image_info['file_name']}"
        support_img = f"{support_dir}/{image_info['file_name']}"  # Use same for support
        
        triplet = f"{query_img} {support_img} {class_id} {x_center} {y_center} {width} {height}"
        triplets.append(triplet)
    
    # Write to file
    with open(output_file, 'w') as f:
        for triplet in triplets:
            f.write(triplet + '\n')
    
    print(f"Converted {len(triplets)} annotations to {output_file}")

# Usage
coco_to_siam(
    'instances_train.json',
    'train_labels.txt',
    'images/train/query',
    'images/train/support'
)
```

### Example 2: Creating Synthetic Triplets

```python
import os
import shutil
from pathlib import Path

def create_synthetic_triplets(images_dir, output_labels, query_support_ratio=0.5):
    """Create synthetic triplets from image directory."""
    
    images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    n = len(images)
    
    # Split into query and support
    split_idx = int(n * query_support_ratio)
    query_images = images[:split_idx]
    support_images = images[split_idx:]
    
    triplets = []
    
    # Pair each query with a support image
    for i, q_img in enumerate(query_images):
        s_img = support_images[i % len(support_images)]
        
        # Assume no bounding boxes for now (background images)
        triplet = f"images/train/query/{q_img} images/train/support/{s_img}"
        triplets.append(triplet)
    
    # Write to file
    with open(output_labels, 'w') as f:
        for triplet in triplets:
            f.write(triplet + '\n')
    
    print(f"Created {len(triplets)} synthetic triplets")

# Usage
create_synthetic_triplets('path/to/images', 'train_labels.txt')
```

### Example 3: Converting from Pascal VOC Format

```python
import xml.etree.ElementTree as ET
from pathlib import Path

def voc_to_siam(voc_annotations_dir, images_dir, output_file):
    """Convert Pascal VOC format to Siamese triplet format."""
    
    triplets = []
    
    for xml_file in Path(voc_annotations_dir).glob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image info
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            image_name = root.find('filename').text
            query_img = f"{images_dir}/{image_name}"
            support_img = f"{images_dir}/{image_name}"  # Use same image for support
            
            bbox_data = []
            
            # Extract bounding boxes
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = 0  # Map to your class ID
                
                bndbox = obj.find('bndbox')
                x_left = int(bndbox.find('xmin').text)
                y_top = int(bndbox.find('ymin').text)
                x_right = int(bndbox.find('xmax').text)
                y_bottom = int(bndbox.find('ymax').text)
                
                # Convert to normalized center format
                x_center = (x_left + x_right) / (2 * img_width)
                y_center = (y_top + y_bottom) / (2 * img_height)
                width = (x_right - x_left) / img_width
                height = (y_bottom - y_top) / img_height
                
                bbox_data.append(f"{class_id} {x_center} {y_center} {width} {height}")
            
            # Create triplet line
            if bbox_data:
                triplet = f"{query_img} {support_img} " + " ".join(bbox_data)
                triplets.append(triplet)
        
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
    
    # Write to file
    with open(output_file, 'w') as f:
        for triplet in triplets:
            f.write(triplet + '\n')
    
    print(f"Converted {len(triplets)} VOC annotations to {output_file}")

# Usage
voc_to_siam('annotations/', 'images/', 'train_labels.txt')
```

---

## ✅ Data Validation Checklist

Before training, verify your dataset:

- [ ] **Directory structure** - All required folders exist
- [ ] **Image files** - Query and support images present and readable
- [ ] **Label file** - All triplets properly formatted
- [ ] **Image-label correspondence** - Each image path is valid
- [ ] **Normalized coordinates** - All bbox values in [0, 1]
- [ ] **No empty lines** - Label file has no blank lines
- [ ] **Path format** - Paths are consistent (relative or absolute)
- [ ] **data.yaml** - Configuration file properly formatted
- [ ] **Class IDs** - Match configuration file
- [ ] **Sample verification** - Visually verify a few samples

### Validation Script

```python
import os
from pathlib import Path
import cv2

def validate_dataset(dataset_root, label_file):
    """Validate Siamese dataset structure and content."""
    
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    
    dataset_root = Path(dataset_root)
    label_file = dataset_root / label_file
    
    # Check label file existence
    if not label_file.exists():
        print(f"❌ Label file not found: {label_file}")
        return False
    
    valid_count = 0
    invalid_count = 0
    
    with open(label_file, 'r') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                print(f"⚠️  Line {line_idx}: Empty line")
                continue
            
            parts = line.split()
            
            # Check minimum parts (at least query and support images)
            if len(parts) < 2:
                print(f"❌ Line {line_idx}: Invalid format (need at least 2 parts)")
                invalid_count += 1
                continue
            
            query_img = parts[0]
            support_img = parts[1]
            
            # Check if images exist
            query_path = dataset_root / query_img
            support_path = dataset_root / support_img
            
            if not query_path.exists():
                print(f"❌ Line {line_idx}: Query image not found: {query_img}")
                invalid_count += 1
                continue
            
            if not support_path.exists():
                print(f"❌ Line {line_idx}: Support image not found: {support_img}")
                invalid_count += 1
                continue
            
            # Validate bbox format if present
            if len(parts) > 2:
                try:
                    for i in range(2, len(parts), 5):
                        if i + 4 >= len(parts):
                            print(f"⚠️  Line {line_idx}: Incomplete bbox at position {i}")
                            break
                        
                        class_id = int(parts[i])
                        x_center = float(parts[i + 1])
                        y_center = float(parts[i + 2])
                        width = float(parts[i + 3])
                        height = float(parts[i + 4])
                        
                        # Check ranges
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                                0 < width <= 1 and 0 < height <= 1):
                            print(f"❌ Line {line_idx}: Invalid bbox coordinates")
                            invalid_count += 1
                            continue
                    
                    valid_count += 1
                
                except ValueError as e:
                    print(f"❌ Line {line_idx}: Invalid value in bbox - {e}")
                    invalid_count += 1
            else:
                valid_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Valid entries: {valid_count}")
    print(f"❌ Invalid entries: {invalid_count}")
    print("=" * 60)
    
    return invalid_count == 0

# Usage
if validate_dataset('/path/to/dataset', 'labels/train.txt'):
    print("✅ Dataset is valid and ready for training!")
else:
    print("❌ Please fix the issues above before training")
```

---

## 📊 Sample Statistics

### Recommended Dataset Size

| Task | Min Images | Recommended | Large |
|------|-----------|------------|-------|
| Quick test | 100 | 500-1000 | N/A |
| Development | 1000 | 5000-10000 | N/A |
| Production | 5000 | 20000-50000 | 100000+ |

### Triplet Distribution

```
Total: 10,000 triplets
├── With objects: 8,000 (80%)
└── Background: 2,000 (20%)

Object distribution:
├── Single object: 6,000 (75%)
├── Multiple objects: 2,000 (25%)
```

---

## 🚀 Training with Your Data

### Quick Start

```python
from ultralytics.models.siam_train import SiamDetectionTrainer

trainer = SiamDetectionTrainer(overrides={
    'model': 'yolo11n.yaml',
    'data': '/path/to/data.yaml',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'device': 0
})

results = trainer.train()
```

### Advanced Configuration

```python
trainer = SiamDetectionTrainer(overrides={
    'model': 'yolo11n.yaml',
    'data': '/path/to/data.yaml',
    
    # Training parameters
    'epochs': 100,
    'batch': 32,
    'imgsz': 640,
    'device': [0, 1],  # Multi-GPU
    
    # Optimizer
    'optimizer': 'SGD',
    'lr0': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    
    # Augmentation
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10,
    'translate': 0.1,
    'scale': 0.5,
    'flipud': 0.0,
    'fliplr': 0.5,
    
    # Performance
    'workers': 8,
    'cache': 'ram',  # or 'disk'
    'patience': 50,
    
    # Checkpointing
    'save': True,
    'save_period': 10
})

results = trainer.train()
```

---

## 🔗 Related Resources

- SIAM_IMPLEMENTATION_GUIDE.md - Component specifications
- SIAM_EXAMPLES.py - Code examples
- README_SIAM.md - Quick start guide
