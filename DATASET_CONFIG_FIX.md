# SiamYOLOv8 Dataset Configuration - Corrected

## 🔧 The Issue You Encountered

```
ERROR: No images found in /mlcv2/.../dataset/labels/train.txt

Reason: Your data.yaml is pointing to the WRONG path
        Framework expects image files, but got label file path
```

---

## ✅ Correct Directory Structure

### Option 1: Standard YOLO Format (RECOMMENDED)

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/
│       ├── img_101.jpg
│       └── ...
├── labels/
│   ├── train.txt       ← Triplet annotations
│   └── val.txt         ← Triplet annotations
└── data.yaml           ← Configuration file
```

### Option 2: Separate Query/Support Images

```
dataset/
├── images/
│   ├── train_query/
│   │   ├── query_001.jpg
│   │   └── ...
│   ├── train_support/
│   │   ├── support_001.jpg
│   │   └── ...
│   ├── val_query/
│   └── val_support/
├── labels/
│   ├── train.txt
│   └── val.txt
└── data.yaml
```

---

## 📄 Correct data.yaml Configuration

### ❌ WRONG (What You Have)
```yaml
path: /mlcv2/.../dataset
train: labels/train.txt          # ❌ WRONG: Points to labels file
val: labels/val.txt              # ❌ WRONG: Points to labels file

nc: 1
names: {0: 'object'}
```

**Problem:** Framework tries to load images from `labels/train.txt` file  
**Result:** `FileNotFoundError: No images found`

### ✅ CORRECT (What You Need)

**For Option 1 (Standard Format):**
```yaml
# Dataset paths
path: /mlcv2/.../dataset

# Point to IMAGE directories (not label files!)
train: images/train              # ✅ Points to images directory
val: images/val                  # ✅ Points to images directory

# Number of classes
nc: 1

# Class names
names: {0: 'object'}
```

**For Option 2 (Separate Query/Support):**
```yaml
path: /mlcv2/.../dataset

# Can use either single train path or separate query/support
train: images/train_query        # ✅ Points to query images
val: images/val_query            # ✅ Points to val query images

# Alternative with both paths (if supported)
train_query: images/train_query
train_support: images/train_support
val_query: images/val_query
val_support: images/val_support

nc: 1
names: {0: 'object'}
```

---

## 📋 Label File Format (For Reference)

Your `labels/train.txt` should look like this:

```
# Single object per image
images/train/img_001.jpg images/train/support_001.jpg 0 0.5 0.5 0.4 0.3
images/train/img_002.jpg images/train/support_002.jpg 0 0.45 0.55 0.35 0.35

# Multiple objects
images/train/img_003.jpg images/train/support_003.jpg 0 0.3 0.3 0.2 0.25 0 0.7 0.7 0.25 0.2

# Background/empty
images/train/img_004.jpg images/train/support_004.jpg
```

**Format:** `query_path support_path [class_id x_center y_center width height ...]`

---

## 🚀 How to Fix Your Setup

### Step 1: Check Your Current Structure
```bash
# List what you have
ls -la /mlcv2/.../dataset/
ls -la /mlcv2/.../dataset/images/
ls -la /mlcv2/.../dataset/labels/
```

### Step 2: Reorganize if Needed

If your structure is wrong, reorganize:

```bash
# Navigate to dataset root
cd /mlcv2/.../dataset/

# If images are misplaced, move them
mkdir -p images/train images/val

# Move query images
mv query_images/*.jpg images/train/
mv query_images_val/*.jpg images/val/

# Or if you want separate query/support structure
mkdir -p images/train_query images/train_support
mkdir -p images/val_query images/val_support

mv query_images/*.jpg images/train_query/
mv support_images/*.jpg images/train_support/
```

### Step 3: Create/Update data.yaml

**Create file: `/mlcv2/.../dataset/data.yaml`**

```yaml
# Dataset root path (absolute or relative)
path: /mlcv2/WorkingSpace/Personal/nguyenmv/ZaloAI/Repo/maibel/src/dataset

# IMAGE directories (not label files!)
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 1

# Class names
names:
  0: 'object'
```

### Step 4: Update Your Training Script

**Verify your train.py uses the correct data.yaml path:**

```python
from ultralytics.models.siam_train import SiamDetectionTrainer

trainer = SiamDetectionTrainer(overrides={
    'model': 'yolo11n.yaml',
    'data': '/mlcv2/.../dataset/data.yaml',  # ✅ Correct path
    'epochs': 100,
    'batch': 16,
    'device': 0
})

results = trainer.train()
```

---

## 📊 Expected File Tree After Fix

```
/mlcv2/.../dataset/
├── data.yaml                    ← Configuration file
├── images/
│   ├── train/                   ← Query images for training
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ... (total N images)
│   ├── val/                     ← Query images for validation
│   │   ├── img_101.jpg
│   │   └── ... (total M images)
│   └── test/
│       └── ...
├── labels/
│   ├── train.txt                ← Triplet annotations for train
│   ├── val.txt                  ← Triplet annotations for val
│   └── test.txt
└── images_support/              ← Optional: separate support images
    ├── train/
    │   ├── support_001.jpg
    │   └── ...
    └── val/
        ├── support_101.jpg
        └── ...
```

---

## 🔍 How SiamDataset Loads Data

```python
# The framework does this:

# 1. Read data.yaml
with open('data.yaml') as f:
    data = yaml.load(f)

# 2. Get train/val paths
train_path = data['train']           # "images/train"
val_path = data['val']               # "images/val"

# 3. Get image files from those directories
img_files = get_img_files(train_path)  # Looks in images/train/ for *.jpg

# 4. Read label file
with open('labels/train.txt') as f:
    for line in f:
        query_img, support_img, *boxes = line.split()
        # Load query_img and support_img from disk
```

**Key Point:** 
- `data.yaml` tells where IMAGES are (`train: images/train`)
- Label files tell which specific images to load (`labels/train.txt`)

---

## ✅ Verification Checklist

- [ ] `data.yaml` has `train: images/train` (NOT `labels/train.txt`)
- [ ] `data.yaml` has `val: images/val` (NOT `labels/val.txt`)
- [ ] Images actually exist in `images/train/` and `images/val/`
- [ ] Label files exist in `labels/train.txt` and `labels/val.txt`
- [ ] Label files contain correct triplet format with image paths
- [ ] Image paths in label files are either absolute or relative correctly
- [ ] `data.yaml` is in the dataset root directory
- [ ] Script points to correct `data.yaml` location

---

## 🛠️ Python Script to Fix and Verify

```python
import os
import yaml
from pathlib import Path

def verify_and_fix_dataset(dataset_root):
    """Verify and help fix dataset structure."""
    
    dataset_root = Path(dataset_root)
    
    print("="*60)
    print("SiamYOLOv8 Dataset Verification")
    print("="*60)
    
    # Check required directories
    required_dirs = {
        'images/train': 'Training images directory',
        'images/val': 'Validation images directory',
        'labels': 'Labels directory'
    }
    
    for dir_path, desc in required_dirs.items():
        full_path = dataset_root / dir_path
        if full_path.exists():
            print(f"✓ {desc}: {dir_path}")
        else:
            print(f"✗ {desc}: {dir_path} - MISSING!")
    
    # Check required files
    required_files = {
        'labels/train.txt': 'Training annotations',
        'labels/val.txt': 'Validation annotations',
    }
    
    for file_path, desc in required_files.items():
        full_path = dataset_root / file_path
        if full_path.exists():
            line_count = len(open(full_path).readlines())
            print(f"✓ {desc}: {file_path} ({line_count} entries)")
        else:
            print(f"✗ {desc}: {file_path} - MISSING!")
    
    # Check data.yaml
    data_yaml_path = dataset_root / 'data.yaml'
    if data_yaml_path.exists():
        with open(data_yaml_path) as f:
            data = yaml.safe_load(f)
        
        print(f"\n✓ data.yaml found")
        print(f"  - train: {data.get('train', 'NOT SET')}")
        print(f"  - val: {data.get('val', 'NOT SET')}")
        print(f"  - nc: {data.get('nc', 'NOT SET')}")
        
        # Verify train/val paths point to images, not labels
        train_path = data.get('train', '')
        val_path = data.get('val', '')
        
        if 'label' in train_path.lower():
            print(f"  ✗ WARNING: train path looks like labels!")
        if 'label' in val_path.lower():
            print(f"  ✗ WARNING: val path looks like labels!")
    else:
        print(f"✗ data.yaml not found!")
    
    # Count actual images
    print(f"\nImage Count:")
    train_imgs = len(list((dataset_root / 'images/train').glob('*.jpg')))
    train_imgs += len(list((dataset_root / 'images/train').glob('*.png')))
    val_imgs = len(list((dataset_root / 'images/val').glob('*.jpg')))
    val_imgs += len(list((dataset_root / 'images/val').glob('*.png')))
    
    print(f"  - Training images: {train_imgs}")
    print(f"  - Validation images: {val_imgs}")
    
    print("\n" + "="*60)

# Run verification
verify_and_fix_dataset('/mlcv2/.../dataset/')
```

---

## 🚀 After Fixing, Test Again

```bash
# Verify setup
python -c "
from pathlib import Path
dataset = Path('/mlcv2/.../dataset')
print('Images in train:', len(list((dataset / 'images/train').glob('*.jpg'))))
print('Images in val:', len(list((dataset / 'images/val').glob('*.jpg'))))
with open(dataset / 'labels/train.txt') as f:
    print('Annotations in train.txt:', len(f.readlines()))
"

# Try training again
python train.py --data /mlcv2/.../dataset/data.yaml
```

---

## 📚 Summary

| Component | Should Point To | ❌ Wrong | ✅ Correct |
|-----------|-----------------|---------|----------|
| `data.yaml` train | Image directory | `labels/train.txt` | `images/train` |
| `data.yaml` val | Image directory | `labels/val.txt` | `images/val` |
| Label file | Image triplets | N/A | `labels/train.txt` |
| Framework finds | Actual images | In labels/ | In images/ |

The key insight: **data.yaml tells where images are, label files tell which images to use.**

---

**Issue:** Configuration pointed to labels instead of images  
**Fix:** Update data.yaml to point to image directories  
**Result:** Training should work! 🚀
