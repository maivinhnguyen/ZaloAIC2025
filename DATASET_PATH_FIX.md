# Dataset Path Fix - curate_data.py

## Problem (Initial)
Support images were being looked for in wrong locations during training:

**Error 1 (Initial fix attempt):**
```
val: Support image missing for /path/dataset/images/val/Laptop_0_frame_004789.jpg
Looking for: /path/dataset/images/support/Laptop_0_img_1.jpg
```
Expected: `/path/dataset/images/val/support/Laptop_0_img_1.jpg`

**Error 2 (After first fix):**
```
train: Support image missing for /path/dataset/images/train/train/query/MobilePhone_0_frame_002595.jpg
Looking for: /path/dataset/images/train/train/support/MobilePhone_0_img_1.jpg
```
❌ Duplicate `train` in path!

## Root Cause
The dataset loader determines the dataset root by going up 2 levels from the label file:

```python
dataset_root = Path(label_file).parent.parent
# If label_file = images/train/labels/train.txt
# Then dataset_root = images/train/ (NOT images/)
```

The problem was computing paths relative to the wrong level. First attempt used `images/` but should use `images/train/`.

## Solution
Compute paths relative to `images/{train,val}/` directory (the actual dataset_root), not `images/`.

### Directory Structure
```
output_dir/
  images/
    train/                     <- Dataset root for train labels
      query/                   <- Query images
      support/                 <- Support images  
      labels/
        train.txt              <- Triplet label file
    val/                       <- Dataset root for val labels
      query/                   <- Query images
      support/                 <- Support images
      labels/
        val.txt                <- Triplet label file
    data.yaml
```

### Path Resolution (Correct)
**Dataset loader code:**
```python
dataset_root = Path(label_file).parent.parent
# If label_file = images/train/labels/train.txt
# Then dataset_root = images/train/
```

**Triplet label file content** (`images/train/labels/train.txt`):
```
query/MobilePhone_0_frame_002595.jpg support/MobilePhone_0_img_1.jpg 0 0.5 0.5 0.1 0.1
```

**Resolution process**:
1. Label file: `images/train/labels/train.txt`
2. Dataset root: `images/train/` (going up 2 levels)
3. Query resolved: `images/train/` + `query/MobilePhone_0_frame_002595.jpg` = ✅ `images/train/query/MobilePhone_0_frame_002595.jpg`
4. Support resolved: `images/train/` + `support/MobilePhone_0_img_1.jpg` = ✅ `images/train/support/MobilePhone_0_img_1.jpg`

## Code Changes
Modified `move_single_triplet()` function to:
1. Accept `img_set_root` parameter (the `images/train/` or `images/val/` directory)
2. Compute all paths relative to `img_set_root` using `os.path.relpath()`
3. Pass `img_set_root = base_out_dir / "images" / img_set` when calling from `move_files_and_write_triplets()`

### Before (Incorrect)
```python
# Tried to use images/ as root, but loader uses images/train/
img_out_root = base_out_dir / "images"
q_rel_path = os.path.relpath(q_final_path, img_out_root)  # train/query/file.jpg
# Result: images/ + train/query/file.jpg = images/train/query/file.jpg ✓
# BUT when combined with dataset_root = images/train/:
# images/train/ + train/query/file.jpg = images/train/train/query/file.jpg ❌ DOUBLED!
```

### After (Correct)
```python
# Use the same level as dataset_root calculation
img_set_root = base_out_dir / "images" / img_set  # images/train/ or images/val/
q_rel_path = os.path.relpath(q_final_path, img_set_root)  # query/file.jpg
# Result: images/train/ + query/file.jpg = images/train/query/file.jpg ✓
```

## Verification
The fix ensures that when the dataset loader loads label files, it can correctly find all support images:
- Query images: `images/{train,val}/query/`
- Support images: `images/{train,val}/support/`

All paths are now resolvable from the correct dataset root with NO DUPLICATION.
