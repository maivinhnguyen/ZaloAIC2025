# Siamese Training Fixes - Quick Reference

## What Was Wrong?

| Issue | Problem | Solution |
|-------|---------|----------|
| **Label Application** | Support images had bboxes (wrong) | Clear support image labels in `__getitem__()` |
| **Visualization** | Both query and support showed bboxes | Set support batch cls=None, bboxes=None |
| **Training Semantics** | Support was treated like query | Only query gets detection supervision |

## Files Modified

### 1. `ultralytics/data/dataset.py` - SiamDataset class

**Function:** `__getitem__()`

**Changes:**
```python
# Clear all labels from support image (lines ~1234-1238)
support_label["cls"] = np.zeros((0, 1), dtype=np.float32)
support_label["bboxes"] = np.zeros((0, 4), dtype=np.float32)
support_label["segments"] = []
support_label["keypoints"] = None

# And in the no-transforms branch (lines ~1254-1258):
if "instances" in support_data:
    instances = support_data.pop("instances")
    nl = len(instances)
    # Support data should NOT have labels
    support_data["bboxes"] = torch.zeros((0, 4), dtype=torch.float32)
    support_data["batch_idx"] = torch.zeros(0, dtype=torch.long)
    support_data["cls"] = torch.zeros((0, 1), dtype=torch.float32)
```

### 2. `ultralytics/models/siam_train.py` - SiamDetectionTrainer class

**Function 1:** `plot_training_samples()`

**Changes:**
```python
# Support batch has no labels (lines ~221-225)
support_batch = {
    "img": support_imgs,
    "cls": None,        # Support has no class labels
    "bboxes": None,     # Support has no bounding boxes
}
```

**Function 2:** `visualize_training_samples()`

**Changes:**
- Improved visualization with proper titles
- Draw bounding boxes on query images only
- Show class labels on query images
- Support images shown clean without any overlays

## How to Verify the Fix

### Step 1: Check visualization output
```bash
# After training starts, check:
# runs/detect/train*/train_batch_plots/
# - train_batch_query_*.jpg should have RED bounding boxes
# - train_batch_support_*.jpg should have NO bounding boxes
```

### Step 2: Inspect the data in training
The training should now correctly:
- Load query images with labels
- Load support images without labels
- Pass both through the backbone
- Only apply detection loss to query predictions

### Step 3: Run a test training session
```bash
python train.py --data dataset/data.yaml --epochs 1 --batch 4
```

After epoch 0, check `runs/detect/train*/train_batch_plots/` for correct visualization.

## Expected Output Structure

```
runs/detect/train1/
├── train_batch_plots/
│   ├── train_batch_query_0.jpg      ✓ Has RED bbox overlays
│   ├── train_batch_support_0.jpg    ✓ Clean image, no boxes
│   ├── train_batch_query_1.jpg      ✓ Has RED bbox overlays
│   ├── train_batch_support_1.jpg    ✓ Clean image, no boxes
│   └── ...
├── visualizations/
│   ├── epoch_0_sample_0.png         ✓ Query with bbox | Support clean
│   ├── epoch_0_sample_1.png         ✓ Query with bbox | Support clean
│   └── ...
└── weights/
    └── ...
```

## Key Principle: One-Shot Detection

**Query Image (Left):**
- Scene with unknown objects
- Has ground truth: "Find object at (x, y)"
- Network learns: "Where is this object?"
- Gets detection loss

**Support Image (Right):**
- Reference showing what target looks like
- Has NO labels (no ground truth annotations)
- Network learns: "Match these features"
- Gets feature matching through MatchingModule

The model learns to detect by matching query features with support reference features!

## Troubleshooting

### Q: Visualizations still show boxes on support?
**A:** Check if plot is using cached data. Delete `.cache` files and retrain.

### Q: Support images look wrong?
**A:** Verify support image paths in `train.txt` are correct and files exist.

### Q: Loss not decreasing?
**A:** Check that query images have valid bounding boxes in label files.

### Q: Training is slower?
**A:** Normal - Siamese networks are more complex than standard detection.
