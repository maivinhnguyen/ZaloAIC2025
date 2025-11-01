# Siamese Network Training Data Format Analysis

## Current Data Format

The label file (train.txt) contains lines with the format:
```
query_img support_img1 support_img2 support_img3 ... class x y w h
```

Example:
```
query\WaterBottle_1_frame_004212.jpg support\WaterBottle_1_img_1.jpg support\WaterBottle_1_img_2.jpg support\WaterBottle_1_img_3.jpg 0 0.53466796875 0.3411458333333333 0.0419921875 0.07465277777777778
```

Breaking down:
- **Query Image**: `query\WaterBottle_1_frame_004212.jpg` - Has the target object to detect
- **Support Images**: 3 support images that show the target object
- **Labels**: Class index (0), and normalized bbox coordinates (x, y, w, h)

## Issues in Current Implementation

### 1. **SiamDataset._load_support_image() Logic** 
The current code only loads ONE support image, but the format provides multiple support images.

### 2. **Label Application**
According to Siamese network principles:
- **Query Image**: Should have the bounding box labels (shows what we want to find)
- **Support Images**: Should NOT have bounding box labels (they just show what the target looks like)

The current implementation applies the same labels to both query and support images, which is incorrect.

### 3. **Visualization Issue**
The plot_training_samples() function shows both query and support with the same bounding boxes, making the visualization confusing.

## Correct Siamese Training Flow

### Data Loading:
1. Load query image from `query/` folder
2. Load ONE random support image from the list provided in the label file
3. Load labels ONLY for the query image
4. Keep support image without labels

### Forward Pass:
- Query image: Goes through backbone → Detection head (with supervision)
- Support image: Goes through backbone → MatchingModule (no detection supervision)
- The matching module learns to align query features with support features

### Visualization:
- Query image: Should show with bounding box overlay (what to find)
- Support image: Should show WITHOUT bounding box (reference)

## Implementation Corrections Needed

1. **In SiamDataset.__getitem__()**:
   - Extract one random support image from the provided support images
   - Apply labels ONLY to query image
   - Clear labels for support image

2. **In plot_training_samples()**:
   - Query batch: Keep `cls` and `bboxes`
   - Support batch: Set `cls=None` and `bboxes=None` or empty

3. **In preprocess_batch()**:
   - Ensure support_data doesn't have bounding box annotations
