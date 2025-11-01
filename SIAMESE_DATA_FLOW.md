# Siamese Network Training - Data Flow Diagram

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  dataset/images/train/                                           │
│  ├── query/                                                      │
│  │   ├── WaterBottle_1_frame_004212.jpg    ← Target scene       │
│  │   ├── Jacket_0_frame_004703.jpg                              │
│  │   └── ...                                                     │
│  ├── support/                                                    │
│  │   ├── WaterBottle_1_img_1.jpg           ← Reference images   │
│  │   ├── WaterBottle_1_img_2.jpg                                │
│  │   ├── WaterBottle_1_img_3.jpg                                │
│  │   └── ...                                                     │
│  └── labels/                                                     │
│      └── train.txt                                               │
│          query\X.jpg support\A.jpg support\B.jpg support\C.jpg │
│          0 0.534 0.341 0.041 0.074                              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              DATASET LOADING (SiamDataset)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  For each sample index i:                                        │
│                                                                   │
│  1. Load Query Image (with labels)                              │
│     ├── Read: dataset/images/train/query/X.jpg                 │
│     ├── Load annotation: class=0, bbox=(0.534, 0.341, 0.041)  │
│     └── Result: query_img (H, W, 3), cls, bboxes              │
│                                                                   │
│  2. Load Support Image (NO labels!)                             │
│     ├── Read: dataset/images/train/support/RANDOM.jpg          │
│     ├── NO annotation applied ← KEY FIX                         │
│     ├── Clear: cls=[], bboxes=[]                                │
│     └── Result: support_img (H, W, 3), cls=[], bboxes=[]      │
│                                                                   │
│  3. Apply Transforms with same seed                             │
│     ├── Query transforms:   Resize, Augment, Add bboxes        │
│     ├── Support transforms: Resize, Augment, NO bboxes         │
│     └── Result: Both same size, support has no annotations     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BATCH COLLATION                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  batch = {                                                       │
│      "query_img":    Tensor(B, 3, 640, 640)  ← Query images    │
│      "img":         Tensor(B, 3, 640, 640)  ← Same as query    │
│      "support_img":  Tensor(B, 3, 640, 640)  ← Support images  │
│      "cls":         Tensor(B, N_objects)     ← Query labels    │
│      "bboxes":      Tensor(B, N_objects, 4)  ← Query boxes    │
│      "batch_idx":   Tensor(B*N_objects)                        │
│  }                                                               │
│                                                                   │
│  WHERE:                                                          │
│  - B = batch size (e.g., 4)                                     │
│  - N_objects = max objects in batch                             │
│  - cls: Empty for support images!                               │
│  - bboxes: Empty for support images!                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PREPROCESSING (preprocess_batch)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Move to device (GPU/CPU)                                    │
│  2. Normalize:                                                   │
│     ├── query_img = query_img / 255                             │
│     └── support_img = support_img / 255                         │
│  3. Optional multi-scale augmentation                           │
│                                                                   │
│  Result: Same tensors, normalized to [0, 1]                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│          FORWARD PASS (SiamDetectionModel)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  model.forward(query_img, support_img)                          │
│  │                                                               │
│  ├─ BACKBONE (shared parameters)                               │
│  │  ├─ query_feat = backbone(query_img)                        │
│  │  │  Shape: [(B, C₁, H₁, W₁), (B, C₂, H₂, W₂), ...]        │
│  │  │                                                            │
│  │  └─ support_feat = backbone(support_img)                    │
│  │     Shape: [(B, C₁, H₁, W₁), (B, C₂, H₂, W₂), ...]        │
│  │                                                               │
│  ├─ MATCHING MODULES (P3, P4, P5 scales)                      │
│  │  ├─ fused_p3 = MatchingModule(query_feat[0], support_feat[0])│
│  │  ├─ fused_p4 = MatchingModule(query_feat[1], support_feat[1])│
│  │  └─ fused_p5 = MatchingModule(query_feat[2], support_feat[2])│
│  │  Formula: Output = Q + sigmoid(Q ⊙ S) ⊙ S                  │
│  │                                                               │
│  ├─ NECK (PathPAN or similar)                                  │
│  │  Fuses matched features [fused_p3, fused_p4, fused_p5]      │
│  │                                                               │
│  └─ DETECTION HEAD (only on fused features)                   │
│     └─ preds = head(fused_features)                            │
│        Shape: (B, num_anchors, 4+1+nc)                         │
│                                                                   │
│  KEY: Detection head ONLY sees fused features                   │
│       No direct supervision on support!                         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           LOSS COMPUTATION (SiamLoss)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ONLY uses query image labels!                                  │
│                                                                   │
│  loss = criterion(preds, batch["cls"], batch["bboxes"])        │
│                            ↓                                     │
│  Computes:                                                       │
│  - IoU Loss (7.5x weight)                                      │
│  - BCE Loss (0.5x weight)                                      │
│  - RPL Loss (0.5x weight) - aspect ratio preservation          │
│  - DICE Loss (0.5x weight)                                     │
│  - DFL Loss (1.5x weight)                                      │
│                                                                   │
│  Support images contribute indirectly through:                  │
│  - Backbone parameters (shared learning)                       │
│  - Matching module (feature alignment learning)                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         BACKWARD PASS & OPTIMIZATION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  loss.backward()                                                 │
│  optimizer.step()                                                │
│                                                                   │
│  Updated parameters:                                             │
│  ✓ Backbone (learns to extract good features for both)         │
│  ✓ Matching Modules (learns to align features)                 │
│  ✓ Neck (learns to process fused features)                     │
│  ✓ Detection Head (learns to detect on fused features)         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Differences: Query vs Support

| Aspect | Query Image | Support Image |
|--------|-------------|---------------|
| **Source** | `query/` folder | `support/` folder |
| **Content** | Scene with target object | Reference showing target |
| **Labels** | ✓ Has bounding boxes | ✗ NO bounding boxes |
| **Backbone** | ✓ Processes features | ✓ Processes features |
| **Supervision** | ✓ Gets detection loss | ✗ No direct loss |
| **MatchingModule** | Input: query features | Input: support features |
| **Learning** | Learns to detect + match | Learns to match features |

## The MatchingModule

```python
def forward(query_feat, support_feat):
    """
    Fuses query and support features.
    
    Formula: Output = Query + sigmoid(Query ⊙ Support) ⊙ Support
    
    Where ⊙ is element-wise multiplication
    """
    # Compute element-wise similarity
    similarity = torch.sigmoid(query_feat * support_feat)
    
    # Weight support features by similarity
    weighted_support = similarity * support_feat
    
    # Add query features (residual connection)
    output = query_feat + weighted_support
    
    return output
```

This learns to emphasize query features that are similar to support features!

## Why This Fix Matters

### Before Fix ❌
- Support images had bounding boxes
- Network learned to detect on both query AND support
- Wasted parameters detecting on reference images
- Confusing training signals
- Visualization showed wrong information

### After Fix ✓
- Support images have NO bounding boxes
- Network learns to MATCH features between images
- MatchingModule learns meaningful alignment
- Clear training signals
- Visualization shows exactly what model learns
