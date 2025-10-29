"""
SiamYOLOv8 Quick Start Guide and Examples

This file demonstrates how to use the implemented SiamYOLOv8 components
for one-shot object detection.
"""

import torch
from ultralytics.nn.modules import MatchingModule
from ultralytics.nn.tasks import SiamDetectionModel
from ultralytics.utils.loss import SiamLoss, RatioPreservingLoss, DiceLoss
from ultralytics.data.dataset import SiamDataset


# Example 1: Using the MatchingModule
def example_matching_module():
    """Demonstrate the MatchingModule for feature fusion."""
    print("=" * 60)
    print("Example 1: MatchingModule")
    print("=" * 60)

    # Initialize the module
    mm = MatchingModule()

    # Create sample feature maps
    batch_size = 4
    channels = 256
    height = width = 40

    query_features = torch.randn(batch_size, channels, height, width)
    support_features = torch.randn(batch_size, channels, height, width)

    # Forward pass - applies: Q + s(Q × S) × S
    fused_features = mm(query_features, support_features)

    print(f"Query features shape: {query_features.shape}")
    print(f"Support features shape: {support_features.shape}")
    print(f"Fused features shape: {fused_features.shape}")
    print(f"✓ Fusion successful!\n")


# Example 2: Creating and using SiamDetectionModel
def example_siam_detection_model():
    """Demonstrate the SiamDetectionModel for one-shot detection."""
    print("=" * 60)
    print("Example 2: SiamDetectionModel")
    print("=" * 60)

    # Initialize model
    model = SiamDetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)

    # Create sample images
    batch_size = 2
    channels = 3
    height = width = 640

    query_img = torch.randn(batch_size, channels, height, width)
    support_img = torch.randn(batch_size, channels, height, width)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(query_img, support_img)

    print(f"Query image shape: {query_img.shape}")
    print(f"Support image shape: {support_img.shape}")
    print(f"Model output type: {type(outputs)}")
    print(f"✓ Model inference successful!\n")


# Example 3: Computing SiamLoss
def example_siam_loss():
    """Demonstrate the SiamLoss computation."""
    print("=" * 60)
    print("Example 3: SiamLoss Components")
    print("=" * 60)

    # Initialize loss components
    rpl_loss = RatioPreservingLoss()
    dice_loss = DiceLoss()

    # Create sample predictions and targets
    batch_size = 4
    num_bboxes = 100

    # Bounding boxes in (x_center, y_center, width, height) format
    pred_bboxes = torch.randn(batch_size, num_bboxes, 4) * 0.5 + 0.25
    target_bboxes = torch.randn(batch_size, num_bboxes, 4) * 0.5 + 0.25

    # Clamp to valid range
    pred_bboxes = torch.clamp(pred_bboxes, 0, 1)
    target_bboxes = torch.clamp(target_bboxes, 0, 1)

    # Compute losses
    rpl = rpl_loss(pred_bboxes, target_bboxes)
    print(f"Ratio-Preserving Loss (RPL): {rpl.item():.4f}")

    # Predictions and targets for DICE loss
    pred_scores = torch.randn(batch_size, num_bboxes, 1)
    target_scores = torch.randint(0, 2, (batch_size, num_bboxes, 1)).float()

    dice = dice_loss(pred_scores, target_scores)
    print(f"Dice Loss: {dice.item():.4f}")

    print(f"\n✓ Loss computation successful!")
    print(f"  Composite Loss = 7.5*L_IoU + 0.5*(L_BCE + L_RPL + L_DICE) + 1.5*L_DFL\n")


# Example 4: SiamDataset
def example_siam_dataset():
    """Demonstrate the SiamDataset for loading one-shot data."""
    print("=" * 60)
    print("Example 4: SiamDataset")
    print("=" * 60)

    print("""
SiamDataset loads data in triplet format:
  - Query image: The image where object detection is performed
  - Support image: Reference image containing the target object
  - Labels: Bounding boxes for the object of interest

Label Format (one line per query-support pair):
  query_path.jpg support_path.jpg cls x_center y_center width height

Example line:
  images/query/001.jpg images/support/001.jpg 0 0.5 0.5 0.3 0.4

Multiple objects in same image:
  images/query/001.jpg images/support/001.jpg 0 0.5 0.5 0.3 0.4 0 0.2 0.3 0.2 0.3
    """)

    print("""Key Features:
  ✓ Loads query and support image pairs
  ✓ Applies identical geometric transforms to preserve alignment
  ✓ Caches support images for efficiency
  ✓ Returns triplets as (query_img, support_img, labels)
  ✓ Batch collation with 'query_img' and 'support_img' keys

Usage:
  dataset = SiamDataset(
      img_path="/path/to/images",
      data={"names": {0: "target_object"}, "nc": 1},
      task="detect"
  )
  batch = next(iter(dataloader))
  print(batch.keys())
  # dict_keys(['query_img', 'support_img', 'bboxes', 'cls', ...])
    """)


# Example 5: Training Configuration
def example_training_config():
    """Show training configuration for SiamYOLOv8."""
    print("=" * 60)
    print("Example 5: Training Configuration")
    print("=" * 60)

    training_config = """
# Dataset Configuration (siam_coco.yaml)
path: /path/to/dataset
train: images/train
support: images/support_reference
val: images/val

nc: 1  # Single class for one-shot detection
names:
  0: "object_of_interest"

# Training Script
from ultralytics.models.siam_train import SiamDetectionTrainer

trainer = SiamDetectionTrainer(
    cfg={
        'model': 'siam_yolo11n.yaml',
        'data': 'siam_coco.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 0,
        'task': 'siam_detect'
    }
)
trainer.train()

# Inference
model = SiamDetectionModel('siam_yolo11n_best.pt')
query_img = cv2.imread('query.jpg')
support_img = cv2.imread('support.jpg')

query_tensor = torch.from_numpy(query_img).permute(2, 0, 1).unsqueeze(0).float() / 255
support_tensor = torch.from_numpy(support_img).permute(2, 0, 1).unsqueeze(0).float() / 255

with torch.no_grad():
    results = model(query_tensor, support_tensor)
    """.strip()

    print(training_config)


# Example 6: Loss Function Weights
def example_loss_weights():
    """Explain the loss function weighting scheme."""
    print("=" * 60)
    print("Example 6: SiamYOLOv8 Loss Function")
    print("=" * 60)

    print("""
Composite Loss Formula:
╔═════════════════════════════════════════════════════════════════╗
║  Loss = 7.5 × L_IoU + 0.5 × (L_BCE + L_RPL + L_DICE) + 1.5 × L_DFL  ║
╚═════════════════════════════════════════════════════════════════╝

Components:
┌──────────────────────────────────────────────────────────────────┐
│ L_IoU (IoU Loss)                     - Weight: 7.5x             │
│   Measures bounding box overlap/localization accuracy           │
│   Heavily weighted as primary detection objective               │
│                                                                  │
│ L_BCE (Binary Cross-Entropy Loss)    - Weight: 0.5x             │
│   Measures classification confidence for object presence        │
│   Part of balanced loss group                                   │
│                                                                  │
│ L_RPL (Ratio-Preserving Loss)        - Weight: 0.5x             │
│   Ensures aspect ratio consistency between predictions/targets  │
│   Helps maintain object proportions                             │
│                                                                  │
│ L_DICE (Dice Loss)                   - Weight: 0.5x             │
│   Alternative localization metric emphasizing overlap           │
│   Particularly effective for imbalanced datasets                │
│                                                                  │
│ L_DFL (Distribution Focal Loss)      - Weight: 1.5x             │
│   Fine-grained localization accuracy for better precision      │
│   Secondary focus after main IoU objective                      │
└──────────────────────────────────────────────────────────────────┘

Design Rationale:
  • IoU loss heavily weighted (7.5x) → Primary objective
  • Balanced loss group (0.5x each) → Complementary objectives
  • DFL moderately weighted (1.5x) → Fine-tuning precision
  • Total weight sum ≈ 11.0 (normalized during training)
    """)


# Example 7: Architecture Overview
def example_architecture():
    """Show the SiamYOLOv8 architecture."""
    print("=" * 60)
    print("Example 7: SiamYOLOv8 Architecture")
    print("=" * 60)

    architecture = """
SiamYOLOv8 Network Architecture
════════════════════════════════════════════════════════════════════

Query Image              Support Image
     ▼                        ▼
   [3×640×640]            [3×640×640]
         │                     │
         └─────────┬───────────┘
                   │
         [Shared Backbone]
       (YOLOv11 backbone)
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
     [P3]      [P4]       [P5]
  (64×64)   (32×32)    (16×16)
        │          │          │
        └────────────────────┘
        Query Feature Maps

     [P3]      [P4]       [P5]
  (64×64)   (32×32)    (16×16)
        │          │          │
        └────────────────────┘
        Support Feature Maps

        │          │          │
        ▼          ▼          ▼
     [MM1]     [MM2]      [MM3]
        │          │          │
        ▼          ▼          ▼
    [Fused P3] [Fused P4] [Fused P5]
        │          │          │
        └────────────────────┘
                   │
            [Detection Head]
              (FPN + PAN)
                   │
                   ▼
          [Detections]
       (Bboxes, Classes, Conf)

MM = MatchingModule (Parameter-Free)
    Q + sigmoid(Q ⊗ S) ⊗ S

Key Design Points:
  • Single shared backbone → Parameter efficiency
  • Three MatchingModules → Multi-scale fusion
  • Parameter-free MM → Lightweight fusion
  • Identical transforms → Spatial alignment
    """

    print(architecture)


def main():
    """Run all examples."""
    example_matching_module()
    example_siam_detection_model()
    example_siam_loss()
    example_siam_dataset()
    example_training_config()
    example_loss_weights()
    example_architecture()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
