"""
Test script to verify SiamDataset can load data properly.
"""

import sys
from pathlib import Path
import tempfile
import os

# Add local ultralytics to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing SiamDataset data loading...\n")

# Create a temporary dataset for testing
test_dir = Path(tempfile.mkdtemp())
print(f"Creating test dataset in: {test_dir}\n")

# Create directory structure
(test_dir / "images/train").mkdir(parents=True)
(test_dir / "images/val").mkdir(parents=True)
(test_dir / "labels").mkdir(parents=True)

# Create dummy images
import cv2
import numpy as np

def create_dummy_image(path, name):
    """Create a dummy image for testing."""
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img_path = path / f"{name}.jpg"
    cv2.imwrite(str(img_path), img)
    return img_path

# Create training images
print("Creating training images...")
for i in range(5):
    create_dummy_image(test_dir / "images/train", f"img_{i:03d}")
    create_dummy_image(test_dir / "images/train", f"support_{i:03d}")

# Create validation images
print("Creating validation images...")
for i in range(2):
    create_dummy_image(test_dir / "images/val", f"img_{i+100:03d}")
    create_dummy_image(test_dir / "images/val", f"support_{i+100:03d}")

# Create label files
print("Creating label files...")

train_labels = """images/train/img_000.jpg images/train/support_000.jpg 0 0.5 0.5 0.4 0.3
images/train/img_001.jpg images/train/support_001.jpg 0 0.45 0.55 0.35 0.35
images/train/img_002.jpg images/train/support_002.jpg 0 0.3 0.3 0.2 0.25 0 0.7 0.7 0.25 0.2
images/train/img_003.jpg images/train/support_003.jpg
images/train/img_004.jpg images/train/support_004.jpg 0 0.5 0.5 0.3 0.4
"""

val_labels = """images/val/img_100.jpg images/val/support_100.jpg 0 0.5 0.5 0.4 0.3
images/val/img_101.jpg images/val/support_101.jpg
"""

with open(test_dir / "labels/train.txt", "w") as f:
    f.write(train_labels)

with open(test_dir / "labels/val.txt", "w") as f:
    f.write(val_labels)

# Create data.yaml
import yaml

data_config = {
    'path': str(test_dir.absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,
    'names': {0: 'object'}
}

with open(test_dir / "data.yaml", "w") as f:
    yaml.dump(data_config, f)

print(f"✓ Test dataset created\n")

# Now try to load the dataset
print("="*60)
print("Testing SiamDataset Loading")
print("="*60)

try:
    from ultralytics.data.dataset import SiamDataset
    
    print("\n1. Creating SiamDataset...")
    dataset = SiamDataset(
        img_path=str(test_dir / "images/train"),
        imgsz=640,
        batch_size=2,
        augment=False,
        cache=False,
        pad=0.5,
        rect=False,
        stride=32,
        names={0: 'object'},
        data={'nc': 1, 'names': {0: 'object'}}
    )
    print(f"   ✓ Dataset created with {len(dataset)} samples")
    
    print("\n2. Getting first sample...")
    sample = dataset[0]
    print(f"   ✓ Sample loaded")
    print(f"   - Keys: {list(sample.keys())}")
    
    if 'query_img' in sample:
        print(f"   ✓ query_img shape: {sample['query_img'].shape}")
    
    if 'support_img' in sample:
        print(f"   ✓ support_img shape: {sample['support_img'].shape}")
    
    if 'bboxes' in sample and hasattr(sample['bboxes'], 'shape'):
        print(f"   ✓ bboxes shape: {sample['bboxes'].shape}")
    
    if 'cls' in sample and hasattr(sample['cls'], 'shape'):
        print(f"   ✓ cls shape: {sample['cls'].shape}")
    
    print("\n3. Collating batch...")
    batch = dataset.collate_fn([dataset[i] for i in range(min(2, len(dataset)))])
    print(f"   ✓ Batch created")
    print(f"   - Keys: {list(batch.keys())}")
    
    if 'query_img' in batch:
        print(f"   ✓ query_img batch shape: {batch['query_img'].shape}")
    
    if 'support_img' in batch:
        print(f"   ✓ support_img batch shape: {batch['support_img'].shape}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nDataset is loading correctly!")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory")
