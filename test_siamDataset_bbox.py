"""
Direct test of SiamDataset to debug bbox loading.
"""

from pathlib import Path
import yaml
from ultralytics.data.dataset import SiamDataset

data_path = Path("dataset/data.yaml")
with open(data_path) as f:
    data_dict = yaml.safe_load(f)

train_path = data_dict.get('train')
if not Path(train_path).is_absolute():
    train_path = data_path.parent / train_path

print(f"Dataset path: {train_path}")
print(f"Data dict: {data_dict}\n")

# Build dataset
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

# Check first 5 samples from the labels directly
labels = dataset.labels
print(f"Checking first 5 labels from dataset.labels:\n")
for idx in range(min(5, len(labels))):
    label = labels[idx]
    bboxes = label.get("bboxes")
    cls_data = label.get("cls")
    print(f"Label {idx}:")
    print(f"  im_file: {Path(label.get('im_file')).name}")
    print(f"  support_file: {label.get('support_file')}")
    print(f"  bboxes type: {type(bboxes)}, shape: {bboxes.shape if hasattr(bboxes, 'shape') else 'N/A'}")
    print(f"  cls type: {type(cls_data)}, shape: {cls_data.shape if hasattr(cls_data, 'shape') else 'N/A'}")
    if hasattr(bboxes, 'shape') and len(bboxes.shape) > 0 and bboxes.shape[0] > 0:
        print(f"    bbox[0]: {bboxes[0]}")
    if hasattr(cls_data, 'shape') and len(cls_data.shape) > 0 and cls_data.shape[0] > 0:
        print(f"    cls[0]: {cls_data[0]}")
    print()

# Now check the actual __getitem__ samples
print("\nChecking first 5 samples from dataset[idx]:\n")
for idx in range(min(5, len(dataset))):
    sample = dataset[idx]
    bboxes = sample.get('bboxes')
    print(f"Sample {idx}:")
    print(f"  bboxes type: {type(bboxes)}, shape: {bboxes.shape if hasattr(bboxes, 'shape') else 'N/A'}")
    if hasattr(bboxes, 'shape') and len(bboxes) > 0:
        print(f"    bbox count: {len(bboxes)}")
        if len(bboxes) > 0:
            print(f"    bbox[0]: {bboxes[0]}")
    print()
