"""
Find samples that actually have bboxes.
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

# Find samples with bboxes
samples_with_bboxes = []
print("Scanning for samples with bboxes...\n")

for idx in range(len(dataset)):
    sample = dataset[idx]
    bboxes = sample.get('bboxes')
    if bboxes is not None and len(bboxes) > 0:
        samples_with_bboxes.append(idx)
        if len(samples_with_bboxes) <= 10:  # Print first 10
            label = dataset.labels[idx]
            print(f"Sample {idx}:")
            print(f"  im_file: {Path(label.get('im_file')).name}")
            print(f"  bboxes: {len(bboxes)} - {bboxes}")
            print()

print(f"\nTotal samples with bboxes: {len(samples_with_bboxes)} / {len(dataset)}")
print(f"Percentage: {len(samples_with_bboxes)/len(dataset)*100:.1f}%")
