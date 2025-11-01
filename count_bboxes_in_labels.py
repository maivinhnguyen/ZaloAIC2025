"""
Check how many samples in the label file have bboxes vs don't.
"""

from pathlib import Path

label_file = Path("dataset/images/train/labels/train.txt")

with_bboxes = 0
without_bboxes = 0

with open(label_file) as f:
    for idx, line in enumerate(f):
        parts = line.strip().split()
        
        # Format: query support1 support2 support3 [bbox_data...]
        # If len(parts) > 4, then there's bbox data
        if len(parts) > 4:
            with_bboxes += 1
        else:
            without_bboxes += 1
            if without_bboxes <= 5:
                print(f"Sample without bbox (line {idx}): {parts[0]}")

print(f"\nTotal lines in label file: {with_bboxes + without_bboxes}")
print(f"With bboxes: {with_bboxes} ({with_bboxes/(with_bboxes+without_bboxes)*100:.1f}%)")
print(f"Without bboxes: {without_bboxes} ({without_bboxes/(with_bboxes+without_bboxes)*100:.1f}%)")
