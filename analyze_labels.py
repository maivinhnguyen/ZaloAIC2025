"""
Analyze the label file format to understand the structure.
"""

from pathlib import Path

label_file = Path("dataset/images/train/labels/train.txt")

print("Analyzing label file format...\n")

with open(label_file) as f:
    for idx, line in enumerate(f):
        if idx >= 10:  # Just look at first 10 lines
            break
        
        parts = line.strip().split()
        print(f"Line {idx}: {len(parts)} parts")
        print(f"  Parts: {parts}")
        
        # Try to identify where bbox data starts
        # Bbox data should be floats between 0 and 1 (for normalized coords)
        for i, part in enumerate(parts):
            try:
                val = float(part)
                is_bbox = 0 <= val <= 1
                print(f"    [{i}]: {part:30s} -> {val:.4f} {'(looks like bbox)' if is_bbox else ''}")
            except ValueError:
                print(f"    [{i}]: {part:30s} -> (string/path)")
        print()
