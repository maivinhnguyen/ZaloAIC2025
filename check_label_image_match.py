"""
Check if image files match label file entries.
"""

from pathlib import Path

img_dir = Path("dataset/images/train/query")
label_file = Path("dataset/images/train/labels/train.txt")

# Get query image names from label file
label_queries = []
with open(label_file) as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.split()
        query_token = parts[0]
        label_queries.append(Path(query_token).name)

# Get actual image files
img_files = sorted([f.name for f in img_dir.glob("*.jpg")])

print(f"Total images in query folder: {len(img_files)}")
print(f"Total entries in label file: {len(label_queries)}")

print(f"\nFirst 10 images in folder:")
for i, img in enumerate(img_files[:10]):
    print(f"  [{i}] {img}")

print(f"\nFirst 10 queries in label file:")
for i, q in enumerate(label_queries[:10]):
    print(f"  [{i}] {q}")

# Check if they're in the same order
print(f"\nMatching first entry:")
print(f"  Image[0]: {img_files[0]}")
print(f"  Label[0]: {label_queries[0]}")
print(f"  Match: {img_files[0] == label_queries[0]}")

# Count matches
matches = sum(1 for i in range(min(len(img_files), len(label_queries))) if img_files[i] == label_queries[i])
print(f"\nMatching entries: {matches} out of {min(len(img_files), len(label_queries))}")

# Check if all label queries are in the images
in_images = sum(1 for q in label_queries if q in img_files)
print(f"Label queries that exist in images: {in_images} out of {len(label_queries)}")
