"""
Check what's in the cache file.
"""

import pickle
from pathlib import Path

cache_file = Path("dataset/images/train/labels/.cache")

if cache_file.exists():
    try:
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        nf, nm, ne, nc, n = cache.get("results", (0, 0, 0, 0, 0))
        labels = cache.get("labels", [])
        
        print(f"Cache file found!")
        print(f"Results: nf={nf}, nm={nm}, ne={ne}, nc={nc}, n={n}")
        print(f"Total labels: {len(labels)}")
        print()
        
        # Check first few labels for bbox data
        for idx in range(min(5, len(labels))):
            label = labels[idx]
            bboxes = label.get("bboxes")
            cls_data = label.get("cls")
            print(f"Label {idx}:")
            print(f"  im_file: {label.get('im_file')}")
            print(f"  bboxes shape: {bboxes.shape if hasattr(bboxes, 'shape') else len(bboxes)}")
            print(f"  cls shape: {cls_data.shape if hasattr(cls_data, 'shape') else len(cls_data)}")
            if hasattr(bboxes, 'shape') and bboxes.shape[0] > 0:
                print(f"    First bbox: {bboxes[0]}")
            if hasattr(cls_data, 'shape') and cls_data.shape[0] > 0:
                print(f"    First cls: {cls_data[0]}")
            print()
    except Exception as e:
        print(f"Error reading cache: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Cache file not found")
