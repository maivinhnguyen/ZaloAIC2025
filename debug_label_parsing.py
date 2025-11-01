"""
Debug script to check label parsing logic.
"""

from pathlib import Path

label_file = Path("dataset/images/train/labels/train.txt")

print("Testing label parsing logic...\n")

with open(label_file) as f:
    for idx, line in enumerate(f):
        if idx >= 5:  # Just test first 5 lines
            break
        
        parts = line.strip().split()
        print(f"Line {idx}: {len(parts)} parts")
        
        query_token = parts[0]
        print(f"  Query: {query_token}")
        
        # Find where bbox data starts
        bbox_start_idx = 1
        support_paths = []
        
        for i in range(1, len(parts)):
            part = parts[i]
            print(f"    [{i}] {part:40s}", end="")
            
            # Check if this looks like a file path
            if '/' in part or '\\' in part or '.' in part:
                print("-> PATH")
                support_paths.append(part)
                bbox_start_idx = i + 1
            else:
                # Try to parse as float
                try:
                    val = float(part)
                    print(f"-> FLOAT ({val:.4f}) **BBOX_START**")
                    bbox_start_idx = i
                    break
                except ValueError:
                    print("-> UNKNOWN")
                    support_paths.append(part)
                    bbox_start_idx = i + 1
        
        print(f"  Support paths: {support_paths}")
        print(f"  Bbox start index: {bbox_start_idx}")
        print(f"  Bbox parts: {parts[bbox_start_idx:]}")
        print()
