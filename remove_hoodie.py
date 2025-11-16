import os
import shutil

# Paths
base_path = "/mlcv2/WorkingSpace/Personal/nguyenmv/ZaloAI/Repo/maibel/ZaloAIC2025/synthetic_dataset/images/val"
labels_file = os.path.join(base_path, "labels", "val.txt")
query_dir = os.path.join(base_path, "query")
support_dir = os.path.join(base_path, "support")

# Read the labels file
with open(labels_file, 'r') as f:
    lines = f.readlines()

# Filter out lines containing 'hoodie'
filtered_lines = [line for line in lines if 'hoodie' not in line]

# Write back the filtered lines
with open(labels_file, 'w') as f:
    f.writelines(filtered_lines)

print(f"Removed {len(lines) - len(filtered_lines)} hoodie entries from {labels_file}")

# Remove hoodie query images
query_hoodie_files = [f for f in os.listdir(query_dir) if 'hoodie' in f]
for file in query_hoodie_files:
    os.remove(os.path.join(query_dir, file))
    print(f"Removed {file} from query")

# Remove hoodie support images
support_hoodie_files = [f for f in os.listdir(support_dir) if 'hoodie' in f]
for file in support_hoodie_files:
    os.remove(os.path.join(support_dir, file))
    print(f"Removed {file} from support")

print("Done removing hoodie samples from training set.")