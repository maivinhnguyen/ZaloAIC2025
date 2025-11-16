import argparse
import os
import random
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2


def resize_image(img_path, max_size=1280):
    """
    Resize image if it's larger than max_size, maintaining aspect ratio.
    Returns the resized image as numpy array.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Only resize if image is larger than max_size
    if max(h, w) > max_size:
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img


def process_single_category(args):
    """
    Process a single category directory.
    Returns triplet data: (query_img_path, support_img_paths, bbox_line)
    """
    category_dir, staging_dir, max_size = args
    category_name = category_dir.name
    triplet_data = []
    
    # --- 1. Get paths ---
    images_dir = category_dir / "images"
    labels_dir = category_dir / "labels"
    support_dir = category_dir / "support"
    
    if not images_dir.exists() or not labels_dir.exists() or not support_dir.exists():
        return triplet_data
    
    # --- 2. Stage Support Images ---
    support_imgs = sorted(list(support_dir.glob('*.png')) + list(support_dir.glob('*.jpg')))
    
    if len(support_imgs) < 3:
        print(f"⚠️  Warning: {category_name} has only {len(support_imgs)} support images, skipping...")
        return triplet_data
    
    # Use first 3 support images
    support_imgs = support_imgs[:3]
    
    support_stage_dir = staging_dir / "support"
    support_stage_dir.mkdir(parents=True, exist_ok=True)
    
    staged_support_paths = []
    for idx, s_img_in in enumerate(support_imgs):
        s_out_name = f"{category_name}_support_{idx+1}{s_img_in.suffix}"
        s_out_abs_path = support_stage_dir / s_out_name
        shutil.copy2(s_img_in, s_out_abs_path)
        staged_support_paths.append(s_out_abs_path)
    
    # --- 3. Process Query Images and Labels ---
    query_stage_dir = staging_dir / "query"
    query_stage_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all query images (combine and sort together to maintain proper order)
    query_images = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))
    
    for query_img in query_images:
        # Find corresponding label file
        label_file = labels_dir / f"{query_img.stem}.txt"
        
        # Read bounding boxes from label file (YOLO format)
        bboxes = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        bboxes.append(line)
        
        # Stage query image with resizing
        q_out_name = f"{category_name}_{query_img.name}"
        q_out_abs_path = query_stage_dir / q_out_name
        
        # Resize image to reasonable resolution
        img = resize_image(query_img, max_size=max_size)
        if img is not None:
            # Save with same extension but handle PNG vs JPG
            if q_out_abs_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                cv2.imwrite(str(q_out_abs_path), img)
            else:
                # Default to jpg if unknown extension
                q_out_abs_path = q_out_abs_path.with_suffix('.jpg')
                cv2.imwrite(str(q_out_abs_path), img)
        else:
            # Fallback: just copy if resize fails
            shutil.copy2(query_img, q_out_abs_path)
        
        # Store triplet data
        # Format: (query_path, support_paths, bboxes_list)
        triplet_data.append(
            (q_out_abs_path, staged_support_paths, bboxes)
        )
    
    return triplet_data


def process_all_categories(category_dirs, staging_dir, max_size=1280, num_workers=None):
    """
    Processes all category directories in parallel.
    Returns a list of all triplet data.
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    # Prepare arguments for each category
    args_list = [
        (category_dir, staging_dir, max_size)
        for category_dir in category_dirs
    ]
    
    all_triplet_data = []
    
    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_single_category, args_list),
            total=len(category_dirs),
            desc="Pass 1/2: Processing all categories"
        ):
            all_triplet_data.extend(result)
    
    return all_triplet_data


def move_single_triplet(args):
    """
    Move a single triplet's files from staging to final location.
    Creates proper triplet format line.
    """
    q_stage_path, s_stage_names, bboxes, query_out_dir, support_out_dir, img_set_root = args
    
    # 1. Move Query Image
    q_final_path = query_out_dir / q_stage_path.name
    shutil.move(str(q_stage_path), str(q_final_path))
    
    # Compute paths relative to img_set_root (train/ or val/ directory)
    q_rel_path = os.path.relpath(q_final_path, img_set_root)
    
    # 2. Get support image relative paths
    s_rel_paths = []
    for s_name in s_stage_names:
        s_final_path = support_out_dir / s_name
        s_rel_path = os.path.relpath(s_final_path, img_set_root)
        s_rel_paths.append(str(s_rel_path))
    
    # 3. Format triplet line
    # Format: <query_img> <support_img1> <support_img2> <support_img3> [<class> <x> <y> <w> <h> ...]
    line_parts = [str(q_rel_path)]
    line_parts.extend(s_rel_paths)
    
    # Add bboxes (already in YOLO format from label files)
    for bbox in bboxes:
        line_parts.extend(bbox.split())
    
    line = " ".join(line_parts)
    return line


def move_files_and_write_triplets(
    triplet_data_list, img_set, base_out_dir, num_workers=None
):
    """
    Moves files from staging to final dir, creates triplet format lines.
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    img_out_dir = base_out_dir / "images" / img_set
    query_out_dir = img_out_dir / "query"
    support_out_dir = img_out_dir / "support"
    labels_out_dir = img_out_dir / "labels"
    
    query_out_dir.mkdir(parents=True, exist_ok=True)
    support_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Step 1: Copy all unique support images SEQUENTIALLY ---
    print(f"Copying support images for {img_set}...")
    all_unique_support_items = set()
    
    for (q_stage_path, s_stage_paths, bboxes) in triplet_data_list:
        for s_stage_path in s_stage_paths:
            all_unique_support_items.add(s_stage_path)
    
    for s_stage_path in all_unique_support_items:
        s_final_path = support_out_dir / s_stage_path.name
        if not s_final_path.exists():
            shutil.copy2(str(s_stage_path), str(s_final_path))
    
    # --- Step 2: Move query files and generate triplet lines IN PARALLEL ---
    img_set_root = base_out_dir / "images" / img_set
    args_list = [
        (q_stage_path, [s_stage_path.name for s_stage_path in s_stage_paths], 
         bboxes, query_out_dir, support_out_dir, img_set_root)
        for (q_stage_path, s_stage_paths, bboxes) in triplet_data_list
    ]
    
    triplet_lines_for_txt = []
    
    with Pool(processes=num_workers) as pool:
        for line in tqdm(
            pool.imap_unordered(move_single_triplet, args_list),
            total=len(triplet_data_list),
            desc=f"Pass 2/2: Moving {img_set} files"
        ):
            triplet_lines_for_txt.append(line)
    
    # --- Step 3: Write label file ---
    label_file_path = labels_out_dir / f"{img_set}.txt"
    with open(label_file_path, 'w') as f:
        for line in triplet_lines_for_txt:
            f.write(line + "\n")
    
    print(f"✅ Written {len(triplet_lines_for_txt)} samples to {label_file_path}")
    
    return triplet_lines_for_txt


def create_yaml(output_dir):
    """
    Creates the data.yaml file with proper paths for SiamYOLO.
    """
    yaml_path = output_dir / "data.yaml"
    
    train_img_dir = output_dir / "images" / "train" / "query"
    val_img_dir = output_dir / "images" / "val" / "query"
    
    train_rel = train_img_dir.relative_to(output_dir)
    val_rel = val_img_dir.relative_to(output_dir)
    
    yaml_content = f"""path: {str(output_dir.resolve())}
train: {str(train_rel)}
val: {str(val_rel)}

nc: 1
names:
  - target
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created data.yaml at {yaml_path}")


def main(args):
    print("Starting synthetic dataset preparation...")
    print(f"Using {max(1, cpu_count() - 1)} CPU cores for parallel processing")
    
    # --- 1. Define Paths ---
    synthetic_dir = Path(args.synthetic_dir).resolve()
    base_out_dir = Path(args.output_dir).resolve()
    staging_dir = base_out_dir / "staging"
    
    # Clean up previous runs
    print("Cleaning up previous output directories...")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    if (base_out_dir / "images").exists():
        shutil.rmtree(base_out_dir / "images")
    
    # Create output base directory
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    if not synthetic_dir.exists():
        print(f"❌ Error: Synthetic directory {synthetic_dir} does not exist")
        return
    
    # --- 2. Get all category directories ---
    category_dirs = sorted([d for d in synthetic_dir.iterdir() if d.is_dir()])
    
    if not category_dirs:
        print(f"❌ Error: No category directories found in {synthetic_dir}")
        return
    
    print(f"Found {len(category_dirs)} category directories: {[d.name for d in category_dirs]}")
    
    # --- 3. Process All Categories into Staging Area (PARALLEL) ---
    all_triplet_data = process_all_categories(
        category_dirs, staging_dir, max_size=args.max_size, num_workers=args.num_workers
    )
    
    print(f"Total processed samples: {len(all_triplet_data)}")
    if not all_triplet_data:
        print("❌ Error: No data was processed. Check input paths and files.")
        return
    
    # --- 4. Split into Train/Val ---
    print("Splitting samples into train/val sets...")
    train_triplets, val_triplets = train_test_split(
        all_triplet_data,
        test_size=args.val_split,
        random_state=42
    )
    print(f"✅ Total samples: {len(all_triplet_data)} | "
          f"Training: {len(train_triplets)} | "
          f"Validation: {len(val_triplets)}")
    
    # --- 5. Move Files to Final Dirs & Write Triplet Files (PARALLEL) ---
    train_txt_lines = move_files_and_write_triplets(
        train_triplets, "train", base_out_dir,
        num_workers=args.num_workers
    )
    
    val_txt_lines = move_files_and_write_triplets(
        val_triplets, "val", base_out_dir,
        num_workers=args.num_workers
    )
    
    # --- 6. Clean up Staging Directory ---
    print("Cleaning up staging directory...")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    
    # --- 7. Create data.yaml ---
    print("Creating data.yaml file...")
    create_yaml(base_out_dir)
    
    # --- 8. Print Summary ---
    print("-" * 50)
    print("✅ Synthetic dataset preparation complete!")
    print(f"Output directory: {base_out_dir}")
    print(f"Config file: {base_out_dir / 'data.yaml'}")
    print(f"Image structure:")
    print(f"  - {base_out_dir / 'images' / 'train' / '{query,support,labels}'}")
    print(f"  - {base_out_dir / 'images' / 'val' / '{query,support,labels}'}")
    print(f"Label files:")
    print(f"  - {base_out_dir / 'images' / 'train' / 'labels' / 'train.txt'}")
    print(f"  - {base_out_dir / 'images' / 'val' / 'labels' / 'val.txt'}")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert synthetic data to SiamYOLO training format."
    )
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        required=True,
        help="Path to the Synthetic directory containing category folders."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the destination directory for processed data."
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of samples to use for validation (default: 0.1)."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of CPU cores to use (default: 4)."
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1280,
        help="Maximum image dimension for query images (default: 1280)."
    )
    
    args = parser.parse_args()
    main(args)
