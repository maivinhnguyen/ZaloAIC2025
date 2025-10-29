import argparse
import json
import os
import random
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from threading import Lock

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# (x1, y1, x2, y2) to (class, x_center, y_center, width, height)
def convert_to_yolo(bbox, img_w, img_h):
    """Converts a [x1, y1, x2, y2] bbox to YOLO [0, x_c, y_c, w, h] format."""
    x1, y1, x2, y2 = bbox
    
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    
    # Clamp values to [0.0, 1.0] to avoid errors
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    width = np.clip(width, 0, 1)
    height = np.clip(height, 0, 1)
    
    return [0, x_center, y_center, width, height]

def process_single_video(args):
    """
    Process a single video. This function is designed to run in parallel.
    Returns triplet data: (query_path, support_paths, bboxes, is_positive)
    """
    video_dir, annot_map, staging_dir, frame_step, neg_ratio = args
    video_name = video_dir.name
    triplet_data = []
    
    # --- 1. Stage Support Images ---
    support_dir_in = video_dir / "object_images"
    support_imgs_in = sorted(support_dir_in.glob('*.jpg'))
    
    if len(support_imgs_in) != 3:
        return triplet_data  # Return empty list if validation fails
        
    support_stage_dir = staging_dir / "support"
    support_stage_dir.mkdir(parents=True, exist_ok=True)
    
    staged_support_paths = []
    for s_img_in in support_imgs_in:
        s_out_name = f"{video_name}_{s_img_in.name}"
        s_out_abs_path = support_stage_dir / s_out_name
        shutil.copy2(s_img_in, s_out_abs_path)
        staged_support_paths.append(s_out_abs_path)

    # --- 2. Process Video and Annotations ---
    video_path = video_dir / "drone_video.mp4"
    if not video_path.exists():
        return triplet_data
        
    cap = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0 or frame_w == 0 or frame_h == 0:
        cap.release()
        return triplet_data
        
    video_annots = annot_map.get(video_name)
    if not video_annots:
        cap.release()
        return triplet_data

    # --- 3. Map Frames to BBoxes (YOLO format) ---
    positive_frames = {}
    all_annotated_frames = set()
    
    for annot_group in video_annots["annotations"]:
        for bbox_data in annot_group["bboxes"]:
            frame_idx = bbox_data["frame"]
            all_annotated_frames.add(frame_idx)
            
            bbox = [bbox_data["x1"], bbox_data["y1"], 
                    bbox_data["x2"], bbox_data["y2"]]
            yolo_bbox = convert_to_yolo(bbox, frame_w, frame_h)
            
            if frame_idx not in positive_frames:
                positive_frames[frame_idx] = []
            positive_frames[frame_idx].append(yolo_bbox)

    # --- 4. Sample Positive and Negative Frames ---
    pos_frame_indices = sorted(positive_frames.keys())
    sampled_pos_indices = set(pos_frame_indices[::frame_step])
    
    all_frame_indices = set(range(total_frames))
    negative_pool = all_frame_indices - all_annotated_frames
    
    num_pos = len(sampled_pos_indices)
    num_neg_to_sample = int((num_pos / (1.0 - neg_ratio)) * neg_ratio)
    
    if len(negative_pool) < num_neg_to_sample:
        sampled_neg_indices = negative_pool
    else:
        sampled_neg_indices = set(
            random.sample(list(negative_pool), num_neg_to_sample)
        )
        
    frames_to_extract = sorted(list(
        sampled_pos_indices | sampled_neg_indices
    ))

    # --- 5. Extract Frames and Build Triplet List ---
    query_stage_dir = staging_dir / "query"
    query_stage_dir.mkdir(parents=True, exist_ok=True)
    
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx not in frames_to_extract:
            frame_idx += 1
            continue

        q_name = f"{video_name}_frame_{frame_idx:06d}.jpg"
        q_out_abs_path = query_stage_dir / q_name
        
        cv2.imwrite(str(q_out_abs_path), frame)
        
        # Store: (query_path, support_paths, bboxes, is_positive)
        is_positive = frame_idx in sampled_pos_indices
        bboxes = positive_frames.get(frame_idx, []) if is_positive else []
        
        triplet_data.append(
            (q_out_abs_path, staged_support_paths, bboxes, is_positive)
        )
        
        frame_idx += 1

    cap.release()
    return triplet_data


def process_all_videos(video_dirs, annot_map, staging_dir, 
                       frame_step, neg_ratio, num_workers=None):
    """
    Processes all videos in parallel and saves them to a single 'staging' directory.
    Returns a list of all triplet file paths.
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Use all cores except 1
    
    # Prepare arguments for each video
    args_list = [
        (video_dir, annot_map, staging_dir, frame_step, neg_ratio)
        for video_dir in video_dirs
    ]
    
    all_triplet_data = []
    
    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_single_video, args_list),
            total=len(video_dirs),
            desc="Pass 1/2: Processing all videos"
        ):
            all_triplet_data.extend(result)
    
    return all_triplet_data

def move_single_triplet(args):
    """
    Move a single triplet's files from staging to final location.
    Creates proper triplet format line.
    Note: Support images are already copied to final location.
    This function only needs to move the query image and compute relative paths.
    
    Paths are computed relative to the query directory so that when label files
    are loaded by the dataset class, the paths resolve correctly.
    """
    q_stage_path, s_stage_names, bboxes, is_positive, query_out_dir, support_out_dir = args
    
    # 1. Move Query Image
    q_final_path = query_out_dir / q_stage_path.name
    shutil.move(str(q_stage_path), str(q_final_path))
    
    # Compute paths relative to query_out_dir (where label file reader starts)
    # Use os.path.relpath to handle sibling directories correctly
    q_rel_path = os.path.relpath(q_final_path, query_out_dir)

    # 2. Get support image relative paths (relative to query_out_dir)
    # Support images are in ../support/ relative to query dir
    s_rel_paths = []
    for s_name in s_stage_names:
        s_final_path = support_out_dir / s_name
        s_rel_path = os.path.relpath(s_final_path, query_out_dir)
        s_rel_paths.append(str(s_rel_path))

    # 3. Format triplet line: query_path support1_path support2_path support3_path [bboxes...]
    # Triplet format: <query_img> <support_img1> <support_img2> <support_img3> <class> <x> <y> <w> <h> [...]
    line_parts = [str(q_rel_path)]
    line_parts.extend(s_rel_paths)
    
    # Add bboxes (already in YOLO format: [class, x_center, y_center, width, height])
    for bbox in bboxes:
        line_parts.extend(map(str, bbox))
    
    line = " ".join(line_parts)
    return line


def move_files_and_write_triplets(
    triplet_data_list, img_set, base_out_dir, num_workers=None
):
    """
    Moves files from staging to final dir, creates triplet format lines.
    
    Creates proper directory structure:
    - images/{train,val}/{query,support}/ for images
    - images/{train,val}/labels/ for label files
    
    Triplet format: <query_path> <support_path1> <support_path2> <support_path3> [<class> <x> <y> <w> <h> ...]
    
    Process:
    1. Copy all unique support images first (sequentially to avoid Windows file locks)
    2. Move query files in parallel and generate triplet lines
    3. Write triplet lines to label file
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    img_out_dir = base_out_dir / "images" / img_set
    query_out_dir = img_out_dir / "query"
    support_out_dir = img_out_dir / "support"
    labels_out_dir = img_out_dir / "labels"  # Labels go in images/{train,val}/labels/
    
    query_out_dir.mkdir(parents=True, exist_ok=True)
    support_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Step 1: Copy all unique support images SEQUENTIALLY (avoid Windows file lock issues) ---
    print(f"Copying support images for {img_set}...")
    all_unique_support_items = set()
    
    for (q_stage_path, s_stage_paths, bboxes, is_positive) in triplet_data_list:
        for s_stage_path in s_stage_paths:
            all_unique_support_items.add(s_stage_path)
    
    for s_stage_path in all_unique_support_items:
        s_final_path = support_out_dir / s_stage_path.name
        if not s_final_path.exists():
            shutil.copy2(str(s_stage_path), str(s_final_path))
    
    # --- Step 2: Move query files and generate triplet lines IN PARALLEL ---
    # Prepare arguments for each triplet
    # Now we only pass the support image filenames (not paths), since files are already copied
    args_list = [
        (q_stage_path, [s_stage_path.name for s_stage_path in s_stage_paths], bboxes, is_positive, query_out_dir, support_out_dir)
        for (q_stage_path, s_stage_paths, bboxes, is_positive) in triplet_data_list
    ]
    
    triplet_lines_for_txt = []
    
    with Pool(processes=num_workers) as pool:
        for line in tqdm(
            pool.imap_unordered(move_single_triplet, args_list),
            total=len(triplet_data_list),
            desc=f"Pass 2/2: Moving {img_set} files"
        ):
            triplet_lines_for_txt.append(line)
    
    # --- Step 3: Write label file to images/{img_set}/labels/ ---
    label_file_path = labels_out_dir / f"{img_set}.txt"
    with open(label_file_path, 'w') as f:
        for line in triplet_lines_for_txt:
            f.write(line + "\n")
    
    print(f"✅ Written {len(triplet_lines_for_txt)} samples to {label_file_path}")
    
    return triplet_lines_for_txt


def create_yaml(output_dir, train_txt_path, val_txt_path):
    """
    Creates the data.yaml file with proper paths for SiamYOLO.
    For SiamYOLO, train/val should point to the query image directories,
    and the labels directory will be found relative to them.
    """
    yaml_path = output_dir / "data.yaml"
    
    # SiamYOLO expects train/val to point to image directories, not label files
    # The get_labels() function will look for labels/ directory relative to img_path
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
    print("Starting dataset preparation...")
    print(f"Using {max(1, cpu_count() - 1)} CPU cores for parallel processing")
    
    # --- 1. Define Paths ---
    base_in_dir = Path(args.data_dir).resolve()
    base_out_dir = Path(args.output_dir).resolve()
    staging_dir = base_out_dir / "staging"
    
    samples_dir = base_in_dir / "samples"
    annot_file = base_in_dir / "annotations" / "annotations.json"

    # Clean up previous runs
    print("Cleaning up previous output directories...")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    if (base_out_dir / "images").exists():
        shutil.rmtree(base_out_dir / "images")
    if (base_out_dir / "labels").exists():
        shutil.rmtree(base_out_dir / "labels")
    
    # Create output base directory
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    out_labels_dir = base_out_dir / "labels"
    train_txt_path = out_labels_dir / "train.txt"
    val_txt_path = out_labels_dir / "val.txt"
    
    if not samples_dir.exists():
        print(f"❌ Error: Input directory {samples_dir} does not exist")
        return
    if not annot_file.exists():
        print(f"❌ Error: Annotation file {annot_file} does not exist")
        return

    # --- 2. Load Annotations ---
    print("Loading annotations...")
    try:
        with open(annot_file, 'r') as f:
            all_annotations = json.load(f)
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return
    
    annot_map = {}
    if isinstance(all_annotations, list):
        for item in all_annotations:
            annot_map[item['video_id']] = item
    elif isinstance(all_annotations, dict):
         annot_map[all_annotations['video_id']] = all_annotations
    else:
        print(f"❌ Error: Unknown JSON structure in {annot_file}")
        return
    
    print(f"✅ Loaded {len(annot_map)} video annotations")

    # --- 3. Process All Videos into Staging Area (PARALLEL) ---
    all_video_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    
    if not all_video_dirs:
        print(f"❌ Error: No video directories found in {samples_dir}")
        return
    
    print(f"Found {len(all_video_dirs)} video directories")
    
    all_triplet_data = process_all_videos(
        all_video_dirs, annot_map, staging_dir,
        args.frame_step, args.neg_ratio,
        num_workers=args.num_workers
    )
    
    print(f"Total processed samples: {len(all_triplet_data)}")
    if not all_triplet_data:
        print("❌ Error: No data was processed. Check input paths and video files.")
        return

    # --- 4. Split All Processed Frames (Frame-Level Split) ---
    print("Splitting all frames into train/val sets...")
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
    
    # Labels are now written inside move_files_and_write_triplets
    # No need to write them again here

    # --- 6. Clean up Staging Directory ---
    print("Cleaning up staging directory...")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    # --- 7. Create data.yaml ---
    print("Creating data.yaml file...")
    create_yaml(base_out_dir, train_txt_path, val_txt_path)
    
    # --- 8. Print Summary ---
    print("-" * 50)
    print("✅ Dataset preparation complete!")
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
        description="Prepare drone video data for SiamYOLO training."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the root of the source dataset (e.g., 'dataset/')."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the destination directory for processed data "
             "(e.g., 'processed_dataset/')."
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of frames to use for validation (default: 0.2)."
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Sample 1 positive frame every N frames (default: 5)."
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=0.2,
        help="Ratio of negative samples to total samples (default: 0.2)."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of CPU cores to use (default: auto-detect, use all except 1)."
    )
    
    args = parser.parse_args()
    main(args)