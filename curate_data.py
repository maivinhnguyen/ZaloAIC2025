import argparse
import json
import os
import random
import shutil
from pathlib import Path

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

def process_all_videos(video_dirs, annot_map, staging_dir, 
                       frame_step, neg_ratio):
    """
    Processes all videos and saves them to a single 'staging' directory.
    Returns a list of all triplet file paths.
    """
    
    query_stage_dir = staging_dir / "query"
    support_stage_dir = staging_dir / "support"
    label_stage_dir = staging_dir / "labels"
    
    query_stage_dir.mkdir(parents=True, exist_ok=True)
    support_stage_dir.mkdir(parents=True, exist_ok=True)
    label_stage_dir.mkdir(parents=True, exist_ok=True)
    
    all_triplet_data = []

    for video_dir in tqdm(video_dirs, desc="Pass 1/2: Processing all videos"):
        video_name = video_dir.name
        
        # --- 1. Stage Support Images ---
        support_dir_in = video_dir / "object_images"
        support_imgs_in = sorted(support_dir_in.glob('*.jpg'))
        
        if len(support_imgs_in) != 3:
            print(f"Warning: Expected 3 support images in {video_dir}, "
                  f"found {len(support_imgs_in)}. Skipping video.")
            continue
            
        staged_support_paths = []
        for s_img_in in support_imgs_in:
            s_out_name = f"{video_name}_{s_img_in.name}"
            s_out_abs_path = support_stage_dir / s_out_name
            shutil.copy2(s_img_in, s_out_abs_path)
            staged_support_paths.append(s_out_abs_path)

        # --- 2. Process Video and Annotations ---
        video_path = video_dir / "drone_video.mp4"
        if not video_path.exists():
            print(f"Warning: {video_path} not found. Skipping video.")
            continue
            
        cap = cv2.VideoCapture(str(video_path))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0 or frame_w == 0 or frame_h == 0:
            print(f"Warning: Could not read {video_path}. Skipping.")
            cap.release()
            continue
            
        video_annots = annot_map.get(video_name)
        if not video_annots:
            print(f"Warning: No annotations found for {video_name}. Skipping.")
            cap.release()
            continue

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

        # --- 5. Extract Frames, Write Labels, and Build Triplet List ---
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
            
            lbl_name = f"{video_name}_frame_{frame_idx:06d}.txt"
            lbl_out_abs_path = label_stage_dir / lbl_name
            
            cv2.imwrite(str(q_out_abs_path), frame)
            
            with open(lbl_out_abs_path, 'w') as f_lbl:
                if frame_idx in sampled_pos_indices:
                    for yolo_box in positive_frames[frame_idx]:
                        f_lbl.write(" ".join(map(str, yolo_box)) + "\n")
                else:
                    pass
            
            # Store a tuple of the absolute paths for this triplet
            all_triplet_data.append(
                (q_out_abs_path, staged_support_paths, lbl_out_abs_path)
            )
            
            frame_idx += 1

        cap.release()

    return all_triplet_data

def move_files_and_write_triplets(
    triplet_data_list, img_set, base_out_dir, copied_support_files
):
    """
    Moves files from staging to final dir, creates triplet file lines.
    `copied_support_files` is a set to prevent duplicate copies.
    """
    
    img_out_dir = base_out_dir / "images" / img_set
    lbl_out_dir = base_out_dir / "labels" / img_set
    
    query_out_dir = img_out_dir / "query"
    support_out_dir = img_out_dir / "support"
    
    query_out_dir.mkdir(parents=True, exist_ok=True)
    support_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    
    triplet_lines_for_txt = []
    
    for (q_stage_path, s_stage_paths, l_stage_path) in \
        tqdm(triplet_data_list, desc=f"Pass 2/2: Moving {img_set} files"):
        
        # 1. Move Query Image
        q_final_path = query_out_dir / q_stage_path.name
        shutil.move(str(q_stage_path), str(q_final_path))
        q_rel_path = q_final_path.relative_to(base_out_dir)

        # 2. Move Label File
        l_final_path = lbl_out_dir / l_stage_path.name
        shutil.move(str(l_stage_path), str(l_final_path))
        l_rel_path = l_final_path.relative_to(base_out_dir)
        
        # 3. Copy Support Images (if not already copied)
        s_rel_paths = []
        for s_stage_path in s_stage_paths:
            s_final_path = support_out_dir / s_stage_path.name
            if s_stage_path.name not in copied_support_files:
                shutil.copy2(str(s_stage_path), str(s_final_path))
                copied_support_files.add(s_stage_path.name)
            s_rel_paths.append(str(s_final_path.relative_to(base_out_dir)))

        # Format: query_path support1_path support2_path support3_path label_path
        line = (
            f"{q_rel_path} "
            f"{s_rel_paths[0]} {s_rel_paths[1]} {s_rel_paths[2]} "
            f"{l_rel_path}"
        )
        triplet_lines_for_txt.append(line)
        
    return triplet_lines_for_txt, copied_support_files


def create_yaml(output_dir, train_txt_path, val_txt_path):
    """Creates the data.yaml file."""
    yaml_path = output_dir / "data.yaml"
    train_rel = train_txt_path.relative_to(output_dir)
    val_rel = val_txt_path.relative_to(output_dir)
    
    yaml_content = f"""
path: {output_dir.resolve()}
train: {train_rel}
val: {val_rel}

nc: 1
names: ['target']
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

def main(args):
    print("Starting dataset preparation...")
    
    # --- 1. Define Paths ---
    base_in_dir = Path(args.data_dir)
    base_out_dir = Path(args.output_dir)
    staging_dir = base_out_dir / "staging"
    
    samples_dir = base_in_dir / "samples"
    annot_file = base_in_dir / "annotations" / "annotations.json"

    # Clean up previous runs
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    # Clean output dirs too for a fresh run
    if (base_out_dir / "images").exists():
        shutil.rmtree(base_out_dir / "images")
    if (base_out_dir / "labels").exists():
        shutil.rmtree(base_out_dir / "labels")
        
    out_labels_dir = base_out_dir / "labels"
    train_txt_path = out_labels_dir / "train.txt"
    val_txt_path = out_labels_dir / "val.txt"
    
    if not samples_dir.exists() or not annot_file.exists():
        print(f"Error: Input directory {base_in_dir} is missing "
              f"'samples' or 'annotations/annotations.json'")
        return

    # --- 2. Load Annotations ---
    print("Loading annotations...")
    with open(annot_file, 'r') as f:
        all_annotations = json.load(f)
    
    annot_map = {}
    if isinstance(all_annotations, list):
        for item in all_annotations:
            annot_map[item['video_id']] = item
    elif isinstance(all_annotations, dict):
         annot_map[all_annotations['video_id']] = all_annotations
    else:
        print(f"Error: Unknown JSON structure in {annot_file}")
        return

    # --- 3. Process All Videos into Staging Area ---
    all_video_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    
    all_triplet_data = process_all_videos(
        all_video_dirs, annot_map, staging_dir,
        args.frame_step, args.neg_ratio
    )
    
    print(f"Total processed samples: {len(all_triplet_data)}")
    if not all_triplet_data:
        print("Error: No data was processed. Check input paths and video files.")
        return

    # --- 4. Split All Processed Frames (Frame-Level Split) ---
    print("Splitting all frames into train/val sets...")
    train_triplets, val_triplets = train_test_split(
        all_triplet_data,
        test_size=args.val_split,
        random_state=42
    )
    print(f"Total samples: {len(all_triplet_data)} | "
          f"Training: {len(train_triplets)} | "
          f"Validation: {len(val_triplets)}")

    # --- 5. Move Files to Final Dirs & Write Triplet Files ---
    copied_support = set() # To track copied support files
    
    train_txt_lines, copied_support = move_files_and_write_triplets(
        train_triplets, "train", base_out_dir, copied_support
    )
    
    val_txt_lines, _ = move_files_and_write_triplets(
        val_triplets, "val", base_out_dir, copied_support
    )
    
    print("Writing final triplet label files...")
    with open(train_txt_path, 'w') as f:
        f.write("\n".join(train_txt_lines))
        
    with open(val_txt_path, 'w') as f:
        f.write("\n".join(val_txt_lines))

    # --- 6. Clean up Staging Directory ---
    print("Cleaning up staging directory...")
    shutil.rmtree(staging_dir)

    # --- 7. Create data.yaml ---
    print("Creating data.yaml file...")
    create_yaml(base_out_dir, train_txt_path, val_txt_path)
    
    print("-" * 30)
    print(f"✅ Dataset preparation complete!")
    print(f"Output saved to: {base_out_dir.resolve()}")
    print(f"Config file: {base_out_dir.resolve() / 'data.yaml'}")
    print("-" * 30)

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
    
    args = parser.parse_args()
    main(args)