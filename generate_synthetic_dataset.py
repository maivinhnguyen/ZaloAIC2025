"""
Synthetic Dataset Generator for SiamYOLO

This script generates a synthetic dataset by:
1. Extracting background frames from non-annotated video frames
2. Pasting augmented support objects (1-8 per image) with realistic transformations
3. Generating triplet format labels compatible with SiamYOLO

Support images starting with "dis_" are distractor objects (pasted but not labeled).
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Set
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def extract_background_frames(
    samples_dir: Path, 
    annotations_file: Path, 
    output_dir: Path,
    frames_per_video: int = 50,
    frame_step: int = 10
) -> List[Path]:
    """
    Extract non-annotated frames from videos as background images.
    
    Args:
        samples_dir: Directory containing video samples
        annotations_file: JSON file with annotations
        output_dir: Directory to save background frames
        frames_per_video: Maximum frames to extract per video
        frame_step: Step between extracted frames
    
    Returns:
        List of paths to extracted background frames
    """
    with open(annotations_file, 'r') as f:
        all_annotations = json.load(f)
    
    # Build annotation map
    annot_map = {}
    if isinstance(all_annotations, list):
        for video_data in all_annotations:
            annot_map[video_data["video_id"]] = video_data
    elif isinstance(all_annotations, dict):
        annot_map = all_annotations
    
    output_dir.mkdir(parents=True, exist_ok=True)
    background_frames = []
    
    video_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    
    for video_dir in tqdm(video_dirs, desc="Extracting backgrounds"):
        video_name = video_dir.name
        video_path = video_dir / "drone_video.mp4"
        
        if not video_path.exists():
            continue
        
        # Get annotated frames for this video
        annotated_frames = set()
        video_annots = annot_map.get(video_name, {})
        
        if video_annots and "annotations" in video_annots:
            for annot_group in video_annots["annotations"]:
                for bbox_data in annot_group.get("bboxes", []):
                    annotated_frames.add(bbox_data["frame"])
        
        # Extract non-annotated frames
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get candidate non-annotated frames
        candidate_frames = [
            i for i in range(0, total_frames, frame_step)
            if i not in annotated_frames
        ]
        
        # Randomly sample frames
        sampled_frames = random.sample(
            candidate_frames, 
            min(frames_per_video, len(candidate_frames))
        )
        
        for frame_idx in sampled_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                bg_filename = f"{video_name}_bg_{frame_idx:06d}.jpg"
                bg_path = output_dir / bg_filename
                cv2.imwrite(str(bg_path), frame)
                background_frames.append(bg_path)
        
        cap.release()
    
    return background_frames


def load_object_with_alpha(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an object image with alpha channel.
    
    Returns:
        Tuple of (BGR image, alpha mask)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Handle different image formats
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Has alpha channel
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # RGB/BGR without alpha, assume white background
        bgr = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    elif len(img.shape) == 2:
        # Grayscale image
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, alpha = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError(f"Unsupported image format: {img.shape}")
    
    return bgr, alpha


def apply_augmentations(
    obj_img: np.ndarray, 
    obj_mask: np.ndarray,
    scale_range: Tuple[float, float] = (0.5, 3.0),
    brightness_range: Tuple[float, float] = (0.85, 1.15),
    rotation_range: Tuple[int, int] = (-30, 30),
    apply_perspective: bool = False,
    target_bg_size: Tuple[int, int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply realistic augmentations to object image.
    
    Args:
        obj_img: Object image
        obj_mask: Object mask
        scale_range: Range for scaling
        brightness_range: Range for brightness
        rotation_range: Range for rotation
        apply_perspective: Whether to apply perspective
        target_bg_size: (height, width) of target background - used to constrain object size
    
    Returns:
        Tuple of (augmented image, augmented mask)
    """
    h, w = obj_img.shape[:2]
    
    # Safety check: skip if too small
    if h < 1 or w < 1:
        return obj_img, obj_mask
    
    # 1. Scale with size constraints relative to background
    scale = random.uniform(*scale_range)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # If we know target background size, constrain object to 5-15% of it
    if target_bg_size is not None:
        bg_h, bg_w = target_bg_size
        # Target object to be 5-15% of background size
        target_size_min = int(min(bg_h, bg_w) * 0.01)
        target_size_max = int(min(bg_h, bg_w) * 0.1)
        current_size = min(new_h, new_w)
        
        if current_size < target_size_min:
            # Scale up to minimum
            scale_factor = target_size_min / current_size
            new_w = int(new_w * scale_factor)
            new_h = int(new_h * scale_factor)
        elif current_size > target_size_max:
            # Scale down to maximum
            scale_factor = target_size_max / current_size
            new_w = int(new_w * scale_factor)
            new_h = int(new_h * scale_factor)
    
    # Clamp to minimum size
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    obj_mask = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 2. Rotation
    angle = random.uniform(*rotation_range)
    center = (new_w // 2, new_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box after rotation
    cos, sin = np.abs(rot_mat[0, 0]), np.abs(rot_mat[0, 1])
    new_w_rot = int(new_h * sin + new_w * cos)
    new_h_rot = int(new_h * cos + new_w * sin)
    
    # Clamp rotated dimensions to minimum size
    new_w_rot = max(1, new_w_rot)
    new_h_rot = max(1, new_h_rot)
    
    # Adjust rotation matrix
    rot_mat[0, 2] += (new_w_rot - new_w) / 2
    rot_mat[1, 2] += (new_h_rot - new_h) / 2
    
    obj_img = cv2.warpAffine(obj_img, rot_mat, (new_w_rot, new_h_rot), 
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    obj_mask = cv2.warpAffine(obj_mask, rot_mat, (new_w_rot, new_h_rot),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
    
    # 3. Perspective transform (optional)
    if apply_perspective and random.random() < 0.3:
        try:
            h_p, w_p = obj_img.shape[:2]
            offset = int(min(h_p, w_p) * 0.1)
            
            src_pts = np.float32([[0, 0], [w_p, 0], [w_p, h_p], [0, h_p]])
            dst_pts = src_pts + np.random.uniform(-offset, offset, src_pts.shape).astype(np.float32)
            
            # Clamp destination points to avoid degenerate cases
            dst_pts[:, 0] = np.clip(dst_pts[:, 0], 0, w_p)
            dst_pts[:, 1] = np.clip(dst_pts[:, 1], 0, h_p)
            
            perspective_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            obj_img = cv2.warpPerspective(obj_img, perspective_mat, (w_p, h_p),
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            obj_mask = cv2.warpPerspective(obj_mask, perspective_mat, (w_p, h_p),
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        except Exception:
            # Skip perspective transform if it fails
            pass
    
    # 4. Brightness adjustment
    brightness = random.uniform(*brightness_range)
    obj_img = np.clip(obj_img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    
    return obj_img, obj_mask


def add_shadow(obj_img: np.ndarray, obj_mask: np.ndarray) -> np.ndarray:
    """
    Add realistic shadow to object image.
    
    Returns:
        Object image with shadow effect
    """
    if random.random() < 0.5:
        # Apply simple darkening
        shadow_intensity = random.uniform(0.6, 0.85)
        obj_img = np.clip(obj_img.astype(np.float32) * shadow_intensity, 0, 255).astype(np.uint8)
    
    return obj_img


def augment_background(background: np.ndarray) -> np.ndarray:
    """
    Apply light augmentations to background image.
    
    Returns:
        Augmented background image
    """
    # 1. Vertical flip (50% chance)
    if random.random() < 0.5:
        background = cv2.flip(background, 0)  # 0 = vertical flip
    
    # 2. Horizontal flip (30% chance)
    if random.random() < 0.3:
        background = cv2.flip(background, 1)  # 1 = horizontal flip
    
    # 3. Slight color jitter (brightness + saturation)
    if random.random() < 0.6:
        # Convert to HSV for better color control
        hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        
        # Brightness adjustment (very slight: ±5%)
        brightness_factor = random.uniform(0.95, 1.05)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        
        # Saturation adjustment (very slight: ±5%)
        saturation_factor = random.uniform(0.95, 1.05)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        background = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return background


def paste_object_alpha_blend(
    background: np.ndarray,
    obj_img: np.ndarray,
    obj_mask: np.ndarray,
    position: Tuple[int, int],
    gaussian_blur: bool = True
) -> np.ndarray:
    """
    Paste object onto background using alpha blending with optional Gaussian blur for smooth edges.
    Much faster than Poisson blending.
    """
    x, y = position
    h_obj, w_obj = obj_img.shape[:2]
    h_bg, w_bg = background.shape[:2]
    
    # Ensure object fits within background
    if x + w_obj > w_bg or y + h_obj > h_bg or x < 0 or y < 0:
        return background
    
    # Alpha blending
    roi = background[y:y+h_obj, x:x+w_obj]
    alpha = obj_mask.astype(np.float32) / 255.0
    
    # Optional: Apply Gaussian blur to alpha mask for smoother edges
    if gaussian_blur and alpha.max() > 0:
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    
    alpha_3ch = np.stack([alpha] * 3, axis=2)
    
    blended = (obj_img.astype(np.float32) * alpha_3ch + 
               roi.astype(np.float32) * (1 - alpha_3ch))
    
    background[y:y+h_obj, x:x+w_obj] = blended.astype(np.uint8)
    return background


def generate_synthetic_sample(
    background_path: Path,
    support_objects: Dict[str, Path],
    distractor_objects: List[Path],
    num_objects_range: Tuple[int, int] = (1, 8),
    max_attempts: int = 50
) -> Tuple[np.ndarray, List[Tuple[str, List[float]]]]:
    """
    Generate a single synthetic sample by pasting objects onto background.
    
    Args:
        background_path: Path to background image
        support_objects: Dict mapping object names to support image paths
        distractor_objects: List of distractor object paths
        num_objects_range: Range for number of objects to paste
        max_attempts: Maximum attempts to find non-overlapping positions
    
    Returns:
        Tuple of (synthetic image, list of (object_name, bbox_yolo))
    """
    # Load background
    background = cv2.imread(str(background_path))
    if background is None:
        raise ValueError(f"Failed to load background: {background_path}")
    
    h_bg, w_bg = background.shape[:2]
    
    # Apply light augmentations to background
    background = augment_background(background)
    
    # Decide number of objects to paste
    num_objects = random.randint(*num_objects_range)
    
    # Select objects to paste (ensure at least one labeled object)
    object_names = list(support_objects.keys())
    num_labeled = max(1, random.randint(1, min(num_objects, len(object_names))))
    num_distractors = min(num_objects - num_labeled, len(distractor_objects))
    
    selected_labeled = random.sample(object_names, num_labeled)
    selected_distractors = random.sample(distractor_objects, num_distractors) if num_distractors > 0 else []
    
    # Track pasted bounding boxes to avoid overlap
    pasted_boxes = []
    labels = []
    
    # Helper function to check overlap
    def check_overlap(new_box, existing_boxes, iou_threshold=0.3):
        x1, y1, x2, y2 = new_box
        for ex_box in existing_boxes:
            ex_x1, ex_y1, ex_x2, ex_y2 = ex_box
            
            # Calculate IoU
            inter_x1 = max(x1, ex_x1)
            inter_y1 = max(y1, ex_y1)
            inter_x2 = min(x2, ex_x2)
            inter_y2 = min(y2, ex_y2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                new_area = (x2 - x1) * (y2 - y1)
                ex_area = (ex_x2 - ex_x1) * (ex_y2 - ex_y1)
                iou = inter_area / (new_area + ex_area - inter_area)
                
                if iou > iou_threshold:
                    return True
        return False
    
    # Paste labeled objects
    for obj_name in selected_labeled:
        obj_path = support_objects[obj_name]
        placed = False
        
        # Load and augment object ONCE outside the loop
        try:
            obj_img, obj_mask = load_object_with_alpha(obj_path)
            obj_img, obj_mask = apply_augmentations(
                obj_img, obj_mask, 
                target_bg_size=(h_bg, w_bg)
            )
            obj_img = add_shadow(obj_img, obj_mask)
        except Exception as e:
            continue
        
        h_obj, w_obj = obj_img.shape[:2]
        
        # Skip if object is too large or too small (shouldn't happen with target_bg_size constraint)
        if h_obj < 5 or w_obj < 5:
            continue
        
        # Now try to find a good placement position
        for attempt in range(max_attempts):
            try:
                # Ensure valid position range
                x_max = max(1, w_bg - w_obj)
                y_max = max(1, h_bg - h_obj)
                
                # Random position
                x = random.randint(0, x_max)
                y = random.randint(0, y_max)
                
                # Check overlap
                new_box = (x, y, x + w_obj, y + h_obj)
                if check_overlap(new_box, pasted_boxes):
                    continue
                
                # Paste object
                background = paste_object_alpha_blend(background, obj_img, obj_mask, (x, y))
                
                # Record bbox in YOLO format
                x_center = (x + w_obj / 2) / w_bg
                y_center = (y + h_obj / 2) / h_bg
                width = w_obj / w_bg
                height = h_obj / h_bg
                
                # Clamp to [0, 1]
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                width = np.clip(width, 0, 1)
                height = np.clip(height, 0, 1)
                
                pasted_boxes.append(new_box)
                labels.append((obj_name, [0, x_center, y_center, width, height]))
                placed = True
                break
                
            except Exception as e:
                # Silently continue - some augmentations may fail
                pass
        
        # If we couldn't place this object after max attempts, skip it
        if not placed:
            pass
    
    # Paste distractor objects (no labels)
    for dist_path in selected_distractors:
        placed = False
        
        # Load and augment distractor ONCE outside the loop
        try:
            obj_img, obj_mask = load_object_with_alpha(dist_path)
            obj_img, obj_mask = apply_augmentations(
                obj_img, obj_mask,
                target_bg_size=(h_bg, w_bg)
            )
            obj_img = add_shadow(obj_img, obj_mask)
        except Exception as e:
            continue
        
        h_obj, w_obj = obj_img.shape[:2]
        
        # Skip if object is too large or too small (shouldn't happen with target_bg_size constraint)
        if h_obj < 5 or w_obj < 5:
            continue
        
        # Now try to find a good placement position
        for attempt in range(max_attempts):
            try:
                # Ensure valid position range
                x_max = max(1, w_bg - w_obj)
                y_max = max(1, h_bg - h_obj)
                
                # Random position
                x = random.randint(0, x_max)
                y = random.randint(0, y_max)
                
                # Check overlap
                new_box = (x, y, x + w_obj, y + h_obj)
                if check_overlap(new_box, pasted_boxes):
                    continue
                
                # Paste object (distractor)
                background = paste_object_alpha_blend(background, obj_img, obj_mask, (x, y))
                pasted_boxes.append(new_box)
                placed = True
                break
                
            except Exception as e:
                # Silently continue - some augmentations may fail
                pass
        
        # If we couldn't place this distractor after max attempts, skip it
        if not placed:
            pass  # Continue to next distractor
    
    return background, labels


def process_single_background(args):
    """
    Process a single background image (for parallel processing).
    """
    (bg_path, support_objects, distractor_objects, staging_dir, 
     samples_per_bg, num_objects_range) = args
    
    triplet_data = []
    
    for sample_idx in range(samples_per_bg):
        try:
            # Generate synthetic sample
            synthetic_img, labels = generate_synthetic_sample(
                bg_path, support_objects, distractor_objects, num_objects_range
            )
            
            
            # Skip if no labels generated
            if not labels:
                continue
            
            # Save query image
            query_stage_dir = staging_dir / "query"
            query_stage_dir.mkdir(parents=True, exist_ok=True)
            
            query_filename = f"{bg_path.stem}_syn_{sample_idx:03d}.jpg"
            query_path = query_stage_dir / query_filename
            cv2.imwrite(str(query_path), synthetic_img)
            
            # For each unique object in labels, create a triplet
            # Group labels by object name
            object_labels = {}
            for obj_name, bbox in labels:
                if obj_name not in object_labels:
                    object_labels[obj_name] = []
                object_labels[obj_name].append(bbox)
            
            # Create one triplet per unique object
            for obj_name, bboxes in object_labels.items():
                support_path = support_objects[obj_name]
                
                # Create triplet: (query_path, support_path, bboxes)
                triplet_data.append((query_path, support_path, bboxes))
        
        except Exception as e:
            # Print error for debugging but continue
            import traceback
            print(f"  ⚠️  Error processing {bg_path.name} sample {sample_idx}: {str(e)[:100]}", flush=True)
            # Uncomment to see full traceback:
            # traceback.print_exc()
            continue

    return triplet_data


def load_support_objects(support_dir: Path) -> Tuple[Dict[str, Path], List[Path]]:
    """
    Load support and distractor objects from directory.
    
    Returns:
        Tuple of (support_objects dict, distractor_objects list)
    """
    support_objects = {}
    distractor_objects = []
    
    for img_path in sorted(support_dir.glob("*.png")):
        filename = img_path.stem
        
        if filename.startswith("dis_"):
            # Distractor object
            distractor_objects.append(img_path)
        else:
            # Support object (extract object name)
            # e.g., "bike_1" -> "bike", "toolbox_1" -> "toolbox_1"
            # Keep the full name to distinguish toolbox_1 from toolbox_2
            support_objects[filename] = img_path
    
    print(f"✅ Loaded {len(support_objects)} support objects: {list(support_objects.keys())}")
    print(f"✅ Loaded {len(distractor_objects)} distractor objects")
    
    return support_objects, distractor_objects


def copy_support_images(support_objects: Dict[str, Path], staging_dir: Path) -> Dict[str, List[Path]]:
    """
    Copy support images to staging directory (duplicate 3 times for SiamYOLO format).
    
    Returns:
        Dict mapping object names to list of 3 support paths
    """
    support_stage_dir = staging_dir / "support"
    support_stage_dir.mkdir(parents=True, exist_ok=True)
    
    support_map = {}
    
    for obj_name, obj_path in support_objects.items():
        staged_paths = []
        
        for i in range(3):
            support_filename = f"{obj_name}_support_{i+1}{obj_path.suffix}"
            support_out_path = support_stage_dir / support_filename
            shutil.copy2(obj_path, support_out_path)
            staged_paths.append(support_out_path)
        
        support_map[obj_name] = staged_paths
    
    return support_map


def move_single_triplet(args):
    """
    Move a single triplet's files from staging to final location.
    """
    q_stage_path, s_stage_names, bboxes, query_out_dir, support_out_dir, img_set_root = args
    
    # 1. Copy query image (multiple triplets may reference same query)
    q_final_path = query_out_dir / q_stage_path.name
    if not q_final_path.exists():
        shutil.copy2(str(q_stage_path), str(q_final_path))
    
    # Compute paths relative to img_set_root
    import os
    q_rel_path = os.path.relpath(q_final_path, img_set_root)
    
    # 2. Get support image relative paths
    s_rel_paths = []
    for s_name in s_stage_names:
        s_final_path = support_out_dir / s_name
        s_rel_path = os.path.relpath(s_final_path, img_set_root)
        s_rel_paths.append(str(s_rel_path))
    
    # 3. Format triplet line
    line_parts = [str(q_rel_path)]
    line_parts.extend(s_rel_paths)
    
    # Add bboxes (already in YOLO format)
    for bbox in bboxes:
        line_parts.extend([str(x) for x in bbox])
    
    line = " ".join(line_parts)
    return line


def move_files_and_write_triplets(
    triplet_data_list: List[Tuple],
    support_map: Dict[str, List[Path]],
    img_set: str,
    base_out_dir: Path,
    num_workers: int = None
):
    """
    Move files from staging to final directory and write triplet labels.
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
    
    # --- Step 1: Copy all unique support images ---
    print(f"Copying support images for {img_set}...")
    all_unique_support = set()
    
    for (q_path, support_path, bboxes) in triplet_data_list:
        # Get staged support paths for this object
        obj_name = support_path.stem
        if obj_name in support_map:
            all_unique_support.update(support_map[obj_name])
    
    for s_stage_path in all_unique_support:
        s_final_path = support_out_dir / s_stage_path.name
        if not s_final_path.exists():
            shutil.copy2(str(s_stage_path), str(s_final_path))
    
    # --- Step 2: Move query files and generate triplet lines ---
    img_set_root = base_out_dir / "images" / img_set
    args_list = []
    
    for (q_path, support_path, bboxes) in triplet_data_list:
        obj_name = support_path.stem
        if obj_name in support_map:
            s_stage_names = [p.name for p in support_map[obj_name]]
            args_list.append((q_path, s_stage_names, bboxes, query_out_dir, 
                            support_out_dir, img_set_root))
    
    triplet_lines = []
    
    with Pool(processes=num_workers) as pool:
        for line in tqdm(
            pool.imap_unordered(move_single_triplet, args_list),
            total=len(args_list),
            desc=f"Moving {img_set} files"
        ):
            triplet_lines.append(line)
    
    # --- Step 3: Write label file ---
    label_file_path = labels_out_dir / f"{img_set}.txt"
    with open(label_file_path, 'w') as f:
        for line in triplet_lines:
            f.write(line + "\n")
    
    print(f"✅ Written {len(triplet_lines)} samples to {label_file_path}")
    
    return triplet_lines


def create_yaml(output_dir: Path):
    """
    Create data.yaml file for SiamYOLO training.
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
    print("=" * 60)
    print("Synthetic Dataset Generator for SiamYOLO")
    print("=" * 60)
    print(f"Using {args.num_workers} CPU cores for parallel processing")
    
    # --- 1. Define Paths ---
    support_dir = Path(args.support_dir).resolve()
    train_data_dir = Path(args.train_data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    staging_dir = output_dir / "staging"
    background_cache_dir = output_dir / "background_cache"
    
    samples_dir = train_data_dir / "samples"
    annotations_file = train_data_dir / "annotations" / "annotations.json"
    
    # Clean up previous runs
    print("\nCleaning up previous output directories...")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    if (output_dir / "images").exists():
        shutil.rmtree(output_dir / "images")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 2. Load Support Objects ---
    print("\nLoading support objects...")
    support_objects, distractor_objects = load_support_objects(support_dir)
    
    if not support_objects:
        print("❌ Error: No support objects found!")
        return
    
    # --- 3. Extract Background Frames (with caching) ---
    if not background_cache_dir.exists() or not list(background_cache_dir.glob("*.jpg")):
        print("\nExtracting background frames from videos...")
        background_frames = extract_background_frames(
            samples_dir, annotations_file, background_cache_dir,
            frames_per_video=args.bg_frames_per_video,
            frame_step=args.bg_frame_step
        )
    else:
        print("\nUsing cached background frames...")
        background_frames = sorted(background_cache_dir.glob("*.jpg"))
        print(f"✅ Found {len(background_frames)} cached background frames")
    
    if not background_frames:
        print("❌ Error: No background frames available!")
        return
    
    # --- 4. Copy Support Images to Staging ---
    print("\nStaging support images...")
    support_map = copy_support_images(support_objects, staging_dir)
    
    # --- 5. Generate Synthetic Samples ---
    print(f"\nGenerating {args.samples_per_bg} synthetic samples per background...")
    print(f"Total backgrounds to process: {len(background_frames)}")
    
    # Prepare arguments for parallel processing
    args_list = [
        (bg_path, support_objects, distractor_objects, staging_dir,
         args.samples_per_bg, tuple(args.num_objects_range))
        for bg_path in background_frames
    ]
    
    all_triplet_data = []
    
    if args.num_workers == 1:
        # Sequential processing for debugging
        print("Running in sequential mode (num_workers=1)")
        for arg in tqdm(args_list, desc="Generating synthetic samples"):
            result = process_single_background(arg)
            all_triplet_data.extend(result)
    else:
        # Parallel processing
        print(f"Running in parallel mode with {args.num_workers} workers")
        with Pool(processes=args.num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_background, args_list),
                total=len(background_frames),
                desc="Generating synthetic samples"
            ):
                all_triplet_data.extend(result)
    
    print(f"\n✅ Generated {len(all_triplet_data)} total triplets")
    
    if not all_triplet_data:
        print("❌ Error: No synthetic samples generated!")
        return
    
    # --- 6. Split into Train/Val ---
    print("\nSplitting into train/val sets...")
    train_triplets, val_triplets = train_test_split(
        all_triplet_data,
        test_size=args.val_split,
        random_state=42
    )
    print(f"✅ Training: {len(train_triplets)} | Validation: {len(val_triplets)}")
    
    # --- 7. Move Files and Write Triplet Labels ---
    print("\nFinalizing dataset structure...")
    
    move_files_and_write_triplets(
        train_triplets, support_map, "train", output_dir, args.num_workers
    )
    
    move_files_and_write_triplets(
        val_triplets, support_map, "val", output_dir, args.num_workers
    )
    
    # --- 8. Clean up Staging ---
    print("\nCleaning up staging directory...")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    
    # --- 9. Create data.yaml ---
    print("\nCreating data.yaml...")
    create_yaml(output_dir)
    
    # --- 10. Summary ---
    print("\n" + "=" * 60)
    print("✅ Synthetic dataset generation complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Config file: {output_dir / 'data.yaml'}")
    print(f"Structure:")
    print(f"  - images/train/{{query,support,labels}}/")
    print(f"  - images/val/{{query,support,labels}}/")
    print(f"  - background_cache/ (reusable)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for SiamYOLO training"
    )
    
    parser.add_argument(
        "--support-dir",
        type=str,
        default="copy_paste/support",
        help="Directory containing support object images (default: copy_paste/support)"
    )
    
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default="/mlcv2/Datasets/ZaloAI2025/track1/train",
        help="Directory containing original training data with annotations"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for synthetic dataset"
    )
    
    parser.add_argument(
        "--bg-frames-per-video",
        type=int,
        default=50,
        help="Maximum background frames to extract per video (default: 50)"
    )
    
    parser.add_argument(
        "--bg-frame-step",
        type=int,
        default=10,
        help="Frame step for background extraction (default: 10)"
    )
    
    parser.add_argument(
        "--samples-per-bg",
        type=int,
        default=3,
        help="Number of synthetic samples to generate per background (default: 3)"
    )
    
    parser.add_argument(
        "--num-objects-range",
        type=int,
        nargs=2,
        default=[1, 8],
        help="Range for number of objects per sample (default: 1 8)"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Validation split ratio (default: 0.2)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, use 1 for debugging)"
    )
    
    args = parser.parse_args()
    main(args)
