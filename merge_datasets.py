#!/usr/bin/env python3
"""
Script to merge two YOLO datasets (copypaste_dataset and synthetic_dataset) into one merged dataset.
This script handles:
- Merging query images and labels
- Merging support images
- Combining train.txt and val.txt files
- Creating a new data.yaml configuration
- Resizing dataset 2 images to width 1024
- Parallel processing with multiple workers
"""

import os
import shutil
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial


def load_yaml(yaml_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, yaml_path: str):
    """Save YAML configuration file."""
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def read_file_list(txt_path: str) -> List[str]:
    """Read list of files from a text file."""
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def write_file_list(txt_path: str, file_list: List[str]):
    """Write list of files to a text file."""
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w') as f:
        for item in file_list:
            f.write(f"{item}\n")


def get_unique_filename(dest_dir: str, filename: str, existing_files: Set[str]) -> str:
    """
    Generate a unique filename if there's a conflict.
    Adds a numeric suffix if the file already exists.
    """
    if filename not in existing_files and not os.path.exists(os.path.join(dest_dir, filename)):
        return filename
    
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = f"{name}_{counter}{ext}"
    
    while new_filename in existing_files or os.path.exists(os.path.join(dest_dir, new_filename)):
        counter += 1
        new_filename = f"{name}_{counter}{ext}"
    
    return new_filename


def resize_image_to_width(image_path: str, target_width: int = 1024) -> Image.Image:
    """
    Resize image to target width while maintaining aspect ratio.
    
    Args:
        image_path: Path to the image file
        target_width: Target width in pixels
    
    Returns:
        Resized PIL Image
    """
    img = Image.open(image_path)
    
    # Calculate new height maintaining aspect ratio
    width, height = img.size
    if width == target_width:
        return img
    
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    
    # Resize with high-quality resampling
    resized_img = img.resize((target_width, target_height), Image.LANCZOS)
    
    return resized_img


def process_single_file(args):
    """
    Process a single file (copy or resize+copy).
    This function is designed to be called by multiprocessing workers.
    
    Args:
        args: Tuple of (src_path, dest_path, resize_to_width)
    
    Returns:
        Tuple of (success: bool, filename: str, error: str or None)
    """
    src_path, dest_path, resize_to_width = args
    filename = os.path.basename(src_path)
    
    try:
        # If resizing is requested and it's an image file
        if resize_to_width and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            resized_img = resize_image_to_width(src_path, resize_to_width)
            resized_img.save(dest_path)
        else:
            shutil.copy2(src_path, dest_path)
        
        return (True, filename, None)
    except Exception as e:
        return (False, filename, str(e))


def copy_files(src_dir: str, dest_dir: str, existing_files: Set[str] = None,
               resize_to_width: int = None, num_workers: int = 4) -> Dict[str, str]:
    """
    Copy files from source to destination directory using parallel processing.
    
    Args:
        src_dir: Source directory
        dest_dir: Destination directory
        existing_files: Set of existing filenames to check for conflicts
        resize_to_width: If specified, resize images to this width (for dataset 2)
        num_workers: Number of parallel workers
    
    Returns:
        Dictionary mapping original filenames to new filenames (same if no conflicts)
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    if existing_files is None:
        existing_files = set()
    
    name_mapping = {}
    
    if not os.path.exists(src_dir):
        print(f"Warning: Source directory {src_dir} does not exist")
        return name_mapping
    
    # Collect all files to process
    files_to_process = []
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        
        if not os.path.isfile(src_path):
            continue
        
        # Handle filename conflicts
        new_filename = get_unique_filename(dest_dir, filename, existing_files)
        existing_files.add(new_filename)
        
        dest_path = os.path.join(dest_dir, new_filename)
        name_mapping[filename] = new_filename
        
        files_to_process.append((src_path, dest_path, resize_to_width))
    
    # Process files in parallel using threads (safer for PIL operations)
    if len(files_to_process) == 0:
        return name_mapping
    
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, args): args for args in files_to_process}
        
        for future in as_completed(futures):
            success, filename, error = future.result()
            if not success:
                failed_files.append((filename, error))
    
    # Report any failures
    if failed_files:
        print(f"Warning: Failed to process {len(failed_files)} files:")
        for filename, error in failed_files[:5]:  # Show first 5 errors
            print(f"  - {filename}: {error}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    
    return name_mapping


def copy_label_files(src_label_dir: str, dest_label_dir: str, 
                    query_name_mapping: Dict[str, str], 
                    existing_files: Set[str] = None,
                    num_workers: int = 4) -> Dict[str, str]:
    """
    Copy label files (.txt with bounding boxes) corresponding to query images.
    Uses the same copy mechanism as image files for consistency.
    
    Args:
        src_label_dir: Source label directory
        dest_label_dir: Destination label directory
        query_name_mapping: Mapping of query image filenames (old -> new)
        existing_files: Set of existing label filenames to check for conflicts
        num_workers: Number of parallel workers
    
    Returns:
        Dictionary mapping original label filenames to new label filenames
    """
    os.makedirs(dest_label_dir, exist_ok=True)
    
    if existing_files is None:
        existing_files = set()
    
    label_mapping = {}
    
    if not os.path.exists(src_label_dir):
        print(f"Warning: Source label directory {src_label_dir} does not exist")
        return label_mapping
    
    # Build list of label files to copy based on query images
    files_to_process = []
    for old_query_name, new_query_name in query_name_mapping.items():
        # Convert image filename to label filename
        old_label_name = os.path.splitext(old_query_name)[0] + '.txt'
        new_label_name = os.path.splitext(new_query_name)[0] + '.txt'
        
        src_label_path = os.path.join(src_label_dir, old_label_name)
        
        if not os.path.exists(src_label_path):
            # Some images might not have labels (e.g., background images)
            continue
        
        # Handle label filename conflicts
        new_label_name = get_unique_filename(dest_label_dir, new_label_name, existing_files)
        existing_files.add(new_label_name)
        
        dest_label_path = os.path.join(dest_label_dir, new_label_name)
        label_mapping[old_label_name] = new_label_name
        
        # Add to processing queue (no resizing for label files)
        files_to_process.append((src_label_path, dest_label_path, None))
    
    # Copy files in parallel
    if len(files_to_process) > 0:
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_file, args): args for args in files_to_process}
            
            for future in as_completed(futures):
                success, filename, error = future.result()
                if not success:
                    failed_files.append((filename, error))
        
        if failed_files:
            print(f"Warning: Failed to copy {len(failed_files)} label files")
    
    return label_mapping


def merge_datasets(dataset1_path: str, dataset2_path: str, output_path: str, num_workers: int = 4):
    """
    Merge two YOLO datasets into one.
    Dataset 2 images will be resized to width 1024.
    
    Args:
        dataset1_path: Path to first dataset (e.g., copypaste_dataset)
        dataset2_path: Path to second dataset (e.g., synthetic_dataset) - will be resized
        output_path: Path for the merged dataset
        num_workers: Number of parallel workers for processing
    """
    print(f"Merging datasets:")
    print(f"  Dataset 1: {dataset1_path} (original size)")
    print(f"  Dataset 2: {dataset2_path} (will be resized to width 1024)")
    print(f"  Output: {output_path}")
    print(f"  Workers: {num_workers}")
    print()
    
    # Load YAML configs
    yaml1 = load_yaml(os.path.join(dataset1_path, 'data.yaml'))
    yaml2 = load_yaml(os.path.join(dataset2_path, 'data.yaml'))
    
    # Verify that datasets are compatible
    if yaml1.get('nc') != yaml2.get('nc'):
        raise ValueError(f"Number of classes mismatch: {yaml1.get('nc')} vs {yaml2.get('nc')}")
    
    if yaml1.get('names') != yaml2.get('names'):
        raise ValueError(f"Class names mismatch: {yaml1.get('names')} vs {yaml2.get('names')}")
    
    # Track all merged files
    merged_train_files = []
    merged_val_files = []

    for split in ['train', 'val']:
        print(f"\nProcessing {split} split...")

        # Move query images from dataset1 to dataset2
        src_query = os.path.join(dataset1_path, 'images', split, 'query')
        dest_query = os.path.join(dataset2_path, 'images', split, 'query')
        os.makedirs(dest_query, exist_ok=True)
        if os.path.exists(src_query):
            for fname in os.listdir(src_query):
                src_file = os.path.join(src_query, fname)
                dest_file = os.path.join(dest_query, fname)
                if os.path.isfile(src_file):
                    if os.path.exists(dest_file):
                        name, ext = os.path.splitext(fname)
                        counter = 1
                        new_fname = f"{name}_{counter}{ext}"
                        while os.path.exists(os.path.join(dest_query, new_fname)):
                            counter += 1
                            new_fname = f"{name}_{counter}{ext}"
                        dest_file = os.path.join(dest_query, new_fname)
                    shutil.move(src_file, dest_file)
        print(f"    Moved query images from dataset1 to dataset2")

        # Move label files from dataset1 to dataset2
        src_labels = os.path.join(dataset1_path, 'images', split, 'labels')
        dest_labels = os.path.join(dataset2_path, 'images', split, 'labels')
        os.makedirs(dest_labels, exist_ok=True)
        if os.path.exists(src_labels):
            for fname in os.listdir(src_labels):
                src_file = os.path.join(src_labels, fname)
                dest_file = os.path.join(dest_labels, fname)
                if os.path.isfile(src_file):
                    if os.path.exists(dest_file):
                        name, ext = os.path.splitext(fname)
                        counter = 1
                        new_fname = f"{name}_{counter}{ext}"
                        while os.path.exists(os.path.join(dest_labels, new_fname)):
                            counter += 1
                            new_fname = f"{name}_{counter}{ext}"
                        dest_file = os.path.join(dest_labels, new_fname)
                    shutil.move(src_file, dest_file)
        print(f"    Moved label files from dataset1 to dataset2")

        # Move support images from dataset1 to dataset2
        src_support = os.path.join(dataset1_path, 'images', split, 'support')
        dest_support = os.path.join(dataset2_path, 'images', split, 'support')
        os.makedirs(dest_support, exist_ok=True)
        if os.path.exists(src_support):
            for fname in os.listdir(src_support):
                src_file = os.path.join(src_support, fname)
                dest_file = os.path.join(dest_support, fname)
                if os.path.isfile(src_file):
                    if os.path.exists(dest_file):
                        name, ext = os.path.splitext(fname)
                        counter = 1
                        new_fname = f"{name}_{counter}{ext}"
                        while os.path.exists(os.path.join(dest_support, new_fname)):
                            counter += 1
                            new_fname = f"{name}_{counter}{ext}"
                        dest_file = os.path.join(dest_support, new_fname)
                    shutil.move(src_file, dest_file)
        print(f"    Moved support images from dataset1 to dataset2")

        # Read train.txt/val.txt from both datasets
        txt_file = f"{split}.txt"
        txt1 = os.path.join(dataset1_path, 'images', split, 'labels', txt_file)
        txt2 = os.path.join(dataset2_path, 'images', split, 'labels', txt_file)
        entries1 = read_file_list(txt1)
        entries2 = read_file_list(txt2)
        print(f"    Read {len(entries1)} entries from dataset1 {txt_file}")
        print(f"    Read {len(entries2)} entries from dataset2 {txt_file}")

        # Merge entries and write to dataset2
        merged_entries = entries1 + entries2
        write_file_list(txt2, merged_entries)
        print(f"  Total entries in merged {txt_file}: {len(merged_entries)}")

        if split == 'train':
            merged_train_files = merged_entries
        else:
            merged_val_files = merged_entries
    
    # Update data.yaml in dataset2
    merged_yaml = {
        'path': dataset2_path,
        'train': 'images/train/query',
        'val': 'images/val/query',
        'nc': yaml1['nc'],
        'names': yaml1['names']
    }
    yaml_path = os.path.join(dataset2_path, 'data.yaml')
    save_yaml(merged_yaml, yaml_path)
    print(f"\nUpdated merged data.yaml at {yaml_path}")

    # Move background_cache from dataset1 to dataset2
    src_cache = os.path.join(dataset1_path, 'background_cache')
    dest_cache = os.path.join(dataset2_path, 'background_cache')
    os.makedirs(dest_cache, exist_ok=True)
    if os.path.exists(src_cache):
        for cache_file in os.listdir(src_cache):
            src_file = os.path.join(src_cache, cache_file)
            dest_file = os.path.join(dest_cache, cache_file)
            if os.path.isfile(src_file):
                if os.path.exists(dest_file):
                    name, ext = os.path.splitext(cache_file)
                    counter = 1
                    while os.path.exists(os.path.join(dest_cache, f"{name}_{counter}{ext}")):
                        counter += 1
                    dest_file = os.path.join(dest_cache, f"{name}_{counter}{ext}")
                shutil.move(src_file, dest_file)
        print(f"Moved background cache from dataset1 to dataset2")

    # Print summary
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    print(f"Total training samples: {len(merged_train_files)}")
    print(f"Total validation samples: {len(merged_val_files)}")
    print(f"Output dataset location: {dataset2_path}")
    print("="*60)


def main():
    """Main function to run the dataset merging."""
    parser = argparse.ArgumentParser(
        description='Merge two YOLO datasets (copypaste_dataset and synthetic_dataset) into one merged dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset1',
        type=str,
        default=None,
        help='Path to dataset 1 (copypaste_dataset). If not specified, uses default path.'
    )
    
    parser.add_argument(
        '--dataset2',
        type=str,
        default=None,
        help='Path to dataset 2 (synthetic_dataset). If not specified, uses default path.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path for merged output dataset. If not specified, uses default path.'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers for processing files.'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite if output directory already exists (no prompt).'
    )
    
    args = parser.parse_args()
    
    # Define paths
    base_path = "/mlcv2/WorkingSpace/Personal/nguyenmv/ZaloAI/Repo/maibel/ZaloAIC2025"
    
    dataset1_path = args.dataset1 or os.path.join(base_path, "copypaste_dataset")
    dataset2_path = args.dataset2 or os.path.join(base_path, "synthetic_dataset")
    output_path = args.output or os.path.join(base_path, "merged_dataset")
    
    # Check if datasets exist
    if not os.path.exists(dataset1_path):
        raise FileNotFoundError(f"Dataset 1 not found: {dataset1_path}")
    
    if not os.path.exists(dataset2_path):
        raise FileNotFoundError(f"Dataset 2 not found: {dataset2_path}")
    
    # Check if output directory already exists
    if os.path.exists(output_path):
        if args.force:
            print(f"Removing existing output directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            response = input(f"Output directory {output_path} already exists. Overwrite? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Merge cancelled.")
                return
            shutil.rmtree(output_path)
    
    # Merge datasets
    merge_datasets(
        dataset1_path=dataset1_path,
        dataset2_path=dataset2_path,
        output_path=output_path,
        num_workers=args.workers
    )
    
    print("\nDataset merge completed successfully!")


if __name__ == "__main__":
    main()
