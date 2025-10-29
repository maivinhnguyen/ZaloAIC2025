#!/usr/bin/env python3
"""
Validates the generated SiamYOLO dataset structure and formats.
"""

import argparse
from pathlib import Path
from collections import defaultdict


def validate_dataset(output_dir):
    """Validate the dataset structure and content."""
    output_dir = Path(output_dir).resolve()
    
    print(f"Validating dataset at: {output_dir}")
    print("-" * 60)
    
    errors = []
    warnings = []
    
    # --- Check directory structure ---
    required_dirs = [
        output_dir / "images" / "train" / "query",
        output_dir / "images" / "train" / "support",
        output_dir / "images" / "val" / "query",
        output_dir / "images" / "val" / "support",
        output_dir / "labels",
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            errors.append(f"❌ Missing directory: {dir_path}")
        else:
            print(f"✅ Found: {dir_path}")
    
    # --- Check label files ---
    train_txt = output_dir / "labels" / "train.txt"
    val_txt = output_dir / "labels" / "val.txt"
    data_yaml = output_dir / "data.yaml"
    
    for file_path in [train_txt, val_txt, data_yaml]:
        if not file_path.exists():
            errors.append(f"❌ Missing file: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    # --- Validate data.yaml ---
    if data_yaml.exists():
        print("\ndata.yaml content:")
        with open(data_yaml, 'r') as f:
            yaml_content = f.read()
            print(yaml_content)
        
        # Check required fields
        required_yaml_fields = ['path:', 'train:', 'val:', 'nc:', 'names:']
        for field in required_yaml_fields:
            if field not in yaml_content:
                warnings.append(f"⚠️  Missing YAML field: {field}")
    
    # --- Validate train.txt format ---
    if train_txt.exists():
        print(f"\nValidating {train_txt}...")
        with open(train_txt, 'r') as f:
            lines = f.readlines()
        
        print(f"Total training samples: {len(lines)}")
        
        if len(lines) > 0:
            # Check first 3 lines
            print("\nFirst 3 training samples:")
            for i, line in enumerate(lines[:3]):
                line = line.strip()
                parts = line.split()
                print(f"  Sample {i+1}: {len(parts)} fields")
                print(f"    {line[:100]}{'...' if len(line) > 100 else ''}")
                
                # Validate format: should have at least query + 3 supports = 4 paths
                if len(parts) < 4:
                    errors.append(f"Line {i+1}: Invalid format (expected >=4 paths, got {len(parts)})")
    
    # --- Validate val.txt format ---
    if val_txt.exists():
        print(f"\nValidating {val_txt}...")
        with open(val_txt, 'r') as f:
            lines = f.readlines()
        
        print(f"Total validation samples: {len(lines)}")
        
        if len(lines) > 0:
            # Check first 3 lines
            print("\nFirst 3 validation samples:")
            for i, line in enumerate(lines[:3]):
                line = line.strip()
                parts = line.split()
                print(f"  Sample {i+1}: {len(parts)} fields")
                print(f"    {line[:100]}{'...' if len(line) > 100 else ''}")
                
                if len(parts) < 4:
                    errors.append(f"Line {i+1}: Invalid format (expected >=4 paths, got {len(parts)})")
    
    # --- Count images ---
    train_query_count = len(list((output_dir / "images" / "train" / "query").glob("*.jpg")))
    train_support_count = len(list((output_dir / "images" / "train" / "support").glob("*.jpg")))
    val_query_count = len(list((output_dir / "images" / "val" / "query").glob("*.jpg")))
    val_support_count = len(list((output_dir / "images" / "val" / "support").glob("*.jpg")))
    
    print(f"\n--- Image counts ---")
    print(f"Train queries: {train_query_count}")
    print(f"Train support: {train_support_count}")
    print(f"Val queries: {val_query_count}")
    print(f"Val support: {val_support_count}")
    
    # --- Print results ---
    print("\n" + "=" * 60)
    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(error)
    else:
        print("✅ No critical errors found!")
    
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(warning)
    
    print("=" * 60)
    
    return len(errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SiamYOLO dataset structure.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output dataset directory to validate."
    )
    
    args = parser.parse_args()
    success = validate_dataset(args.output_dir)
    exit(0 if success else 1)
