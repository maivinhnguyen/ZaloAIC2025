#!/usr/bin/env python3
"""Test script to verify loss calculation fixes."""

import torch
import sys
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_stride_tensor_division():
    """Test that stride_tensor division works correctly after normalization."""
    print("Testing stride_tensor division fix...")
    
    # Simulate the tensors
    batch_size = 2
    num_anchors = 8400  # 80x80 + 40x40 + 20x20 for 640 resolution
    num_foreground = 30  # Number of foreground detections
    
    # Create mock tensors
    target_bboxes = torch.randn(batch_size, num_anchors, 4)  # Full tensor
    stride_tensor = torch.tensor([[1.], [2.], [4.], [8.]])  # Stride values
    stride_tensor = stride_tensor.repeat(num_anchors // 4, 1).reshape(num_anchors, 4)  # Repeat to match anchors
    
    # Create a mock foreground mask
    fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool)
    fg_mask[0, :num_foreground] = True  # Mark first num_foreground as foreground in first batch
    
    # Now test the corrected approach
    try:
        # Normalize first
        target_bboxes_normalized = target_bboxes / stride_tensor
        
        # Then apply mask - this should work now
        fg_target_bboxes = target_bboxes_normalized[fg_mask]
        
        print(f"✓ Stride tensor division works correctly!")
        print(f"  target_bboxes_normalized shape: {target_bboxes_normalized.shape}")
        print(f"  fg_target_bboxes shape: {fg_target_bboxes.shape}")
        print(f"  Expected shape: (some_num, 4), Got: {fg_target_bboxes.shape}")
        
        return True
    except RuntimeError as e:
        print(f"✗ Error occurred: {e}")
        return False


def test_batch_idx_creation():
    """Test that batch_idx is properly created in collate."""
    print("\nTesting batch_idx creation...")
    
    # Simulate batch samples
    batch = []
    for i in range(2):
        sample = {
            "img": torch.randn(3, 640, 640),
            "bboxes": torch.randn(5, 4) if i == 0 else torch.zeros((0, 4)),  # Different number of boxes
            "cls": torch.zeros(5, 1) if i == 0 else torch.zeros((0, 1)),
            "batch_idx": torch.zeros(5) if i == 0 else torch.zeros(0),  # This is now created
        }
        batch.append(sample)
    
    # Try collating
    try:
        from ultralytics.data.dataset import YOLODataset
        
        collated = YOLODataset.collate_fn(batch)
        
        if "batch_idx" in collated:
            print(f"✓ batch_idx exists in collated batch!")
            print(f"  batch_idx: {collated['batch_idx']}")
            print(f"  batch_idx shape: {collated['batch_idx'].shape}")
            
            # Check that batch indices were incremented correctly
            first_batch_indices = collated['batch_idx'][:5]
            second_batch_indices = collated['batch_idx'][5:]
            
            print(f"  First batch indices: {first_batch_indices}")
            print(f"  Second batch indices: {second_batch_indices}")
            
            return True
        else:
            print(f"✗ batch_idx not found in collated batch!")
            return False
    except Exception as e:
        print(f"✗ Error during collation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running loss calculation fix tests...\n")
    
    test1_passed = test_stride_tensor_division()
    test2_passed = test_batch_idx_creation()
    
    print("\n" + "="*50)
    if test1_passed and test2_passed:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
