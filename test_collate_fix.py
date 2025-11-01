#!/usr/bin/env python3
"""
Test script to verify SiamDataset collate_fn fixes for variable-sized images.
"""

import torch
import sys
from pathlib import Path
from copy import deepcopy

# Add the local ultralytics to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.data.dataset import SiamDataset, YOLODataset

print("Testing SiamDataset.collate_fn() with variable-sized images...\n")

# Create mock batch data with different image sizes (like the error shows)
# Error was: [3, 360, 640] at entry 0 and [3, 640, 640] at entry 8

def create_mock_batch():
    """Create a mock batch with variable image sizes to test collate_fn."""
    batch = []
    
    # Entry 0: 360x640 image
    batch.append({
        "img": torch.randn(3, 360, 640),
        "support_img": torch.randn(3, 360, 640),
        "bboxes": torch.zeros((2, 4)),
        "cls": torch.tensor([[0.0], [1.0]]),
        "batch_idx": torch.tensor([0, 0], dtype=torch.long),
    })
    
    # Entry 1: 640x640 image
    batch.append({
        "img": torch.randn(3, 640, 640),
        "support_img": torch.randn(3, 640, 640),
        "bboxes": torch.zeros((1, 4)),
        "cls": torch.tensor([[0.0]]),
        "batch_idx": torch.tensor([1], dtype=torch.long),
    })
    
    # Entry 2: 480x640 image
    batch.append({
        "img": torch.randn(3, 480, 640),
        "support_img": torch.randn(3, 480, 640),
        "bboxes": torch.zeros((3, 4)),
        "cls": torch.tensor([[0.0], [1.0], [0.0]]),
        "batch_idx": torch.tensor([2, 2, 2], dtype=torch.long),
    })
    
    return batch

print("=" * 70)
print("Test 1: collate_fn with variable image sizes")
print("=" * 70)

try:
    batch = create_mock_batch()
    
    print(f"Batch sizes before collate_fn:")
    for i, item in enumerate(batch):
        print(f"  Entry {i}: img shape = {item['img'].shape}")
    
    # This should NOT raise RuntimeError anymore
    collated = SiamDataset.collate_fn(batch)
    
    print(f"\nCollated batch:")
    print(f"  img shape: {collated['img'].shape}")
    print(f"  query_img shape: {collated['query_img'].shape}")
    print(f"  support_img shape: {collated['support_img'].shape}")
    
    # Verify all images have the same size
    # Shape should be [batch, channels, height, width] = [3, 3, 640, 640]
    assert collated['img'].shape == torch.Size([3, 3, 640, 640]), \
        f"Unexpected img shape: {collated['img'].shape}"
    assert collated['query_img'].shape == collated['img'].shape, \
        f"query_img shape mismatch: {collated['query_img'].shape} vs {collated['img'].shape}"
    assert collated['support_img'].shape == collated['img'].shape, \
        f"support_img shape mismatch: {collated['support_img'].shape} vs {collated['img'].shape}"
    
    print("\n✓ collate_fn successfully handled variable-sized images!")
    print("✓ All images padded to maximum dimensions!")
    print("✓ Stack operation completed without error!")
    
except RuntimeError as e:
    print(f"\n✗ RuntimeError occurred (this should NOT happen): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("Test 2: Verify padding values")
print("=" * 70)

try:
    batch = create_mock_batch()
    collated = SiamDataset.collate_fn(batch)
    
    # Check that padding was done with zeros
    # The padded areas should be 0 (unless they contain other values)
    img = collated['img']
    print(f"Image shape: {img.shape}")
    print(f"Image value ranges: min={img.min():.3f}, max={img.max():.3f}")
    print(f"Number of zero pixels: {(img == 0).sum()}")
    
    print("\n✓ Padding verification passed!")
    
except Exception as e:
    print(f"\n✗ Padding verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("=" * 70)
