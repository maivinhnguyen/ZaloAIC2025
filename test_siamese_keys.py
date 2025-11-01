#!/usr/bin/env python3
"""
Test script to verify SiamDataset collate_fn handles Siamese-specific keys correctly.
"""

import torch
import sys
from pathlib import Path

# Add the local ultralytics to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.data.dataset import SiamDataset

print("Testing SiamDataset.collate_fn() with Siamese-specific keys...\n")

def create_siamese_batch():
    """Create a realistic mock batch with Siamese keys."""
    batch = []
    
    # Entry 0
    batch.append({
        "img": torch.randn(3, 360, 640),
        "support_img": torch.randn(3, 360, 640),
        "support_file": "support_image_0.jpg",
        "bboxes": torch.zeros((2, 4)),
        "cls": torch.tensor([[0.0], [1.0]]),
        "batch_idx": torch.tensor([0, 0], dtype=torch.long),
    })
    
    # Entry 1
    batch.append({
        "img": torch.randn(3, 640, 640),
        "support_img": torch.randn(3, 640, 640),
        "support_file": "support_image_1.jpg",
        "bboxes": torch.zeros((1, 4)),
        "cls": torch.tensor([[0.0]]),
        "batch_idx": torch.tensor([1], dtype=torch.long),
    })
    
    return batch

print("=" * 70)
print("Test 1: collate_fn with Siamese keys (support_img, support_file)")
print("=" * 70)

try:
    batch = create_siamese_batch()
    
    print(f"Batch before collate_fn:")
    for i, item in enumerate(batch):
        print(f"  Entry {i}:")
        print(f"    img shape: {item['img'].shape}")
        print(f"    support_img shape: {item['support_img'].shape}")
        print(f"    support_file: {item['support_file']}")
        print(f"    bboxes shape: {item['bboxes'].shape}")
    
    # This should NOT raise IndexError anymore
    collated = SiamDataset.collate_fn(batch)
    
    print(f"\nCollated batch:")
    print(f"  img shape: {collated['img'].shape}")
    print(f"  query_img shape: {collated['query_img'].shape}")
    print(f"  support_img shape: {collated['support_img'].shape}")
    print(f"  support_file: {collated.get('support_file', 'N/A')}")
    print(f"  bboxes shape: {collated['bboxes'].shape}")
    print(f"  Keys in collated batch: {sorted(collated.keys())}")
    
    # Verify structure
    assert collated['img'].shape == torch.Size([2, 3, 640, 640]), f"Wrong img shape: {collated['img'].shape}"
    assert collated['support_img'].shape == torch.Size([2, 3, 640, 640]), f"Wrong support_img shape: {collated['support_img'].shape}"
    assert "support_file" in collated, "support_file missing from collated batch"
    assert len(collated['support_file']) == 2, f"Wrong number of support files: {len(collated['support_file'])}"
    assert "query_img" in collated, "query_img alias missing"
    
    print("\n✓ collate_fn successfully handled Siamese keys!")
    print("✓ No IndexError occurred!")
    print("✓ All images properly padded and stacked!")
    print("✓ Siamese-specific metadata preserved!")
    
except IndexError as e:
    print(f"\n✗ IndexError occurred (this should NOT happen): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("Test 2: collate_fn with different-sized batches")
print("=" * 70)

try:
    # Create a larger batch with more variety
    batch = []
    sizes = [(360, 640), (640, 640), (480, 640), (560, 640)]
    
    for idx, (h, w) in enumerate(sizes):
        batch.append({
            "img": torch.randn(3, h, w),
            "support_img": torch.randn(3, h, w),
            "support_file": f"support_{idx}.jpg",
            "bboxes": torch.zeros((max(1, idx), 4)),
            "cls": torch.tensor([[float(idx % 2)] for _ in range(max(1, idx))]),
            "batch_idx": torch.tensor([idx] * max(1, idx), dtype=torch.long),
        })
    
    print(f"Input batch sizes: {[(b['img'].shape[1], b['img'].shape[2]) for b in batch]}")
    
    collated = SiamDataset.collate_fn(batch)
    
    print(f"Output collated shapes:")
    print(f"  img: {collated['img'].shape}")
    print(f"  support_img: {collated['support_img'].shape}")
    print(f"  support_file count: {len(collated['support_file'])}")
    
    assert collated['img'].shape == torch.Size([4, 3, 640, 640])
    assert collated['support_img'].shape == torch.Size([4, 3, 640, 640])
    
    print("\n✓ Large batch with various sizes handled correctly!")
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("=" * 70)
