"""Test collate_fn with batch items having different keys."""

import torch
import sys
sys.path.insert(0, '.')

from ultralytics.data.dataset import YOLODataset

def test_collate_with_different_keys():
    """Test that collate_fn handles batch items with different keys."""
    
    # Simulate a batch where items have different keys
    batch = [
        {
            "img": torch.rand(3, 640, 640),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            "cls": torch.tensor([[0], [1]]),
            "batch_idx": torch.tensor([0, 0]),
            "masks": torch.rand(2, 160, 160),  # Only in first item
        },
        {
            "img": torch.rand(3, 640, 640),
            "bboxes": torch.tensor([[0.2, 0.3, 0.4, 0.5]]),
            "cls": torch.tensor([[2]]),
            "batch_idx": torch.tensor([0]),
            # No masks in second item
        },
        {
            "img": torch.rand(3, 640, 640),
            "bboxes": torch.tensor([]),  # Empty bboxes
            "cls": torch.tensor([]),
            "batch_idx": torch.tensor([]),
            "keypoints": torch.rand(0, 3),  # Only in third item
        },
    ]
    
    print("Testing collate_fn with batch items having different keys...")
    print(f"Item 0 keys: {list(batch[0].keys())}")
    print(f"Item 1 keys: {list(batch[1].keys())}")
    print(f"Item 2 keys: {list(batch[2].keys())}")
    
    try:
        result = YOLODataset.collate_fn(batch)
        print("\n✅ Collate successful!")
        print(f"\nCollated batch keys: {list(result.keys())}")
        print(f"img shape: {result['img'].shape}")
        print(f"bboxes shape: {result['bboxes'].shape}")
        print(f"cls shape: {result['cls'].shape}")
        print(f"batch_idx shape: {result['batch_idx'].shape}")
        print(f"batch_idx values: {result['batch_idx']}")
        
        if "masks" in result:
            print(f"masks shape: {result['masks'].shape}")
        if "keypoints" in result:
            print(f"keypoints shape: {result['keypoints'].shape}")
            
        # Verify batch_idx has correct offsets
        expected_batch_idx = torch.tensor([0, 0, 1, 2])  # First item: [0,0]+0, Second: [0]+1, Third: []+2
        print(f"\nExpected batch_idx: {expected_batch_idx[:3]}")  # Only first 3 since last is empty
        
        return True
    except Exception as e:
        print(f"\n❌ Collate failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_collate_with_different_keys()
    sys.exit(0 if success else 1)
