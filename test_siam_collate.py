"""Test SiamDataset collate_fn with query and support images."""

import torch
import sys
sys.path.insert(0, '.')

from ultralytics.data.dataset import SiamDataset

def test_siam_collate():
    """Test SiamDataset collate_fn with realistic batch data."""
    
    # Simulate a batch from SiamYOLODataset.__getitem__
    batch = [
        {
            # Query image with 2 objects
            "img": torch.rand(3, 640, 640),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            "cls": torch.tensor([[0.], [0.]]),
            "batch_idx": torch.tensor([0, 0]),
            "support_img": torch.rand(3, 640, 640),
            "support_file": "support1.jpg",
            "im_file": "query1.jpg",
            "ori_shape": (480, 640),
            "resized_shape": (640, 640),
        },
        {
            # Query image with 1 object
            "img": torch.rand(3, 640, 640),
            "bboxes": torch.tensor([[0.2, 0.3, 0.4, 0.5]]),
            "cls": torch.tensor([[1.]]),
            "batch_idx": torch.tensor([0]),
            "support_img": torch.rand(3, 640, 640),
            "support_file": "support2.jpg",
            "im_file": "query2.jpg",
            "ori_shape": (720, 1280),
            "resized_shape": (640, 640),
        },
        {
            # Query image with no objects (edge case)
            "img": torch.rand(3, 640, 640),
            "bboxes": torch.zeros((0, 4)),
            "cls": torch.zeros((0, 1)),
            "batch_idx": torch.zeros(0, dtype=torch.long),
            "support_img": torch.rand(3, 640, 640),
            "support_file": "support3.jpg",
            "im_file": "query3.jpg",
            "ori_shape": (640, 640),
            "resized_shape": (640, 640),
        },
    ]
    
    print("Testing SiamDataset.collate_fn...")
    print(f"Batch size: {len(batch)}")
    print(f"Item 0 keys: {list(batch[0].keys())}")
    print(f"Item 0 bboxes: {batch[0]['bboxes'].shape}")
    print(f"Item 1 bboxes: {batch[1]['bboxes'].shape}")
    print(f"Item 2 bboxes: {batch[2]['bboxes'].shape}")
    
    try:
        result = SiamDataset.collate_fn(batch)
        print("\n✅ Collate successful!")
        print(f"\nCollated batch keys: {sorted(result.keys())}")
        print(f"\nQuery images (img): {result['img'].shape}")
        print(f"Support images (support_img): {result['support_img'].shape}")
        print(f"Query alias (query_img): {result.get('query_img', 'Not present').shape if 'query_img' in result else 'Not present'}")
        print(f"\nBounding boxes: {result['bboxes'].shape}")
        print(f"Classes: {result['cls'].shape}")
        print(f"Batch indices: {result['batch_idx'].shape}")
        print(f"Batch index values: {result['batch_idx']}")
        
        # Verify batch structure
        assert result['img'].shape == torch.Size([3, 3, 640, 640]), "Query images shape mismatch"
        assert result['support_img'].shape == torch.Size([3, 3, 640, 640]), "Support images shape mismatch"
        assert result['bboxes'].shape[0] == 3, "Total bboxes should be 3 (2+1+0)"
        assert result['cls'].shape[0] == 3, "Total classes should be 3"
        assert result['batch_idx'].shape[0] == 3, "Total batch_idx should be 3"
        
        # Verify batch_idx has correct offsets
        expected_batch_idx = torch.tensor([0, 0, 1])  # [0,0] from item 0, [0]+1=1 from item 1, [] from item 2
        assert torch.equal(result['batch_idx'], expected_batch_idx.float()), \
            f"batch_idx mismatch: expected {expected_batch_idx}, got {result['batch_idx']}"
        
        # Verify support files
        assert len(result['support_file']) == 3, "Should have 3 support files"
        print(f"\nSupport files: {result['support_file']}")
        
        # Verify query_img alias
        if 'query_img' in result:
            assert torch.equal(result['query_img'], result['img']), "query_img should be alias of img"
            print("\n✅ query_img alias correctly set")
        
        print("\n✅ All assertions passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_siam_collate()
    sys.exit(0 if success else 1)
