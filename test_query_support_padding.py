"""Test that query and support images are padded to same dimensions."""

import torch
import sys
sys.path.insert(0, '.')

from ultralytics.data.dataset import SiamDataset

def test_query_support_size_matching():
    """Test that collate_fn pads query and support images to same size."""
    
    # Simulate a batch where query and support images have different sizes
    batch = [
        {
            "img": torch.rand(3, 640, 640),  # Query image
            "support_img": torch.rand(3, 635, 638),  # Support image (different size!)
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "cls": torch.tensor([[0.]]),
            "batch_idx": torch.tensor([0]),
            "support_file": "support1.jpg",
        },
        {
            "img": torch.rand(3, 638, 642),  # Different query size
            "support_img": torch.rand(3, 640, 640),  # Different support size
            "bboxes": torch.tensor([[0.2, 0.3, 0.4, 0.5]]),
            "cls": torch.tensor([[1.]]),
            "batch_idx": torch.tensor([0]),
            "support_file": "support2.jpg",
        },
    ]
    
    print("Testing query/support size matching in collate_fn...")
    print(f"\nBefore collation:")
    print(f"Item 0 - query: {batch[0]['img'].shape}, support: {batch[0]['support_img'].shape}")
    print(f"Item 1 - query: {batch[1]['img'].shape}, support: {batch[1]['support_img'].shape}")
    
    try:
        result = SiamDataset.collate_fn(batch)
        print("\n✅ Collate successful!")
        
        print(f"\nAfter collation:")
        print(f"Query images (img): {result['img'].shape}")
        print(f"Support images (support_img): {result['support_img'].shape}")
        
        # Verify both have same spatial dimensions
        query_shape = result['img'].shape
        support_shape = result['support_img'].shape
        
        assert query_shape == support_shape, \
            f"Query and support shapes don't match: {query_shape} vs {support_shape}"
        
        # Verify they're padded to the max dimensions
        assert query_shape[-2:] == torch.Size([640, 642]), \
            f"Expected padded size (640, 642), got {query_shape[-2:]}"
        
        print("\n✅ Query and support images have matching dimensions!")
        print(f"   Both padded to: {query_shape[-2:]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_query_support_size_matching()
    sys.exit(0 if success else 1)
