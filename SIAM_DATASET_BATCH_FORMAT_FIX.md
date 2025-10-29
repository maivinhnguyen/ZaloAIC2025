# SiamDataset Batch Format Fix

## Problem
Training fails with tensor shape mismatch:
```
RuntimeError: The size of tensor a (10) must match the size of tensor b (8400) at non-singleton dimension 0
```
At: `fg_target_bboxes = target_bboxes[fg_mask] / stride_tensor`

The loss function expects `target_bboxes` to have shape `[batch_size, num_boxes, 4]` but was getting an incompatible shape.

## Root Causes
1. **Duplicate collate_fn methods**: The `SiamDataset` class had two identical `collate_fn` methods defined, causing only the incomplete second one to be used
2. **Missing instance-to-tensor conversion**: When transforms are disabled, the `__getitem__` method wasn't converting `instances` objects back to raw `bboxes`, `cls`, and `batch_idx` tensors that the loss function expects

## Fixes Applied

### Fix 1: Remove Duplicate collate_fn
Removed the second incomplete `collate_fn` method that wasn't properly handling instances conversion.

**Before**: Two collate_fn methods, second one incomplete
```python
@staticmethod
def collate_fn(batch: list[dict]) -> dict:
    # First implementation - uses YOLODataset.collate_fn
    ...

@staticmethod
def collate_fn(batch: list[dict]) -> dict:
    # Second incomplete implementation
    ...
```

**After**: Single, working collate_fn
```python
@staticmethod
def collate_fn(batch: list[dict]) -> dict:
    """Collate Siamese samples, stacking support images alongside standard YOLO tensors."""
    collated = YOLODataset.collate_fn(batch)
    if "support_img" in collated:
        support_imgs = collated["support_img"]
        if isinstance(support_imgs, (list, tuple)):
            collated["support_img"] = torch.stack(list(support_imgs), 0)
    if "img" in collated and "query_img" not in collated:
        collated["query_img"] = collated["img"]
    return collated
```

### Fix 2: Extract bboxes/cls from instances when no transforms
When augmentation transforms are disabled, the code was leaving `instances` objects in the dict instead of extracting the raw tensors.

**Before**:
```python
else:
    query_data = deepcopy(label)
    support_data = deepcopy(support_label)
    query_data["img"] = torch.from_numpy(query_data["img"].transpose(2, 0, 1))
    support_data["img"] = torch.from_numpy(support_data["img"].transpose(2, 0, 1))
    # Missing bboxes and cls extraction!
```

**After**:
```python
else:
    # When no transforms, manually extract bboxes and cls from instances
    query_data = deepcopy(label)
    support_data = deepcopy(support_label)
    
    # Convert image to tensor
    query_data["img"] = torch.from_numpy(query_data["img"].transpose(2, 0, 1))
    support_data["img"] = torch.from_numpy(support_data["img"].transpose(2, 0, 1))
    
    # Extract bboxes and cls from instances
    if "instances" in query_data:
        instances = query_data.pop("instances")
        query_data["bboxes"] = torch.from_numpy(instances.bboxes) if len(instances) else torch.zeros((0, 4))
        query_data["cls"] = torch.from_numpy(instances.cls) if len(instances) else torch.zeros((0, 1))
        query_data["batch_idx"] = torch.zeros(len(instances))
    
    if "instances" in support_data:
        instances = support_data.pop("instances")
        support_data["bboxes"] = torch.from_numpy(instances.bboxes) if len(instances) else torch.zeros((0, 4))
        support_data["cls"] = torch.from_numpy(instances.cls) if len(instances) else torch.zeros((0, 1))
        support_data["batch_idx"] = torch.zeros(len(instances))
```

## Expected Batch Format
After these fixes, the batch dict should have:
```python
{
    "img": torch.Tensor,              # [B, 3, H, W]
    "query_img": torch.Tensor,        # [B, 3, H, W]
    "support_img": torch.Tensor,      # [B, 3, H, W]
    "bboxes": torch.Tensor,           # [sum(num_boxes), 4]
    "cls": torch.Tensor,              # [sum(num_boxes), 1]
    "batch_idx": torch.Tensor,        # [sum(num_boxes)]
    "instances": Instances (optional),
    ...
}
```

This matches what the loss function expects to receive.

## Verification
The loss function should now correctly extract:
```python
targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
```

And properly compute:
```python
fg_target_bboxes = target_bboxes[fg_mask] / stride_tensor
```

With matching tensor shapes.
