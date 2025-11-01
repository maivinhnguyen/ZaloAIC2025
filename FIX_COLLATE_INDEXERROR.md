# Fix: IndexError in YOLODataset.collate_fn

## Problem
Training failed immediately with:
```
IndexError: list index out of range
  File "ultralytics/data/dataset.py", line 367, in collate_fn
    value = values[i]
```

## Root Cause
The `YOLODataset.collate_fn` method assumed all batch items have **identical keys in the same order**:

```python
# Original problematic code
keys = batch[0].keys()  # Only gets keys from first item
values = list(zip(*[list(b.values()) for b in batch]))  # Fails if items have different keys
for i, k in enumerate(keys):
    value = values[i]  # IndexError when items have mismatched keys
```

When batch items had different keys (e.g., some with "masks", others without), the `zip` operation created a list shorter than expected, causing `values[i]` to fail with IndexError.

This commonly occurs when:
- Some images have masks/keypoints/segments while others don't
- Transforms conditionally add/remove keys
- Support images in Siamese networks have different keys than query images

## Solution
Modified `collate_fn` to:
1. **Collect all unique keys** from all batch items
2. **Process each key independently**, only including values from items that have that key
3. **Handle batch_idx specially** to add proper offsets

```python
# Fixed code
all_keys = set()
for b in batch:
    all_keys.update(b.keys())

for k in all_keys:
    values = []
    for b in batch:
        if k in b:  # Only include if item has this key
            values.append(b[k])
    
    if not values:
        continue
    
    # Handle batch_idx specially
    if k == "batch_idx":
        new_batch[k] = values
        continue
    
    # Process other keys...
    if k in {"img", "text_feats"}:
        value = torch.stack(values, 0)
    elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
        value = torch.cat(values, 0)
    # ...
```

## Changes Made

**File**: `ultralytics/data/dataset.py`

**Function**: `YOLODataset.collate_fn` (lines ~362-395)

### Key Modifications:
1. **Collect all keys** from all batch items instead of just `batch[0]`
2. **Gracefully handle missing keys** by checking `if k in b` before adding values
3. **Skip empty value lists** with `if not values: continue`
4. **Handle batch_idx separately** to properly add offsets before concatenation

## Testing
Created `test_collate_batch_keys.py` to verify the fix handles:
- ✅ Items with different optional keys (masks, keypoints)
- ✅ Empty tensors (no bboxes/labels)
- ✅ Proper batch_idx offset calculation
- ✅ Correct tensor stacking and concatenation

## Impact
- **Fixes**: Immediate training crash with IndexError
- **Enables**: Siamese training with different query/support keys
- **Maintains**: Backward compatibility with uniform batches
- **Improves**: Robustness for mixed datasets

## Related Files
- `ultralytics/data/dataset.py` - Main fix
- `test_collate_batch_keys.py` - Verification test

## Next Steps
1. ✅ Test fix with actual training
2. Monitor for any downstream issues with model forward pass
3. Verify batch_idx offsets are used correctly in loss calculation
