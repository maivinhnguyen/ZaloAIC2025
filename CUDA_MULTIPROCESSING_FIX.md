# CUDA Multiprocessing Fix - train.py

## Problem
Training fails with error:
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```

## Root Cause
The training script uses multiple data loader workers (originally 4) which create subprocesses. On Linux, Python's multiprocessing uses `fork()` by default, which doesn't work well with CUDA:

1. Main process initializes CUDA
2. Data loader spawns worker processes using `fork()`
3. Forked child processes try to access CUDA that was initialized in parent
4. CUDA cannot be re-initialized in forked subprocess → **Error!**

## Solution
Two changes to `train.py`:

### 1. Set multiprocessing start method to 'spawn'
```python
if __name__ == '__main__':
    # Required for proper CUDA/multiprocessing support on Linux
    multiprocessing.set_start_method('spawn', force=True)
    main()
```

This makes Python create completely new processes instead of forking, allowing each process to properly initialize CUDA.

### 2. Reduce number of workers
```python
'workers': 2,  # Reduced from 4 - multiprocessing with CUDA can be unstable with many workers
```

Fewer workers = fewer subprocesses = more stable CUDA operations and less resource contention.

## Why This Works
- **spawn method**: Each worker process is a fresh Python interpreter with its own CUDA context
- **Fewer workers**: Reduces CPU overhead and makes CUDA operations more predictable
- **Combined**: Stable training loop without CUDA reinitialization errors

## Notes
- This is especially important on Linux (Windows uses 'spawn' by default)
- The `force=True` parameter ensures it works even if called multiple times
- Training will be slightly slower but stable - data loading won't be the bottleneck with 2 workers anyway
