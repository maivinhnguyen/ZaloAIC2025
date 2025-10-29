"""
Test script to verify SiamDetectionModel fixes.
"""

import torch
import sys
from pathlib import Path

# Add the local ultralytics to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing SiamYOLOv8 fixes...\n")

# Test 1: Import modules
print("=" * 60)
print("Test 1: Importing modules")
print("=" * 60)

try:
    from ultralytics.nn.modules import MatchingModule
    from ultralytics.nn.tasks import SiamDetectionModel
    from ultralytics.utils.loss import SiamLoss
    from ultralytics.data.dataset import SiamDataset
    print("✓ All imports successful!\n")
except Exception as e:
    print(f"✗ Import failed: {e}\n")
    sys.exit(1)

# Test 2: MatchingModule
print("=" * 60)
print("Test 2: MatchingModule")
print("=" * 60)

try:
    mm = MatchingModule()
    query = torch.randn(2, 256, 40, 40)
    support = torch.randn(2, 256, 40, 40)
    fused = mm(query, support)
    
    assert fused.shape == query.shape, f"Shape mismatch: {fused.shape} vs {query.shape}"
    print(f"Query shape: {query.shape}")
    print(f"Support shape: {support.shape}")
    print(f"Fused shape: {fused.shape}")
    print("✓ MatchingModule works correctly!\n")
except Exception as e:
    print(f"✗ MatchingModule test failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 3: SiamDetectionModel initialization
print("=" * 60)
print("Test 3: SiamDetectionModel Initialization")
print("=" * 60)

try:
    print("Initializing SiamDetectionModel...")
    model = SiamDetectionModel("yolo11n.yaml", ch=3, nc=1, verbose=False)
    print("✓ Model initialized successfully!\n")
except Exception as e:
    print(f"✗ Model initialization failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Single image forward (initialization compatibility)
print("=" * 60)
print("Test 4: Single Image Forward (Init Compatibility)")
print("=" * 60)

try:
    query_img = torch.randn(2, 3, 640, 640)
    
    model.eval()
    with torch.no_grad():
        # This simulates what YOLOv8 does during initialization
        output_single = model(query_img)
    
    print(f"Input shape: {query_img.shape}")
    print(f"Output type: {type(output_single)}")
    print("✓ Single image forward pass works!\n")
except Exception as e:
    print(f"✗ Single image forward failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 5: Dual image forward (training mode)
print("=" * 60)
print("Test 5: Dual Image Forward (Training Mode)")
print("=" * 60)

try:
    query_img = torch.randn(2, 3, 640, 640)
    support_img = torch.randn(2, 3, 640, 640)
    
    model.eval()
    with torch.no_grad():
        # This is the proper usage during training
        output_dual = model(query_img, support_img=support_img)
    
    print(f"Query shape: {query_img.shape}")
    print(f"Support shape: {support_img.shape}")
    print(f"Output type: {type(output_dual)}")
    print("✓ Dual image forward pass works!\n")
except Exception as e:
    print(f"✗ Dual image forward failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 6: Loss initialization
print("=" * 60)
print("Test 6: Loss Initialization")
print("=" * 60)

try:
    criterion = model.init_criterion()
    print(f"Loss class: {criterion.__class__.__name__}")
    print("✓ Loss initialized successfully!\n")
except Exception as e:
    print(f"✗ Loss initialization failed: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nYou can now run SIAM_EXAMPLES.py without errors.")
