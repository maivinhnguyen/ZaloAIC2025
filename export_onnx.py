import torch
import onnx
from ultralytics.nn.tasks import SiamDetectionModel
from ultralytics.utils.torch_utils import select_device

def export_siam_to_onnx(weights_path, cfg_path, output_name="siam_yolo.onnx", opset=13, device_str="0"):
    device = select_device(device_str)
    print(f"Device: {device}")

    print(f"Loading checkpoint from {weights_path}...")
    ckpt = torch.load(weights_path, map_location="cpu")
    
    model = None
    
    # Method A: Try to load the full model object directly
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
        print("Detected full model object in checkpoint. Loading directly...")
        model = ckpt["model"]
        
        # Fix: Handle args whether it's a dict or Namespace
        if hasattr(model, "args"):
            if isinstance(model.args, dict):
                model.args["device"] = device
            else:
                try:
                    model.args.device = device
                except Exception:
                    pass # If args is immutable or weird, skip (usually fine for export)
    
    # Method B: Initialize from Config (Fallback)
    if model is None:
        print("Full model not found. Building from config...")
        # Force nc=1 for Siamese models
        model = SiamDetectionModel(cfg=cfg_path, nc=1) 
        
        # Load weights
        state_dict = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
        model.load_state_dict(state_dict)

    # Prepare model for export
    model.to(device)
    model.float()
    model.eval()

    # Set export flags
    for m in model.modules():
        if hasattr(m, "export"):
            m.export = True
        if hasattr(m, "format"):
            m.format = "onnx"

    # Create Dummy Inputs (Dynamic axes usually handle size changes, 
    # but starting with 640 is standard practice)
    imgsz = 960
    dummy_query = torch.randn(1, 3, imgsz, imgsz).to(device)
    dummy_support = torch.randn(1, 3, imgsz, imgsz).to(device)

    dynamic_axes = {
        "query": {0: "batch", 2: "height", 3: "width"},
        "support": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 1: "anchors"},
    }

    print(f"Exporting ONNX to {output_name} (opset {opset}) ...")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_query, dummy_support),
            output_name,
            input_names=["query", "support"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            verbose=False
        )

    # Validate
    onnx_model = onnx.load(output_name)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX export succeeded: {output_name}")

if __name__ == "__main__":
    # Update these paths if needed
    WEIGHTS = "runs/detect/train4/weights/last.pt"
    CFG = "ultralytics/cfg/models/11/yolo11.yaml"
    
    export_siam_to_onnx(WEIGHTS, CFG, output_name="siam_yolo.onnx", opset=13, device_str="0")