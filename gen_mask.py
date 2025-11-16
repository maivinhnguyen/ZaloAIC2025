import os
import re
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from ultralytics import YOLOE


def remove_padding(mask):
    """Crop out black borders from mask region."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return y0, y1, x0, x1


def process_image(image_path, model_path):
    pattern = re.compile(r"^([A-Za-z]+)_\d+_img_\d+\.jpg$")
    file_name = os.path.basename(image_path)
    match = pattern.match(file_name)
    if not match:
        print(f"Skipping unmatched file: {file_name}")
        return

    obj_name = match.group(1)
    print(f"\n>>> Processing {file_name} (object='{obj_name}')")

    # Initialize YOLOE fresh each time (your requirement)
    model = YOLOE(model_path)
    model.set_classes([obj_name], model.get_text_pe([obj_name]))

    # Predict
    results = model.predict(image_path, conf=0.01)
    if not results or results[0].masks is None:
        print(f"⚠️  No mask detected for {file_name}")
        return

    r = results[0]
    mask_found = False

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    h0, w0 = img_np.shape[:2]

    for i, cls_name in enumerate(r.names.values()):
        if cls_name.lower() == obj_name.lower():
            # --- Extract and resize mask to original image resolution ---
            mask = r.masks.data[i].float().unsqueeze(0).unsqueeze(0)  # ensure float32 and 4D
            mask_resized = F.interpolate(mask, size=(h0, w0), mode="bilinear", align_corners=False)[0, 0]
            mask_np = (mask_resized.cpu().numpy() > 0.5).astype(np.uint8)

            # --- Apply mask on black background and crop ---
            y0, y1, x0, x1 = remove_padding(mask_np)
            masked_img = np.zeros_like(img_np)
            masked_img[mask_np == 1] = img_np[mask_np == 1]
            cropped = masked_img[y0:y1, x0:x1]
            out_img = Image.fromarray(cropped)

            # --- Save result ---
            base, ext = os.path.splitext(file_name)
            out_path = os.path.join(os.path.dirname(image_path), f"{base}_mask{ext}")
            out_img.save(out_path)
            print(f"✅ Saved: {out_path}")

            mask_found = True
            break

    if not mask_found:
        print(f"⚠️  No matching mask for '{obj_name}' in {file_name}")

    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOE segmentation using text prompts inferred from filenames."
    )
    parser.add_argument("--folder", required=True, help="Path to folder containing images")
    parser.add_argument("--model", default="yoloe-11l-seg.pt", help="YOLOE model checkpoint path or name")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise ValueError(f"Invalid folder path: {args.folder}")

    files = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.lower().endswith(".jpg")]
    if not files:
        print("No .jpg images found in the given folder.")
        return

    for image_path in files:
        process_image(image_path, args.model)


if __name__ == "__main__":
    main()
