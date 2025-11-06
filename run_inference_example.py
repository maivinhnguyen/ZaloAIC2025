"""
Quick Example Script - Test Siamese YOLO Inference

This script demonstrates how to quickly test the inference pipeline.
Modify the paths below to match your setup.
"""

import subprocess
import sys
from pathlib import Path

# ========== CONFIGURATION - MODIFY THESE PATHS ==========

# Path to your trained model
MODEL_PATH = "runs/last.pt"

# Path to your query image (the object you want to detect)
QUERY_IMAGE = "C:/Projects/ZaloAIC/train/samples/Lifering_1/object_images/img_3.jpg"

# For video inference
VIDEO_PATH = "C:/Projects/ZaloAIC/train/samples/Lifering_1/drone_video.mp4"
VIDEO_OUTPUT = "output/video_results"

# For image inference
IMAGE_PATH = "dataset/images/val/query/Lifering_1_frame_002823.jpg"  # or folder path
IMAGE_OUTPUT = "output/image_results"

# Inference settings
DEVICE = "cuda"  # or "cpu"
CONFIDENCE = 0.5
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 960

# =========================================================


def check_paths():
    """Check if required paths exist."""
    errors = []
    
    if not Path(MODEL_PATH).exists():
        errors.append(f"❌ Model not found: {MODEL_PATH}")
        errors.append("   → Train your model first using train.py")
    
    if not Path(QUERY_IMAGE).exists():
        errors.append(f"❌ Query image not found: {QUERY_IMAGE}")
        errors.append("   → Update QUERY_IMAGE path in this script")
    
    if errors:
        print("\n⚠️  Setup Required:\n")
        for error in errors:
            print(error)
        print("\n📝 Update the paths at the top of this script\n")
        return False
    
    return True


def run_video_inference():
    """Run inference on video."""
    if not Path(VIDEO_PATH).exists():
        print(f"⚠️  Video not found: {VIDEO_PATH}")
        print("   → Update VIDEO_PATH in this script or skip video inference")
        return False
    
    print("\n🎬 Running Video Inference...")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Query: {QUERY_IMAGE}")
    print(f"   Video: {VIDEO_PATH}")
    print(f"   Output: {VIDEO_OUTPUT}\n")
    
    cmd = [
        sys.executable, "inference.py",
        "--model", MODEL_PATH,
        "--query", QUERY_IMAGE,
        "--video", VIDEO_PATH,
        "--output", VIDEO_OUTPUT,
        "--device", DEVICE,
        "--conf", str(CONFIDENCE),
        "--iou", str(IOU_THRESHOLD),
        "--imgsz", str(IMAGE_SIZE)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Video inference complete! Results saved to: {VIDEO_OUTPUT}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Video inference failed: {e}")
        return False


def run_image_inference():
    """Run inference on image(s)."""
    if not Path(IMAGE_PATH).exists():
        print(f"⚠️  Image/folder not found: {IMAGE_PATH}")
        print("   → Update IMAGE_PATH in this script or skip image inference")
        return False
    
    print("\n🖼️  Running Image Inference...")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Query: {QUERY_IMAGE}")
    print(f"   Source: {IMAGE_PATH}")
    print(f"   Output: {IMAGE_OUTPUT}\n")
    
    cmd = [
        sys.executable, "inference_image.py",
        "--model", MODEL_PATH,
        "--query", QUERY_IMAGE,
        "--source", IMAGE_PATH,
        "--output", IMAGE_OUTPUT,
        "--device", DEVICE,
        "--conf", str(CONFIDENCE),
        "--iou", str(IOU_THRESHOLD),
        "--imgsz", str(IMAGE_SIZE)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Image inference complete! Results saved to: {IMAGE_OUTPUT}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Image inference failed: {e}")
        return False


def main():
    """Main function."""
    print("="*60)
    print("🚀 Siamese YOLO Inference Example")
    print("="*60)
    
    # Check if paths are configured
    if not check_paths():
        return
    
    print("\n✅ Required files found!")
    print("\nWhat would you like to do?")
    print("  1. Run video inference")
    print("  2. Run image inference")
    print("  3. Run both")
    print("  4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_video_inference()
    elif choice == "2":
        run_image_inference()
    elif choice == "3":
        print("\n" + "="*60)
        run_video_inference()
        print("\n" + "="*60)
        run_image_inference()
        print("\n" + "="*60)
    elif choice == "4":
        print("\n👋 Goodbye!")
        return
    else:
        print("\n❌ Invalid choice. Please run again and select 1-4.")
        return
    
    print("\n" + "="*60)
    print("✨ Done! Check the output folders for results.")
    print("="*60)


if __name__ == "__main__":
    main()
