"""
Quick Example Script - Test Siamese YOLO Inference

This script demonstrates how to quickly test the inference pipeline.
Use command-line arguments to configure paths and settings.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Siamese YOLO Inference Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Video inference
  python run_inference_example.py --model runs/last.pt --query path/to/query.jpg --video path/to/video.mp4
  
  # Image inference
  python run_inference_example.py --model runs/last.pt --query path/to/query.jpg --image path/to/image.jpg
  
  # Both (interactive mode)
  python run_inference_example.py --model runs/last.pt --query path/to/query.jpg --both
        """
    )
    
    # Model and query (required for inference)
    parser.add_argument("--model", type=str, default="runs/last.pt",
                        help="Path to trained model (default: runs/last.pt)")
    parser.add_argument("--query", type=str, 
                        default="C:/Projects/ZaloAIC/train/samples/Lifering_1/object_images/img_3.jpg",
                        help="Path to query image")
    
    # Input sources (at least one required)
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file for inference")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image or folder of images")
    parser.add_argument("--both", action="store_true",
                        help="Run both video and image inference")
    
    # Output paths
    parser.add_argument("--video-output", type=str, default="output/video_results",
                        help="Output directory for video results")
    parser.add_argument("--image-output", type=str, default="output/image_results",
                        help="Output directory for image results")
    
    # Inference settings
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use (default: cuda)")
    parser.add_argument("--conf", type=float, default=0.8,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IOU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Image size for inference (default: 960)")
    
    return parser.parse_args()


def check_paths(args):
    """Check if required paths exist."""
    errors = []
    
    if not Path(args.model).exists():
        errors.append(f"❌ Model not found: {args.model}")
        errors.append("   → Train your model first using train.py")
    
    if not Path(args.query).exists():
        errors.append(f"❌ Query image not found: {args.query}")
        errors.append("   → Check the --query argument")
    
    if errors:
        print("\n⚠️  Setup Required:\n")
        for error in errors:
            print(error)
        print("\n📝 Check your arguments and try again\n")
        return False
    
    return True


def run_video_inference(args):
    """Run inference on video."""
    if not Path(args.video).exists():
        print(f"⚠️  Video not found: {args.video}")
        print("   → Check the --video argument or provide a valid video path")
        return False
    
    print("\n🎬 Running Video Inference...")
    print(f"   Model: {args.model}")
    print(f"   Query: {args.query}")
    print(f"   Video: {args.video}")
    print(f"   Output: {args.video_output}\n")
    
    cmd = [
        sys.executable, "inference.py",
        "--model", args.model,
        "--query", args.query,
        "--video", args.video,
        "--output", args.video_output,
        "--device", args.device,
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--imgsz", str(args.imgsz)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Video inference complete! Results saved to: {args.video_output}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Video inference failed: {e}")
        return False


def run_image_inference(args):
    """Run inference on image(s)."""
    if not Path(args.image).exists():
        print(f"⚠️  Image/folder not found: {args.image}")
        print("   → Check the --image argument or provide a valid path")
        return False
    
    print("\n🖼️  Running Image Inference...")
    print(f"   Model: {args.model}")
    print(f"   Query: {args.query}")
    print(f"   Source: {args.image}")
    print(f"   Output: {args.image_output}\n")
    
    cmd = [
        sys.executable, "inference_image.py",
        "--model", args.model,
        "--query", args.query,
        "--source", args.image,
        "--output", args.image_output,
        "--device", args.device,
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--imgsz", str(args.imgsz)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Image inference complete! Results saved to: {args.image_output}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Image inference failed: {e}")
        return False


def main():
    """Main function."""
    args = parse_arguments()
    
    print("="*60)
    print("🚀 Siamese YOLO Inference Example")
    print("="*60)
    
    # Check if paths are configured
    if not check_paths(args):
        return
    
    print("\n✅ Required files found!")
    
    # Determine what to run
    run_video = args.video or args.both
    run_image = args.image or args.both
    
    if not run_video and not run_image:
        print("\n⚠️  No inference target specified!")
        print("   Use --video, --image, or --both")
        print("\n   Example: python run_inference_example.py --model runs/last.pt --query query.jpg --video video.mp4")
        return
    
    if run_video:
        if args.video:
            run_video_inference(args)
        else:
            print("\n⚠️  No video path provided with --both flag")
    
    if run_image:
        if run_video:
            print("\n" + "="*60)
        if args.image:
            run_image_inference(args)
        else:
            print("\n⚠️  No image path provided with --both flag")
    
    print("\n" + "="*60)
    print("✨ Done! Check the output folders for results.")
    print("="*60)


if __name__ == "__main__":
    main()
