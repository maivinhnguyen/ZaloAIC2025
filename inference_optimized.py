"""
ZaloAI 2025 Track 1: One-Shot Object Detection
Batch Inference Script (Asynchronous, Multi-GPU)

Usage:
    # Run on specific GPUs
    python inference_batch.py --model weights/best.pt --input /path/to/public_test/samples --output submission.json --gpus 0,1,2,3

    # Run on CPU
    python inference_batch.py --model weights/best.pt --input /path/to/public_test/samples --output submission.json --gpus cpu
"""

import argparse
import cv2
import torch
import numpy as np
import json
import os
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# Ensure multiprocessing uses 'spawn' start method for CUDA safety
multiprocessing.set_start_method('spawn', force=True)

from ultralytics.nn.tasks import SiamDetectionModel
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.nms import non_max_suppression


class SiameseDetector:
    """
    Siamese YOLO detector.
    This class is instantiated *within* each worker process.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 960
    ):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.query_tensor = None
        
        # Suppress Ultralytics logging in worker processes
        LOGGER.setLevel("ERROR")
        
        # print(f"[PID {os.getpid()}] Detector initialized on {device}") # Uncomment for debug

    def _load_model(self, model_path):
        """Load the Siamese YOLO model onto the specified device."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            if 'ema' in checkpoint and checkpoint['ema'] is not None:
                model = checkpoint['ema']
            elif 'model' in checkpoint and checkpoint['model'] is not None:
                model = checkpoint['model']
            else:
                cfg = checkpoint.get('model_cfg', 'yolo11n.yaml')
                nc = checkpoint.get('nc', 1)
                model = SiamDetectionModel(cfg=cfg, nc=nc, verbose=False)
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        
        if hasattr(model, 'float'):
            model = model.float()
            
        model = model.to(self.device)
        return model

    def load_support_images(self, support_path: str):
        """
        Load and preprocess support images.
        """
        support_images = []
        support_tensors = []
        support_path_obj = Path(support_path)
        
        if support_path_obj.is_dir():
            valid_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            for ext in valid_exts:
                for img_path in support_path_obj.glob(ext):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        support_images.append(img)
        
        if not support_images:
            raise ValueError(f"No valid support images found in {support_path}")

        for img in support_images:
            tensor, _ = self._preprocess_image(img)
            support_tensors.append(tensor)
        
        # Use the first valid image as the query tensor
        self.query_tensor = support_tensors[0] 

    def _preprocess_image(self, image, target_size=None):
        if target_size is None:
            target_size = self.imgsz
        
        orig_shape = image.shape[:2]
        img = self._letterbox(image, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        return img, orig_shape
    
    def _letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img

    def detect(self, frame):
        """Run detection on a single frame."""
        if self.query_tensor is None:
            raise RuntimeError("Support images not loaded!")

        frame_tensor, orig_shape = self._preprocess_image(frame)
        
        with torch.no_grad():
            predictions = self.model(frame_tensor, support_img=self.query_tensor)
        
        detections = self._postprocess(predictions, orig_shape, frame_tensor.shape)
        
        if len(detections) > 0:
            best_idx = np.argmax(detections[:, 4])
            bbox = detections[best_idx, :4]
            conf = detections[best_idx, 4]
            return bbox, float(conf)
        
        return None, 0.0

    def _postprocess(self, predictions, orig_shape, input_shape):
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0] if len(predictions) > 0 else predictions
        
        predictions = non_max_suppression(
            predictions,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            max_det=300
        )
        
        pred = predictions[0] if len(predictions) > 0 else torch.empty((0, 6))
        
        if len(pred):
            pred[:, :4] = ops.scale_boxes(input_shape[2:], pred[:, :4], orig_shape).round()
            pred = pred[pred[:, 4] >= self.conf_threshold]
            result = pred.cpu().numpy()
        else:
            result = np.empty((0, 6))
        
        return result


def process_video_wrapper(job_args):
    """
    Wrapper function for multiprocessing.
    This function is executed by each worker process.
    """
    # 1. Unpack arguments
    folder, model_path, device, conf, iou, imgsz = job_args
    video_id = folder.name
    
    try:
        # 2. Initialize detector *in this process*
        detector = SiameseDetector(
            model_path=model_path,
            device=device,
            conf_threshold=conf,
            iou_threshold=iou,
            imgsz=imgsz
        )

        # 3. Define paths
        video_path = folder / "drone_video.mp4"
        support_dir = folder / "object_images"

        if not video_path.exists() or not support_dir.exists():
            LOGGER.warning(f"Skipping {video_id}: Missing video or object_images")
            return {"video_id": video_id, "detections": []}

        # 4. Load support images
        detector.load_support_images(str(support_dir))

        # 5. Process video
        cap = cv2.VideoCapture(str(video_path))
        bboxes_list = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            bbox, _ = detector.detect(frame)

            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                bboxes_list.append({
                    "frame": frame_idx,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
            
            frame_idx += 1
        
        cap.release()

        # 6. Format result
        video_entry = {"video_id": video_id, "detections": []}
        if bboxes_list:
            video_entry["detections"].append({"bboxes": bboxes_list})
        
        return video_entry

    except Exception as e:
        print(f"❌ Error processing {video_id} on {device}: {e}")
        # Return empty result to ensure video is in submission
        return {"video_id": video_id, "detections": []}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch Inference for ZaloAI 2025 (Multi-GPU)")
    
    parser.add_argument("--input", type=str, required=True,
                        help="Path to 'public_test/samples' folder")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pt model file")
    parser.add_argument("--output", type=str, default="submission.json",
                        help="Path to output JSON file")
    
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated list of GPU IDs (e.g., '0,1,2') or 'cpu'")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # 1. Determine devices and pool size
    if args.gpus.lower() == 'cpu':
        devices = ['cpu']
        pool_size = max(1, multiprocessing.cpu_count() // 2)
        print("🔧 Running on CPU with {pool_size} processes...")
    else:
        if not torch.cuda.is_available():
            print("❌ CUDA not available! Falling back to CPU.")
            devices = ['cpu']
            pool_size = max(1, multiprocessing.cpu_count() // 2)
        else:
            devices = [f'cuda:{gid.strip()}' for gid in args.gpus.split(',') if gid.strip()]
            pool_size = len(devices)
            print(f"🚀 Running on {pool_size} GPUs: {devices}")

    # 2. Find all video folders
    root_dir = Path(args.input)
    if not root_dir.exists():
        print(f"❌ Input directory not found: {root_dir}")
        return
        
    subfolders = [f for f in root_dir.iterdir() if f.is_dir()]
    subfolders.sort()
    
    if not subfolders:
        print(f"❌ No video subfolders found in {root_dir}")
        return
        
    print(f"📂 Found {len(subfolders)} videos to process...")

    # 3. Create job argument list
    job_args = []
    for i, folder in enumerate(subfolders):
        device = devices[i % len(devices)] # Cycle through devices
        job_args.append((
            folder,
            args.model,
            device,
            args.conf,
            args.iou,
            args.imgsz
        ))

    # 4. Run multiprocessing pool
    final_submission = []
    
    with multiprocessing.Pool(processes=pool_size) as pool:
        with tqdm(total=len(job_args), desc="Processing Videos") as pbar:
            # imap_unordered is async and returns results as they complete
            for result in pool.imap_unordered(process_video_wrapper, job_args):
                if result:
                    final_submission.append(result)
                pbar.update(1)

    # 5. Sort results for deterministic output
    final_submission.sort(key=lambda x: x['video_id'])

    # 6. Save final submission
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(final_submission, f, indent=4)

    print(f"\n✅ Processing Complete.")
    print(f"📄 Submission file saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()