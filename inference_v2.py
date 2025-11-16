import argparse
import cv2
import torch
import numpy as np
import json
import os
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# Force 'spawn' for CUDA safety
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from ultralytics.nn.tasks import SiamDetectionModel
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.nms import non_max_suppression

# --- STRATEGY CONFIGURATION ---
TRACK_CFG = {
    "conf_init": 0.7,      # High confidence needed to START tracking (Anti-False Positive)
    "conf_sustain": 0.25,   # Low confidence allowed ONLY if overlapping previous box
    "iou_gate": 0.20,       # The new box must overlap the old box by at least 30%
    "max_coast": 5,         # If we lose the object, remember position for 5 frames before giving up
}

class SiameseDetector:
    def __init__(self, model_path, device="cuda:0", imgsz=640):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        self.imgsz = imgsz
        self.query_tensor = None
        # Suppress generic logs
        LOGGER.setLevel("ERROR")

    def _load_model(self, model_path):
        """Robust model loader for .pt or .engine"""
        # Note: For .engine (TensorRT), this simple loader might need adjustment 
        # depending on your ultralytics version, but usually AutoBackend handles it.
        # For this script, we assume standard .pt loading for safety.
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
            
        return model.to(self.device)

    def load_support_images(self, support_path: str):
        support_images = []
        path_obj = Path(support_path)
        if path_obj.is_dir():
            valid_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            for ext in valid_exts:
                for img_path in path_obj.glob(ext):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        support_images.append(img)
        
        if support_images:
            # Just use the first image for speed
            tensor, _ = self._preprocess_image(support_images[0])
            self.query_tensor = tensor
        else:
            self.query_tensor = None

    def _preprocess_image(self, image):
        orig_shape = image.shape[:2]
        img = self._letterbox(image, self.imgsz)
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

    def detect_candidates(self, frame):
        """
        Return ALL detections above the LOWEST threshold (conf_sustain).
        We will filter them intelligently in the main loop.
        """
        if self.query_tensor is None:
            return []
        
        frame_tensor, orig_shape = self._preprocess_image(frame)
        
        with torch.no_grad():
            predictions = self.model(frame_tensor, support_img=self.query_tensor)
        
        # NMS with the LOWEST threshold to ensure we don't kill weak detections early
        detections = non_max_suppression(
            predictions,
            conf_thres=TRACK_CFG["conf_sustain"], 
            iou_thres=0.45,
            max_det=10 # We only need top few candidates
        )
        
        results = []
        if len(detections) > 0 and len(detections[0]) > 0:
            det = detections[0]
            # Rescale boxes to original image size
            det[:, :4] = ops.scale_boxes(frame_tensor.shape[2:], det[:, :4], orig_shape).round()
            
            for i in range(len(det)):
                bbox = det[i, :4].cpu().numpy()
                conf = float(det[i, 4].cpu())
                results.append((bbox, conf))
                
        return results

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union <= 0: return 0
    return intersection / union

def process_video_wrapper(job_args):
    folder, model_path, device, _, _, imgsz = job_args
    video_id = folder.name
    
    try:
        # Initialize detector
        detector = SiameseDetector(model_path, device, imgsz=imgsz)
        
        video_path = folder / "drone_video.mp4"
        support_dir = folder / "object_images"

        if not video_path.exists():
            return {"video_id": video_id, "detections": []}

        detector.load_support_images(str(support_dir))
        if detector.query_tensor is None:
            return {"video_id": video_id, "detections": []}

        cap = cv2.VideoCapture(str(video_path))
        
        final_detections = []
        
        # --- TRACKER STATE VARIABLES ---
        last_bbox = None
        coast_counter = 0
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Get all potential candidates > 0.15 confidence
            candidates = detector.detect_candidates(frame)
            
            best_bbox = None
            # best_conf = 0.0 
            
            # 2. LOGIC ENGINE
            if last_bbox is None:
                # --- SEARCH MODE ---
                # We are lost or just starting. We need HIGH confidence (conf_init)
                # to accept a new object. This filters out random noise.
                
                valid_candidates = [c for c in candidates if c[1] >= TRACK_CFG["conf_init"]]
                
                if valid_candidates:
                    # Pick the one with highest confidence
                    valid_candidates.sort(key=lambda x: x[1], reverse=True)
                    best_bbox, _ = valid_candidates[0]
                    
            else:
                # --- TRACK MODE ---
                # We know where the object was. We look for a box that OVERLAPS (iou_gate).
                # If it overlaps, we accept it even if confidence is low (conf_sustain).
                
                best_match = None
                best_match_iou = -1
                
                for bbox, conf in candidates:
                    # Ignore trash
                    if conf < TRACK_CFG["conf_sustain"]: 
                        continue
                    
                    iou = calculate_iou(last_bbox, bbox)
                    
                    # SPATIAL GATE: Must overlap previous position
                    if iou > TRACK_CFG["iou_gate"]:
                        if iou > best_match_iou:
                            best_match_iou = iou
                            best_match = (bbox, conf)
                
                if best_match:
                    best_bbox, _ = best_match
                    coast_counter = 0 # Reset coasting, we found it!
                else:
                    # --- COASTING ---
                    # We didn't find a match this frame. 
                    # Should we keep the last known box? (optional)
                    # For ST-IoU, it's risky to freeze the box if it moved.
                    # We just increment counter. If we don't find it soon, we reset.
                    coast_counter += 1
                    if coast_counter > TRACK_CFG["max_coast"]:
                        last_bbox = None # Lost track completely
            
            # 3. UPDATE STATE & SAVE
            if best_bbox is not None:
                last_bbox = best_bbox
                x1, y1, x2, y2 = map(int, best_bbox)
                final_detections.append({
                    "frame": frame_idx,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
            
            frame_idx += 1
        
        cap.release()

        # Construct Result
        video_entry = {"video_id": video_id, "detections": []}
        if final_detections:
            video_entry["detections"].append({"bboxes": final_detections})
        
        return video_entry

    except Exception as e:
        print(f"❌ Error {video_id}: {e}")
        return {"video_id": video_id, "detections": []}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input data")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--output", type=str, default="submission.json")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640, help="640 for Jetson speed")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Device selection
    if args.gpus.lower() == 'cpu':
        devices = ['cpu']
    else:
        if torch.cuda.is_available():
            devices = [f'cuda:{gid.strip()}' for gid in args.gpus.split(',') if gid.strip()]
        else:
            devices = ['cpu']
    
    pool_size = len(devices) if 'cpu' not in devices else max(1, multiprocessing.cpu_count() // 2)
    
    print(f"🚀 Strategy: Sticky Box (Init: {TRACK_CFG['conf_init']}, Sustain: {TRACK_CFG['conf_sustain']})")
    print(f"🚀 Workers: {pool_size} | Devices: {devices}")

    root_dir = Path(args.input)
    subfolders = sorted([f for f in root_dir.iterdir() if f.is_dir()])
    
    if not subfolders:
        print("❌ No videos found.")
        return

    # Prepare jobs
    # Note: passing 0 as dummy conf/iou because we handle it in TRACK_CFG
    job_args = []
    for i, folder in enumerate(subfolders):
        device = devices[i % len(devices)]
        job_args.append((folder, args.model, device, 0, 0, args.imgsz))

    final_submission = []
    
    # Run Multiprocessing
    with multiprocessing.Pool(processes=pool_size) as pool:
        with tqdm(total=len(job_args), desc="Processing") as pbar:
            for result in pool.imap_unordered(process_video_wrapper, job_args):
                if result:
                    final_submission.append(result)
                pbar.update(1)

    # Save and Finish
    final_submission.sort(key=lambda x: x['video_id'])
    with open(args.output, 'w') as f:
        json.dump(final_submission, f, indent=4)

    print(f"\n✅ Done! Saved to: {args.output}")

if __name__ == "__main__":
    main()