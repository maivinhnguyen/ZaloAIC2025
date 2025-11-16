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
    "conf_init": 0.7,      # High confidence needed to START a new scene
    "conf_sustain": 0.25,   # Low confidence allowed ONLY if overlapping previous box
    "iou_gate": 0.20,       # The new box must overlap the old box by at least 20%
    "max_coast": 5,         # If we lose the object, wait 5 frames before starting new scene
    "template_weight": 0.6, # Weight for template-based detection (vs support image)
}

class SiameseDetector:
    def __init__(self, model_path, device="cuda:0", imgsz=640):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        self.imgsz = imgsz
        self.query_tensor = None  # Support image tensor
        self.template_tensor = None  # Template from first appearance
        LOGGER.setLevel("ERROR")

    def _load_model(self, model_path):
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
            tensor, _ = self._preprocess_image(support_images[0])
            self.query_tensor = tensor
        else:
            self.query_tensor = None

    def set_template_from_bbox(self, frame, bbox):
        """Extract template from frame using bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        # Add small padding if possible
        h, w = frame.shape[:2]
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        template_crop = frame[y1:y2, x1:x2]
        if template_crop.size > 0:
            tensor, _ = self._preprocess_image(template_crop)
            self.template_tensor = tensor
        else:
            self.template_tensor = None

    def clear_template(self):
        """Clear the template tensor"""
        self.template_tensor = None

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

    def detect_candidates(self, frame, use_dual_template=False):
        """
        Detect candidates using support image and optionally template
        
        Args:
            frame: Current frame
            use_dual_template: If True and template exists, combine both detections
        """
        if self.query_tensor is None:
            return []
        
        frame_tensor, orig_shape = self._preprocess_image(frame)
        
        # Detection with support image
        with torch.no_grad():
            predictions_support = self.model(frame_tensor, support_img=self.query_tensor)
        
        detections_support = non_max_suppression(
            predictions_support,
            conf_thres=TRACK_CFG["conf_sustain"], 
            iou_thres=0.45,
            max_det=10
        )
        
        results_support = []
        if len(detections_support) > 0 and len(detections_support[0]) > 0:
            det = detections_support[0]
            det[:, :4] = ops.scale_boxes(frame_tensor.shape[2:], det[:, :4], orig_shape).round()
            
            for i in range(len(det)):
                bbox = det[i, :4].cpu().numpy()
                conf = float(det[i, 4].cpu())
                results_support.append((bbox, conf, 'support'))
        
        # If dual template is enabled and we have a template, run detection with it
        if use_dual_template and self.template_tensor is not None:
            with torch.no_grad():
                predictions_template = self.model(frame_tensor, support_img=self.template_tensor)
            
            detections_template = non_max_suppression(
                predictions_template,
                conf_thres=TRACK_CFG["conf_sustain"],
                iou_thres=0.45,
                max_det=10
            )
            
            results_template = []
            if len(detections_template) > 0 and len(detections_template[0]) > 0:
                det = detections_template[0]
                det[:, :4] = ops.scale_boxes(frame_tensor.shape[2:], det[:, :4], orig_shape).round()
                
                for i in range(len(det)):
                    bbox = det[i, :4].cpu().numpy()
                    conf = float(det[i, 4].cpu())
                    results_template.append((bbox, conf, 'template'))
            
            # Merge results: boost template detections
            merged_results = self._merge_dual_detections(results_support, results_template)
            return merged_results
        
        return [(bbox, conf, 'support') for bbox, conf, _ in results_support]

    def _merge_dual_detections(self, support_dets, template_dets):
        """
        Merge detections from support and template
        Boost confidence for detections that appear in both
        """
        if not template_dets:
            return [(bbox, conf, src) for bbox, conf, src in support_dets]
        
        if not support_dets:
            return [(bbox, conf, src) for bbox, conf, src in template_dets]
        
        merged = []
        used_template = set()
        
        # For each support detection, check if it matches a template detection
        for s_bbox, s_conf, _ in support_dets:
            best_match = None
            best_iou = 0
            best_idx = -1
            
            for idx, (t_bbox, t_conf, _) in enumerate(template_dets):
                if idx in used_template:
                    continue
                iou = calculate_iou(s_bbox, t_bbox)
                if iou > 0.3 and iou > best_iou:  # IoU threshold for matching
                    best_iou = iou
                    best_match = (t_bbox, t_conf)
                    best_idx = idx
            
            if best_match:
                # Matched: combine confidences (weighted average favoring template)
                t_bbox, t_conf = best_match
                combined_conf = (1 - TRACK_CFG["template_weight"]) * s_conf + TRACK_CFG["template_weight"] * t_conf
                # Use average bbox
                avg_bbox = (s_bbox + t_bbox) / 2
                merged.append((avg_bbox, combined_conf, 'dual'))
                used_template.add(best_idx)
            else:
                # No match: keep support detection
                merged.append((s_bbox, s_conf, 'support'))
        
        # Add unmatched template detections
        for idx, (t_bbox, t_conf, src) in enumerate(template_dets):
            if idx not in used_template:
                merged.append((t_bbox, t_conf, src))
        
        return merged

def calculate_iou(box1, box2):
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
        detector = SiameseDetector(model_path, device, imgsz=imgsz)
        
        video_path = folder / "drone_video.mp4"
        support_dir = folder / "object_images"

        if not video_path.exists():
            return {"video_id": video_id, "detections": []}

        detector.load_support_images(str(support_dir))
        if detector.query_tensor is None:
            return {"video_id": video_id, "detections": []}

        cap = cv2.VideoCapture(str(video_path))
        
        scene_first_detections = []  # List of first frames per scene
        
        # --- TRACKER STATE VARIABLES ---
        last_bbox = None
        coast_counter = 0
        in_scene = False
        current_frame = None
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame = frame.copy()
            
            # Decide whether to use dual template
            use_dual = in_scene and detector.template_tensor is not None
            
            candidates = detector.detect_candidates(frame, use_dual_template=use_dual)
            
            best_bbox = None
            
            # LOGIC ENGINE
            if last_bbox is None:
                # --- SEARCH MODE: Looking for NEW SCENE ---
                valid_candidates = [c for c in candidates if c[1] >= TRACK_CFG["conf_init"]]
                
                if valid_candidates:
                    valid_candidates.sort(key=lambda x: x[1], reverse=True)
                    best_bbox, _, _ = valid_candidates[0]
                    
                    # NEW SCENE STARTED - Save first appearance and create template
                    x1, y1, x2, y2 = map(int, best_bbox)
                    scene_first_detections.append({
                        "frame": frame_idx,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })
                    in_scene = True
                    
                    # Extract template from first detection
                    detector.set_template_from_bbox(current_frame, best_bbox)
                    
            else:
                # --- TRACK MODE: Continue tracking current scene ---
                best_match = None
                best_match_iou = -1
                
                for bbox, conf, source in candidates:
                    if conf < TRACK_CFG["conf_sustain"]: 
                        continue
                    
                    iou = calculate_iou(last_bbox, bbox)
                    
                    if iou > TRACK_CFG["iou_gate"]:
                        if iou > best_match_iou:
                            best_match_iou = iou
                            best_match = (bbox, conf, source)
                
                if best_match:
                    best_bbox, _, _ = best_match
                    coast_counter = 0
                else:
                    # COASTING: Lost object temporarily
                    coast_counter += 1
                    if coast_counter > TRACK_CFG["max_coast"]:
                        # SCENE ENDED - Reset to search for next scene
                        last_bbox = None
                        in_scene = False
                        detector.clear_template()  # Clear template for next scene
            
            # UPDATE STATE
            if best_bbox is not None:
                last_bbox = best_bbox
            
            frame_idx += 1
        
        cap.release()

        # Construct Result
        video_entry = {"video_id": video_id, "detections": []}
        if scene_first_detections:
            video_entry["detections"].append({"bboxes": scene_first_detections})
        
        return video_entry

    except Exception as e:
        print(f"❌ Error {video_id}: {e}")
        return {"video_id": video_id, "detections": []}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input data")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--output", type=str, default="dual_template_output.json")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640, help="640 for Jetson speed")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.gpus.lower() == 'cpu':
        devices = ['cpu']
    else:
        if torch.cuda.is_available():
            devices = [f'cuda:{gid.strip()}' for gid in args.gpus.split(',') if gid.strip()]
        else:
            devices = ['cpu']
    
    pool_size = len(devices) if 'cpu' not in devices else max(1, multiprocessing.cpu_count() // 2)
    
    print(f"🚀 Strategy: Dual Template (Support + First Appearance)")
    print(f"🚀 Workers: {pool_size} | Devices: {devices}")

    root_dir = Path(args.input)
    subfolders = sorted([f for f in root_dir.iterdir() if f.is_dir()])
    
    if not subfolders:
        print("❌ No videos found.")
        return

    job_args = []
    for i, folder in enumerate(subfolders):
        device = devices[i % len(devices)]
        job_args.append((folder, args.model, device, 0, 0, args.imgsz))

    final_submission = []
    
    with multiprocessing.Pool(processes=pool_size) as pool:
        with tqdm(total=len(job_args), desc="Processing") as pbar:
            for result in pool.imap_unordered(process_video_wrapper, job_args):
                if result:
                    final_submission.append(result)
                pbar.update(1)

    final_submission.sort(key=lambda x: x['video_id'])
    with open(args.output, 'w') as f:
        json.dump(final_submission, f, indent=4)

    print(f"\n✅ Done! Saved to: {args.output}")

if __name__ == "__main__":
    main()
