import onnxruntime as ort
import numpy as np
import cv2
import argparse
import os

class SiamONNXInference:
    def __init__(self, onnx_path, imgsz=640):
        self.imgsz = imgsz
        
        # Load ONNX model
        print(f"Loading {onnx_path}...")
        # Use CUDA if available, otherwise CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input names (critical for dual-input models)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        print(f"Inputs: {self.input_names}")
        print(f"Outputs: {self.output_names}")
        
        # Support cache
        self.support_tensor = None

    def preprocess(self, img_bgr):
        """Resize and normalize image to [1, 3, 640, 640]"""
        # 1. Letterbox Resize (preserve aspect ratio)
        shape = img_bgr.shape[:2]
        new_shape = (self.imgsz, self.imgsz)
        
        # Calculate scale factor
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        # Resize
        if shape[::-1] != new_unpad:
            img_resized = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_bgr

        # Add padding (Letterbox) to reach 640x640
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2  # Divide padding by 2 (center image)
        dh /= 2
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # 2. Convert to Blob (Batch, Channel, Height, Width)
        # Convert BGR -> RGB
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        
        # Transpose (H, W, C) -> (C, H, W)
        img_rgb = img_rgb.transpose(2, 0, 1)
        
        # Normalize 0-255 -> 0.0-1.0
        img_norm = img_rgb.astype(np.float32) / 255.0
        
        # Add Batch dimension -> (1, 3, 640, 640)
        img_batch = np.expand_dims(img_norm, axis=0)
        
        return img_batch, (r, dw, dh)

    def set_support(self, img_path):
        img = cv2.imread(img_path)
        if img is None: raise ValueError(f"Could not load support image: {img_path}")
        self.support_tensor, _ = self.preprocess(img)

    def predict(self, query_path, conf_thres=0.25, iou_thres=0.45):
        img = cv2.imread(query_path)
        if img is None: raise ValueError(f"Could not load query image: {query_path}")
        
        query_tensor, (ratio, pad_w, pad_h) = self.preprocess(img)
        
        if self.support_tensor is None:
            print("Error: Support image not set!")
            return

        # --- RUN INFERENCE ---
        # This matches the tuple input structure we exported
        inputs = {
            self.input_names[0]: query_tensor,   # 'query'
            self.input_names[1]: self.support_tensor # 'support'
        }
        
        # output[0] shape is typically (1, 5, 8400) for 1 class (4 coords + 1 conf)
        # or (1, 84, 8400) for 80 classes.
        outputs = self.session.run(self.output_names, inputs)
        raw_output = outputs[0] # [Batch, Channels, Anchors] -> [1, 5, 8400]
        
        # --- POST PROCESSING (Manual NMS) ---
        # Transpose to [Batch, Anchors, Channels] -> [1, 8400, 5]
        # Note: Check your raw output shape. Ultralytics export usually puts anchors in last dim
        predictions = np.squeeze(raw_output).T # Result: [8400, 5]
        
        boxes = []
        scores = []
        
        for i in range(predictions.shape[0]):
            row = predictions[i]
            # Row format: [x_center, y_center, width, height, confidence] (assuming nc=1)
            # If nc > 1, it would be [xc, yc, w, h, score1, score2...]
            
            # Confidence score (for 1 class, it's usually index 4. For generic YOLO, it's the max of classes)
            # Since we exported with nc=1, shape is likely [xc, yc, w, h, score]
            score = row[4]
            
            if score > conf_thres:
                # Convert xywh center to xyxy top-left
                xc, yc, w, h = row[0], row[1], row[2], row[3]
                x1 = (xc - w/2 - pad_w) / ratio
                y1 = (yc - h/2 - pad_h) / ratio
                x2 = (xc + w/2 - pad_w) / ratio
                y2 = (yc + h/2 - pad_h) / ratio
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))
        
        if not boxes:
            return img, []

        # Apply NMS (using OpenCV's built-in NMS)
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_thres, nms_threshold=iou_thres)
        
        final_dets = []
        for i in indices:
            # i might be a list [index] or just index depending on opencv version
            idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            box = boxes[idx]
            conf = scores[idx]
            final_dets.append((box, conf))
            
            # Draw
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return img, final_dets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='siam_yolo.onnx', help='Path to ONNX model')
    parser.add_argument('--query', type=str, required=True, help='Path to query image (scene)')
    parser.add_argument('--support', type=str, required=True, help='Path to support image (object)')
    parser.add_argument('--out', type=str, default='result.jpg', help='Output path')
    args = parser.parse_args()

    # Initialize
    detector = SiamONNXInference(args.model, imgsz=960)
    
    # Set Support
    detector.set_support(args.support)
    
    # Predict
    result_img, dets = detector.predict(args.query, conf_thres=0.1)
    
    print(f"Found {len(dets)} objects.")
    cv2.imwrite(args.out, result_img)
    print(f"Saved result to {args.out}")