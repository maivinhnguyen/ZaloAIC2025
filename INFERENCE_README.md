# Siamese YOLO Inference Scripts - Complete Guide

This package includes two inference scripts for one-shot object detection:

## 📁 Files Included

1. **`inference.py`** - Video inference script
2. **`inference_image.py`** - Image inference script  
3. **`run_inference_example.py`** - Interactive example script
4. This guide

## 🎯 What is One-Shot Detection?

One-shot detection means you provide:
- **Query image**: A single clear example of what to find
- **Video/Images**: Where to find that object

The model detects all instances of that object in the video/images.

## 🚀 Quick Start

### 1. Video Inference

Detect objects in a video:

```bash
python inference.py \
    --model runs/detect/train/weights/best.pt \
    --query path/to/query_image.jpg \
    --video path/to/video.mp4 \
    --output results
```

**What you get:**
- `results/output_video.mp4` - Video with detected objects highlighted
- `results/frames/` - Sample frames extracted

### 2. Image Inference (Single Image)

```bash
python inference_image.py \
    --model runs/detect/train/weights/best.pt \
    --query path/to/query_image.jpg \
    --source path/to/image.jpg \
    --output results
```

### 3. Image Inference (Folder)

```bash
python inference_image.py \
    --model runs/detect/train/weights/best.pt \
    --query path/to/query_image.jpg \
    --source path/to/image_folder/ \
    --output results
```

### 4. Interactive Example

Edit paths in `run_inference_example.py` and run:

```bash
python run_inference_example.py
```

## ⚙️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | - | Path to model weights (.pt file) |
| `--query` | - | Path to query image (object to find) |
| `--source` or `--video` | - | Path to input image/video |
| `--output` | `inference_output` | Output directory |
| `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--conf` | 0.25 | Confidence threshold (0.0-1.0) |
| `--iou` | 0.45 | IoU threshold for NMS (0.0-1.0) |
| `--imgsz` | 640 | Input image size in pixels |
| `--no-save-video` | - | Only for video: don't save output video |

## 💡 Tips & Tricks

### Getting Better Results

1. **More detections** - Lower `--conf` to 0.1 or 0.15
2. **Fewer false positives** - Raise `--conf` to 0.4 or 0.5
3. **Query image tips:**
   - Use clear, well-lit images
   - Avoid cluttered backgrounds
   - Match the viewpoint/angle of target objects

### Performance Optimization

1. **Speed up inference:**
   ```bash
   --device cuda --imgsz 416
   ```

2. **Reduce memory usage:**
   ```bash
   --imgsz 320 --device cpu
   ```

3. **High precision mode:**
   ```bash
   --conf 0.5 --iou 0.5
   ```

## 📊 Output Structure

### Video Output
```
results/
├── output_video.mp4           # Annotated video
└── frames/
    ├── frame_000000.jpg       # Sample frames
    ├── frame_000100.jpg
    └── ...
```

### Image Output
```
results/
├── image1.jpg                 # Annotated images
├── image2.jpg
└── ...
```

## 🔧 Troubleshooting

### No Detections Found
**Solution:**
- Lower confidence threshold: `--conf 0.1`
- Check query image quality
- Verify model was trained properly

### Out of Memory / CUDA Error
**Solution:**
- Reduce image size: `--imgsz 320`
- Use CPU: `--device cpu`
- Process in batches

### Slow Processing
**Solution:**
- Use GPU: `--device cuda`
- Reduce image size: `--imgsz 416`
- Close other programs

### Model Load Error
**Solution:**
- Verify model path exists
- Check file is a valid .pt file
- Ensure model matches script (Siamese model)

## 📖 Example Workflows

### Example 1: Find a person in a video
```bash
python inference.py \
    --model runs/detect/train/weights/best.pt \
    --query person_reference.jpg \
    --video crowd_video.mp4 \
    --output results/person_detection \
    --conf 0.3
```

### Example 2: Search multiple images
```bash
python inference_image.py \
    --model runs/detect/train/weights/best.pt \
    --query target_object.jpg \
    --source test_images/ \
    --output results/search_results \
    --conf 0.25 \
    --device cuda
```

### Example 3: High precision detection
```bash
python inference.py \
    --model runs/detect/train/weights/best.pt \
    --query object.jpg \
    --video video.mp4 \
    --output results/precise \
    --conf 0.5 \
    --iou 0.5 \
    --imgsz 640
```

## 🎨 Output Features

Both scripts create annotated outputs with:
- **Bounding boxes** around detected objects
- **Confidence scores** for each detection
- **Class labels** (if multi-class model)
- **Query image** overlay in corner (shows what was detected)
- **Statistics** about detections

## 🔍 Advanced Usage

### Process with Custom Settings
```python
from inference import SiamYOLOInference

# Create inference engine
inference = SiamYOLOInference(
    model_path="runs/detect/train/weights/best.pt",
    device="cuda",
    conf_threshold=0.3,
    iou_threshold=0.4,
    imgsz=640
)

# Process video
inference.process_video(
    query_image_path="query.jpg",
    video_path="video.mp4",
    output_dir="results"
)
```

### Batch Processing
```bash
#!/bin/bash
for video in videos/*.mp4; do
    python inference.py \
        --model runs/detect/train/weights/best.pt \
        --query query.jpg \
        --video "$video" \
        --output "results/$(basename "$video" .mp4)"
done
```

## 📝 Requirements

- Python 3.8+
- PyTorch with CUDA support (optional but recommended)
- OpenCV
- tqdm
- NumPy

All installed with: `pip install -r requirements.txt`

## 🆘 Common Questions

**Q: Can I use different model architectures?**
A: These scripts are designed for Siamese YOLO models. They may work with other YOLO variants with modification.

**Q: How long does inference take?**
A: Depends on video length/resolution and hardware:
- GPU (CUDA): ~10-50 fps
- CPU: ~1-5 fps

**Q: Can I modify the query image during video?**
A: The current scripts use a fixed query. To change it mid-video, you'd need to modify the code.

**Q: What video formats are supported?**
A: MP4, AVI, MOV, MKV, and other formats supported by OpenCV (depends on ffmpeg installation).

---

**Happy detecting! 🎯**

For model training, see `train.py`
For validation, see `validate_dataset.py`
