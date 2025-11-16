python inference_dual_template.py --input /mlcv2/Datasets/ZaloAI2025/track1/public_test/samples/ \
    --model runs/detect/train4/weights/last.pt \
    --output en/first_appearance.json \
    --imgsz 960 \
    --gpus 0,1,2,3,4,6

# Visualize specific video (use actual video_id like "BlackBox_0")
# python visualize_detections.py --json en/first_appearance.json --input /mlcv2/Datasets/ZaloAI2025/track1/public_test/samples/ --video-id BlackBox_0 --mode frames --output en/frames_output/

# Visualize all videos
# python visualize_detections.py --json en/first_appearance.json --input /mlcv2/Datasets/ZaloAI2025/track1/public_test/samples/ --mode frames --output en/frames_output/

# Crop detected objects (individual crops)
# python crop_detections.py --json en/first_appearance.json --input /mlcv2/Datasets/ZaloAI2025/track1/public_test/samples/ --output en/crops/ --padding 0

# Crop detected objects with grid view
# python crop_detections.py --json en/first_appearance.json --input /mlcv2/Datasets/ZaloAI2025/track1/public_test/samples/ --output en/crops/ --padding 15 --grid --grid-cols 6 --grid-size 180 180
