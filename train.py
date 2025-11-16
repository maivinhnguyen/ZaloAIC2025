import sys
import os
import argparse
import multiprocessing

# Ensure local ultralytics is importable before any ultralytics imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Also propagate the project root to child processes (DDP spawns separate Python interpreters)
os.environ["PYTHONPATH"] = (
    PROJECT_ROOT + (os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")
)

from ultralytics.models.siam_train import SiamDetectionTrainer

def train(data, model, epochs, imgsz, batch, device):
    trainer = SiamDetectionTrainer(overrides={
        'project': 'runs/siam_detection',
        'name': '11s_cp_syn_stage_1',
        'model': model,
        'data': data,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'close_mosaic': 15,
        # 'resume': True,
        'workers': 4,  # Reduced from 4 - multiprocessing with CUDA can be unstable with many workers
        # 'close_mosaic': 5,
        # 'multi_scale': True,
        # 'optimizer': "AdamW",
        # "lr0": 0.00001,
    })

    results = trainer.train()

def main():
    parser = argparse.ArgumentParser(description='Train SIAM detection model')
    parser.add_argument('--data', type=str, default='siam_coco.yaml', help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=960, help='Input image size')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default=0, help='GPU device ID')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Model configuration file')

    args = parser.parse_args()
    train(data=args.data, model=args.model, epochs=args.epochs, 
          imgsz=args.imgsz, batch=args.batch, device=args.device)

if __name__ == '__main__':
    # Required for proper CUDA/multiprocessing support on Linux
    multiprocessing.set_start_method('spawn', force=True)
    main()