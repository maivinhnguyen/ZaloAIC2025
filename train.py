import sys
import os
# Ensure local ultralytics is used instead of pip-installed version
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics.models.siam_train import SiamDetectionTrainer
import argparse
import multiprocessing

def train(data, model, epochs, imgsz, batch, device):
    trainer = SiamDetectionTrainer(overrides={
        'model': model,
        'data': data,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'workers': 2,  # Reduced from 4 - multiprocessing with CUDA can be unstable with many workers
    })

    results = trainer.train()

def main():
    parser = argparse.ArgumentParser(description='Train SIAM detection model')
    parser.add_argument('--data', type=str, default='siam_coco.yaml', help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default=0, help='GPU device ID')
    parser.add_argument('--model', type=str, default='yolo11n.yaml', help='Model configuration file')

    args = parser.parse_args()
    train(data=args.data, model=args.model, epochs=args.epochs, 
          imgsz=args.imgsz, batch=args.batch, device=args.device)

if __name__ == '__main__':
    # Required for proper CUDA/multiprocessing support on Linux
    multiprocessing.set_start_method('spawn', force=True)
    main()


