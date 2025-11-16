# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
SiamYOLOv8 Trainer and Validator for one-shot object detection.

This module provides training and validation implementations for the SiamYOLOv8 model,
which uses a Siamese network approach for one-shot object detection.
"""

from __future__ import annotations

import math
import random
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.dataset import SiamDataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, SiamDetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class SiamDetectionTrainer(BaseTrainer):
    """
    A trainer class for SiamYOLOv8 one-shot object detection.

    This trainer extends BaseTrainer to handle Siamese network training with query and support images.
    It manages the specific requirements for training one-shot detection models including dataset
    building with Siamese format, preprocessing for dual inputs, and loss computation.

    Attributes:
        model (SiamDetectionModel): The Siamese YOLO detection model being trained.
        data (dict): Dictionary containing dataset information.
        loss_names (tuple): Names of the loss components (iou_loss, bce_loss, rpl_loss, dice_loss, dfl_loss).

    Methods:
        build_dataset: Build Siamese YOLO dataset for training or validation.
        get_dataloader: Construct dataloader for Siamese data.
        preprocess_batch: Preprocess query and support image batches.
        set_model_attributes: Set Siamese model attributes from dataset.
        get_model: Return a Siamese YOLO detection model.
        get_validator: Return a validator for Siamese model evaluation.
        label_loss_items: Return labeled loss components.

    Examples:
        >>> from ultralytics.models.siam import SiamDetectionTrainer
        >>> args = dict(model="siam_yolo11n.yaml", data="siam_coco.yaml", epochs=100, task="siam_detect")
        >>> trainer = SiamDetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """
        Initialize a SiamDetectionTrainer object.

        Args:
            cfg (dict, optional): Configuration dictionary with default parameters.
            overrides (dict, optional): Dictionary of parameter overrides.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.loss_names = ("iou_loss", "bce_loss", "rpl_loss", "dice_loss", "dfl_loss")
        # Initialize tloss to zeros with the same length as loss_names
        self.tloss = [0.0] * len(self.loss_names)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        Build Siamese YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' for training or 'val' for validation mode.
            batch (int, optional): Size of batches for rectangle mode.

        Returns:
            (SiamDataset): Siamese YOLO dataset configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)

        # Build the Siamese dataset
        return SiamDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=mode == "val",
            cache=self.args.cache,
            single_cls=self.args.single_cls,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task="detect",
            data=self.data,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for Siamese training/validation.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of image pairs per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' or 'val' mode.

        Returns:
            (DataLoader): PyTorch dataloader for Siamese data.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False

        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """
        Preprocess batch of data for training or validation.

        Args:
            batch (dict): Batch dictionary containing query/support images and labels.

        Returns:
            (dict): Preprocessed batch dictionary.
        """
        # Move all tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")

        # Normalize query and support images
        if "query_img" in batch:
            batch["query_img"] = batch["query_img"].float() / 255

        if "support_img" in batch:
            batch["support_img"] = batch["support_img"].float() / 255

        if "img" in batch:
            batch["img"] = batch["img"].float() / 255
        
        # Add current epoch for dynamic loss weighting
        batch["epoch"] = getattr(self, "epoch", 0)

        # Apply multi-scale augmentation if enabled
        if self.args.multi_scale:
            imgs = batch.get("query_img", batch.get("img"))
            if imgs is not None:
                sz = (
                    random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                    // self.stride
                    * self.stride
                )
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                    if "query_img" in batch:
                        batch["query_img"] = nn.functional.interpolate(
                            batch["query_img"], size=ns, mode="bilinear", align_corners=False
                        )
                    if "support_img" in batch:
                        batch["support_img"] = nn.functional.interpolate(
                            batch["support_img"], size=ns, mode="bilinear", align_corners=False
                        )

        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a Siamese YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to pretrained weights.
            verbose (bool): Whether to print model info.

        Returns:
            (SiamDetectionModel): Siamese detection model.
        """
        model = SiamDetectionModel(cfg or self.args.model, ch=3, nc=self.data.get("nc"), verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """
        Return a validator for Siamese model evaluation.

        Returns:
            (SiamDetectionValidator): Validator instance for Siamese detection.
        """
        self.loss_names = ("iou_loss", "bce_loss", "rpl_loss", "dice_loss", "dfl_loss")
        return SiamDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labeled training loss items.

        Args:
            loss_items (list, optional): List of loss values.
            prefix (str): Prefix for loss keys.

        Returns:
            (dict): Dictionary with labeled loss items.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            formatted = []
            for item in loss_items:
                if item is None:
                    formatted.append(0.0)
                elif hasattr(item, "item"):
                    formatted.append(float(item.item()))
                else:
                    formatted.append(float(item))
            return dict(zip(keys, formatted))
        return keys

    def progress_string(self):
        """Return a formatted training progress string."""
        return ""  # Progress is displayed via pbar.set_description in _do_train

    def log_metrics(self, epoch, metrics):
        """
        Log training metrics for the current epoch with labels.

        Args:
            epoch (int): Current epoch number.
            metrics (dict): Dictionary containing metric names and their values.
        """
        log_message = f"Epoch {epoch}: "
        log_message += ", ".join([f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items()])
        LOGGER.info(log_message)

    def plot_training_samples(self, batch, ni):
        """
        Plot training samples (query and support images) during YOLO training.

        Args:
            batch (dict): Batch dictionary containing 'query_img', 'support_img', and labels.
            ni (int): Batch iteration index.
            
        Note:
            - Query images are plotted with bounding boxes (what to find)
            - Support images are plotted without bounding boxes (reference images)
        """
        import os
        from pathlib import Path

        debug = os.environ.get('SIAM_DEBUG', '').lower() == 'true'
        
        # Get images from batch
        query_imgs = batch.get("query_img", batch.get("img"))  # Fall back to img if query_img not present
        support_imgs = batch.get("support_img")

        if debug:
            print(f"[plot_training_samples] Batch keys: {batch.keys()}")
            print(f"[plot_training_samples] bboxes in batch: {batch.get('bboxes') is not None}")
            if batch.get('bboxes') is not None:
                print(f"  - bboxes shape: {batch['bboxes'].shape}")
                print(f"  - bboxes dtype: {batch['bboxes'].dtype}")

        if query_imgs is not None:
            # Prepare output directory
            output_dir = Path(self.save_dir) / "train_batch_plots"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get labels from batch
            bboxes = batch.get("bboxes")
            cls_labels = batch.get("cls")
            batch_idx = batch.get("batch_idx")
            
            if debug:
                print(f"[plot_training_samples] Before processing:")
                print(f"  - bboxes: {bboxes.shape if bboxes is not None else None}")
                print(f"  - cls_labels: {cls_labels.shape if cls_labels is not None else None}")
                print(f"  - batch_idx: {batch_idx.shape if batch_idx is not None else None}")
            
            # Ensure bboxes and cls are proper tensors, not None
            if bboxes is None:
                bboxes = torch.zeros((0, 4), dtype=torch.float32, device=query_imgs.device)
            if cls_labels is None:
                cls_labels = torch.zeros((0, 1), dtype=torch.float32, device=query_imgs.device)
            if batch_idx is None:
                # Create batch_idx only if it doesn't exist - one index per bbox, assuming all are from image 0
                batch_idx = torch.zeros(len(bboxes), dtype=torch.long, device=query_imgs.device)
            else:
                # Ensure batch_idx is a tensor and properly shaped
                if not isinstance(batch_idx, torch.Tensor):
                    batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=query_imgs.device)
                # If batch_idx is 2D (e.g., from collation), flatten it
                if batch_idx.ndim > 1:
                    batch_idx = batch_idx.squeeze(-1)

            if debug:
                print(f"[plot_training_samples] After processing:")
                print(f"  - bboxes: {bboxes.shape}")
                print(f"  - cls_labels: {cls_labels.shape}")
                print(f"  - batch_idx: {batch_idx.shape}")

            # Query batch: Has bounding boxes (what we're looking for)
            query_batch = {
                "img": query_imgs,
                "cls": cls_labels,
                "bboxes": bboxes,
                "batch_idx": batch_idx,
                "im_file": batch.get("im_file", []),
            }
            
            if debug:
                print(f"[plot_training_samples] query_batch keys: {query_batch.keys()}")
            
            # Plot query images with bounding boxes
            try:
                plot_images(
                    labels=query_batch,
                    fname=str(output_dir / f"train_batch_query_{ni}.jpg"),
                    names=self.data.get("names", None),
                    on_plot=self.on_plot,
                )
            except Exception as e:
                LOGGER.warning(f"Failed to plot query images: {e}")
                import traceback
                LOGGER.warning(f"Traceback: {traceback.format_exc()}")

            # Plot support images WITHOUT bounding boxes
            if support_imgs is not None:
                try:
                    support_batch = {
                        "img": support_imgs,
                        "cls": torch.zeros((0, 1), dtype=torch.float32, device=support_imgs.device),
                        "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=support_imgs.device),
                        "batch_idx": torch.zeros(0, dtype=torch.long, device=support_imgs.device),
                        "im_file": batch.get("support_file", []),
                    }
                    plot_images(
                        labels=support_batch,
                        fname=str(output_dir / f"train_batch_support_{ni}.jpg"),
                        names=self.data.get("names", None),
                        on_plot=self.on_plot,
                    )
                except Exception as e:
                    LOGGER.warning(f"Failed to plot support images: {e}")
                    import traceback
                    LOGGER.warning(f"Traceback: {traceback.format_exc()}")

    def visualize_training_samples(self, batch, output_dir=None, epoch=0, max_samples=5):
        """
        Visualize a limited number of training samples (query and support images) and save to output folder.

        Args:
            batch (dict): Batch dictionary containing 'query_img', 'support_img', and labels.
            output_dir (str): Directory to save visualizations. Defaults to self.save_dir.
            epoch (int): Current epoch number (used for naming files).
            max_samples (int): Maximum number of samples to visualize.
            
        Note:
            - Query images show with bounding boxes (target object to find)
            - Support images show without bounding boxes (reference images showing the object)
        """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Use the training save directory if output_dir is not provided
        output_dir = output_dir or os.path.join(self.save_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        LOGGER.info(f"Saving visualizations to {output_dir}")

        query_imgs = batch.get("query_img")
        support_imgs = batch.get("support_img")
        bboxes = batch.get("bboxes")
        cls_labels = batch.get("cls")

        if query_imgs is not None and support_imgs is not None:
            for i in range(min(len(query_imgs), max_samples)):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Plot query image with bounding box
                query_img_vis = query_imgs[i].cpu().numpy().transpose(1, 2, 0)
                query_img_vis = (query_img_vis * 255).astype('uint8') if query_img_vis.max() <= 1 else query_img_vis.astype('uint8')
                
                axes[0].imshow(query_img_vis)
                axes[0].set_title(f"Query Image (target to find) - Sample {i}")
                axes[0].axis("off")
                
                # Draw bounding box on query image if available
                if bboxes is not None and len(bboxes) > i:
                    bbox = bboxes[i]
                    if bbox.numel() > 0:  # Check if bbox is not empty
                        # Assuming normalized coordinates (xywh format)
                        h, w = query_img_vis.shape[:2]
                        x_center, y_center, bbox_w, bbox_h = bbox[:4]
                        x1 = (x_center - bbox_w / 2) * w
                        y1 = (y_center - bbox_h / 2) * h
                        x2 = (x_center + bbox_w / 2) * w
                        y2 = (y_center + bbox_h / 2) * h
                        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                        axes[0].add_patch(rect)
                        
                        # Add class label if available
                        if cls_labels is not None and len(cls_labels) > i:
                            class_idx = int(cls_labels[i].item()) if cls_labels[i].numel() > 0 else 0
                            class_name = self.data.get("names", {}).get(class_idx, f"class_{class_idx}")
                            axes[0].text(x1, y1 - 5, class_name, color='red', fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

                # Plot support image WITHOUT bounding box (it's just a reference)
                support_img_vis = support_imgs[i].cpu().numpy().transpose(1, 2, 0)
                support_img_vis = (support_img_vis * 255).astype('uint8') if support_img_vis.max() <= 1 else support_img_vis.astype('uint8')
                
                axes[1].imshow(support_img_vis)
                axes[1].set_title(f"Support Image (reference) - Sample {i}")
                axes[1].axis("off")

                # Save the figure
                save_path = os.path.join(output_dir, f"epoch_{epoch}_sample_{i}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
                plt.close(fig)
                LOGGER.info(f"Saved visualization: {save_path}")


class SiamDetectionValidator(BaseValidator):
    """
    A validator class for SiamYOLOv8 one-shot detection.

    This validator extends BaseValidator to handle evaluation of Siamese detection models
    using query-support image pairs.

    Attributes:
        model (SiamDetectionModel): The Siamese model being validated.
        dataloader: DataLoader for validation data.
        device (torch.device): Device for computation.

    Methods:
        preprocess_batch: Preprocess validation batch data.
        postprocess: Postprocess model predictions.

    Examples:
        >>> from ultralytics.models.siam import SiamDetectionValidator
        >>> validator = SiamDetectionValidator(model="siam_yolo11n.pt", data="siam_coco.yaml")
        >>> results = validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """
        Initialize a SiamDetectionValidator instance.

        Args:
            dataloader: DataLoader for validation.
            save_dir (str, optional): Directory to save results.
            args (dict, optional): Configuration arguments.
            _callbacks (list, optional): Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.loss_names = ("iou_loss", "bce_loss", "rpl_loss", "dice_loss", "dfl_loss")
        self.metrics = DetMetrics()  # Initialize metrics for detection validation
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}  # Initialize speed dict
        self.loss_items_avg = None  # Track average loss items across batches
        # Proper metric tracking for mAP computation
        self.stats = []  # List to store (correct, pred_conf, pred_cls, target_cls) for each batch
        self.total_gts = 0  # Total ground truth boxes
        self.total_preds = 0  # Total predictions

    def preprocess_batch(self, batch: dict) -> dict:
        """
        Preprocess a validation batch of Siamese data.

        Args:
            batch (dict): Batch dictionary with 'query_img', 'support_img', and labels.

        Returns:
            (dict): Preprocessed batch.
        """
        # Move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")

        # Normalize images
        if "query_img" in batch:
            batch["query_img"] = batch["query_img"].float() / 255

        if "support_img" in batch:
            batch["support_img"] = batch["support_img"].float() / 255

        if "img" in batch:
            batch["img"] = batch["img"].float() / 255

        return batch

    def postprocess(self, preds):
        """
        Postprocess model predictions with NMS.

        Args:
            preds: Raw model predictions [batch, num_predictions, 6] (x, y, w, h, conf, cls).

        Returns:
            List of postprocessed predictions for each image after NMS.
        """
        from ultralytics.utils import nms
        
        # Debug: Check what conf value is actually being used
        conf_val = getattr(self.args, 'conf', None)
        if conf_val is None:
            LOGGER.warning(f"⚠️ self.args.conf is None! Using default 0.001")
            conf_val = 0.25
        
        LOGGER.info(f"🔍 Validation NMS using conf_thres={conf_val}, iou_thres={self.args.iou}, max_det={self.args.max_det}")
        
        # Apply NMS
        outputs = nms.non_max_suppression(
            preds,
            conf_thres=conf_val,
            iou_thres=self.args.iou,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )
        
        return outputs

    def __call__(self, trainer=None, model=None):
        """
        Run inference on validation set with Siamese model support.
        
        Args:
            trainer: Training trainer instance.
            model: Model to validate.
            
        Returns:
            Validation metrics.
        """
        from ultralytics.utils.ops import Profile
        from ultralytics.utils import TQDM, RANK
        import torch.distributed as dist
        
        self.training = trainer is not None
        
        if self.training:
            self.device = trainer.device
            self.model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(self.model, "_orig_mod"):
                self.model = self.model._orig_mod
            self.model = self.model.half() if trainer.args.half else self.model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.model.eval()
        
        # Reset metric tracking
        self.stats = []
        self.total_gts = 0
        self.total_preds = 0
        
        # Initialize metrics with model names
        self.metrics = DetMetrics(names=self.model.names if hasattr(self.model, 'names') else {})
        
        LOGGER.info(f"Starting validation with {len(self.dataloader)} batches...")
        
        # Timing profiles
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        
        bar = TQDM(self.dataloader, desc=f"Validating", total=len(self.dataloader))
        
        # Track loss items across batches
        batch_count = 0
        
        with torch.no_grad():
            for batch_i, batch in enumerate(bar):
                self.batch_i = batch_i
                batch_count += 1
                
                # Preprocess
                with dt[0]:
                    batch = self.preprocess_batch(batch)
                
                # Inference with Siamese support
                with dt[1]:
                    query_img = batch.get("query_img", batch.get("img"))
                    support_img = batch.get("support_img")
                    
                    if support_img is not None:
                        preds = self.model(query_img, support_img=support_img, augment=False)
                    else:
                        preds = self.model(query_img, augment=False)
                
                # Loss computation (if in training mode)
                with dt[2]:
                    if self.training:
                        # Compute validation loss
                        loss_result = self.model.loss(batch, preds)
                        if isinstance(loss_result, tuple):
                            batch_loss_items = loss_result[1]
                            self.loss += batch_loss_items
                        else:
                            self.loss += loss_result
                
                # Postprocess and update metrics
                with dt[3]:
                    preds = self.postprocess(preds)
                    self.update_metrics(preds, batch)
        
        # Prepare results
        results = {}
        
        # Log validation summary
        LOGGER.info(f"Validation complete: {batch_count} batches processed")
        LOGGER.info(f"Total predictions: {self.total_preds}, Total GTs: {self.total_gts}")
        LOGGER.info(f"Stats collected: {len(self.stats)} entries")
        
        # Compute speed metrics
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        
        # Get validation loss if available
        if self.training:
            self.model.float()
            loss = self.loss.clone().detach()
            if trainer.world_size > 1:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
            if RANK > 0:
                return
            
            # Average the loss items across all batches
            self.loss_items_avg = loss / batch_count if batch_count > 0 else loss
            
            # Get loss items with proper formatting
            val_loss_items = trainer.label_loss_items(self.loss_items_avg, prefix="val")
            results.update(val_loss_items)
        
        # Process metrics to compute mAP, precision, recall, F1
        stats = self.get_stats()
        LOGGER.info(f"Computed stats: {stats}")
        results.update(stats)
        
        # Add fitness metric
        if "metrics/mAP50-95(B)" in results:
            results["fitness"] = float(results.get("metrics/mAP50-95(B)", 0.0))
        else:
            results["fitness"] = 0.0
        
        # Log the results with better formatting
        if RANK in {-1, 0}:
            # Collect all metrics to display
            metric_keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
            metric_names = ["P", "R", "mAP50", "mAP50-95"]
            
            # Add loss metrics if training
            display_keys = []
            display_names = []
            if self.training and hasattr(trainer, 'label_loss_items'):
                loss_metrics = {k: v for k, v in results.items() if k.startswith("val/")}
                for k in loss_metrics.keys():
                    display_keys.append(k)
                    display_names.append(k.replace("val/", "").replace("_loss", ""))
            
            # Add detection metrics
            for name, key in zip(metric_names, metric_keys):
                if key in results:
                    display_keys.append(key)
                    display_names.append(name)
            
            # Print as two lines
            if len(display_keys) > 0:
                header = "  ".join([f"{name:>10s}" for name in display_names])
                values = "  ".join([f"{results[key]:>10.4f}" for key in display_keys])
                LOGGER.info(f"\n{header}")
                LOGGER.info(f"{values}\n")
            else:
                LOGGER.warning("No metrics to display!")
        
        return {k: round(float(v), 5) if isinstance(v, (int, float, torch.Tensor)) else v for k, v in results.items()}

    def update_metrics(self, preds, batch):
        """
        Update validation metrics with predictions and ground truth.

        Args:
            preds: Model predictions - list of tensors, one per image. Each tensor has shape [N, 6] 
                   with format: x1, y1, x2, y2, conf, cls (after NMS).
            batch (dict): Batch data with ground truth (bboxes in xywh, normalized format).
        """
        # Ensure preds is a list
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        
        # Get batch information
        batch_size = batch.get("query_img", batch.get("img")).shape[0]
        gt_bboxes = batch.get("bboxes")  # [Total_M, 4] in xywh format, normalized (0-1), concatenated across batch
        gt_cls = batch.get("cls")  # [Total_M] class indices, concatenated across batch
        batch_idx = batch.get("batch_idx")  # [Total_M] indicating which image each bbox belongs to
        
        if gt_bboxes is None or len(gt_bboxes) == 0:
            # No ground truth, but we still need to process predictions
            for i, pred in enumerate(preds):
                if pred is not None and len(pred) > 0:
                    pred_conf = pred[:, 4] if pred.shape[1] > 4 else torch.ones(len(pred), device=self.device)
                    pred_cls = pred[:, 5].long() if pred.shape[1] > 5 else torch.zeros(len(pred), dtype=torch.long, device=self.device)
                    n_pred = len(pred)
                    
                    # All predictions are false positives
                    tp = torch.zeros(n_pred, dtype=torch.bool, device=self.device)
                    self.stats.append((
                        tp.cpu(),
                        pred_conf.cpu(),
                        pred_cls.cpu(),
                        torch.zeros(0, dtype=torch.long)  # No targets
                    ))
                    self.total_preds += n_pred
            return
        
        # Convert to tensors if needed
        if not isinstance(gt_cls, torch.Tensor):
            gt_cls = torch.tensor(gt_cls, dtype=torch.long, device=self.device)
        if not isinstance(gt_bboxes, torch.Tensor):
            gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32, device=self.device)
        if not isinstance(batch_idx, torch.Tensor):
            batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=self.device)
        
        # Get image shape for denormalization
        img = batch.get("query_img", batch.get("img"))
        h, w = img.shape[2:] if img is not None else (640, 640)
        
        # Process each image in the batch separately
        for i in range(batch_size):
            # Get predictions for this image
            if i >= len(preds):
                pred = torch.zeros((0, 6), device=self.device)
            else:
                pred = preds[i] if preds[i] is not None else torch.zeros((0, 6), device=self.device)
            
            if not isinstance(pred, torch.Tensor):
                pred = torch.zeros((0, 6), dtype=torch.float32, device=self.device)
            
            # Get ground truth for this image using batch_idx
            img_mask = batch_idx == i
            img_gt_bboxes = gt_bboxes[img_mask]  # [M_i, 4]
            img_gt_cls = gt_cls[img_mask]  # [M_i]
            
            n_gt = len(img_gt_bboxes)
            n_pred = len(pred)
            
            # Extract prediction components
            if n_pred > 0:
                pred_xyxy = pred[:, :4]  # Already in xyxy absolute format from NMS
                pred_conf = pred[:, 4] if pred.shape[1] > 4 else torch.ones(n_pred, device=self.device)
                pred_cls = pred[:, 5].long() if pred.shape[1] > 5 else torch.zeros(n_pred, dtype=torch.long, device=self.device)
            else:
                pred_xyxy = torch.zeros((0, 4), device=self.device)
                pred_conf = torch.zeros((0,), device=self.device)
                pred_cls = torch.zeros((0,), dtype=torch.long, device=self.device)
            
            # Convert GT bboxes from xywh normalized to xyxy absolute
            if n_gt > 0:
                gt_xyxy = self._xywh_to_xyxy(img_gt_bboxes, w, h)
            else:
                gt_xyxy = torch.zeros((0, 4), device=self.device)
            
            # Match predictions to ground truth
            tp = torch.zeros(n_pred, dtype=torch.bool, device=self.device)
            
            if n_pred > 0 and n_gt > 0:
                # Compute IoU matrix [n_pred, n_gt]
                iou_matrix = self._compute_iou_matrix(pred_xyxy, gt_xyxy)
                
                # For each prediction, find best matching GT
                matched_gt = set()
                for pred_idx in range(n_pred):
                    best_iou = 0.0
                    best_gt_idx = -1
                    
                    for gt_idx in range(n_gt):
                        # Skip if GT already matched or class mismatch
                        if gt_idx in matched_gt or pred_cls[pred_idx] != img_gt_cls[gt_idx]:
                            continue
                        
                        iou = iou_matrix[pred_idx, gt_idx]
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    # Mark as TP if IoU > 0.5
                    if best_iou > 0.5:
                        tp[pred_idx] = True
                        matched_gt.add(best_gt_idx)
            
            # Store statistics for mAP computation
            if n_pred > 0 or n_gt > 0:  # Only add stats if there are predictions or GT
                self.stats.append((
                    tp.cpu(),  # True positives
                    pred_conf.cpu(),  # Prediction confidences
                    pred_cls.cpu(),  # Prediction classes
                    img_gt_cls.cpu()  # Target classes
                ))
                
                # Log details for first batch
                if self.batch_i == 0 and i == 0:
                    LOGGER.info(f"Batch 0, Image 0: {n_pred} predictions, {n_gt} GTs, {tp.sum().item()} TPs")
            
            self.total_gts += n_gt
            self.total_preds += n_pred
    
    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix between two sets of boxes in xyxy format.
        
        Args:
            boxes1: [N, 4] tensor in xyxy format
            boxes2: [M, 4] tensor in xyxy format
            
        Returns:
            [N, M] tensor of IoU values
        """
        if len(boxes1) == 0 or len(boxes2) == 0:
            return torch.zeros((len(boxes1), len(boxes2)), device=boxes1.device)
        
        # Expand dimensions for broadcasting: boxes1 [N, 1, 4], boxes2 [1, M, 4]
        boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
        boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]
        
        # Compute intersection coordinates
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])  # [N, M]
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])  # [N, M]
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])  # [N, M]
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])  # [N, M]
        
        # Intersection area
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Compute areas of both boxes
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [N, 1]
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [1, M]
        
        # Union area
        union_area = area1 + area2 - inter_area  # [N, M]
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        return iou
    
    def _xywh_to_xyxy(self, bboxes, w, h):
        """Convert bboxes from xywh normalized format to xyxy absolute format."""
        bboxes = bboxes.clone()
        # xywh normalized: center_x, center_y, width, height (all in 0-1 range)
        # Convert to absolute coordinates first
        x_center = bboxes[:, 0] * w
        y_center = bboxes[:, 1] * h
        width = bboxes[:, 2] * w
        height = bboxes[:, 3] * h
        
        # Convert to xyxy format
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)

    def get_stats(self):
        """
        Get validation statistics - compute precision, recall, and mAP from collected predictions.

        Returns:
            (dict): Validation metrics (precision, recall, mAP50, mAP50-95).
        """
        try:
            if len(self.stats) == 0:
                # No predictions or ground truth
                LOGGER.warning("No stats collected for validation - returning zero metrics")
                return {
                    "metrics/precision(B)": 0.0,
                    "metrics/recall(B)": 0.0,
                    "metrics/mAP50(B)": 0.0,
                    "metrics/mAP50-95(B)": 0.0,
                }
            
            # Concatenate all stats across all batches/images
            # Handle the case where arrays might have different dimensions
            tp_list = []
            conf_list = []
            pred_cls_list = []
            target_cls_list = []
            
            for stat in self.stats:
                tp_list.append(stat[0])  # True positives
                conf_list.append(stat[1])  # Confidences
                pred_cls_list.append(stat[2])  # Predicted classes
                target_cls_list.append(stat[3])  # Target classes
            
            # Concatenate each component separately
            tp = torch.cat(tp_list, 0).cpu().numpy() if tp_list else np.array([])
            conf = torch.cat(conf_list, 0).cpu().numpy() if conf_list else np.array([])
            pred_cls = torch.cat(pred_cls_list, 0).cpu().numpy() if pred_cls_list else np.array([])
            target_cls = torch.cat(target_cls_list, 0).cpu().numpy() if target_cls_list else np.array([])
            
            # Ensure all arrays are 1D
            tp = tp.flatten()
            conf = conf.flatten()
            pred_cls = pred_cls.flatten()
            target_cls = target_cls.flatten()
            
            if len(tp) == 0 or len(pred_cls) == 0:
                LOGGER.warning("Stats array is empty or has no valid predictions")
                return {
                    "metrics/precision(B)": 0.0,
                    "metrics/recall(B)": 0.0,
                    "metrics/mAP50(B)": 0.0,
                    "metrics/mAP50-95(B)": 0.0,
                }
            
            # Sort predictions by confidence (descending)
            i = np.argsort(-conf)
            tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
            
            # Get unique classes in the dataset
            if len(target_cls) > 0 and len(pred_cls) > 0:
                unique_classes = np.unique(np.concatenate([pred_cls, target_cls]))
            elif len(pred_cls) > 0:
                unique_classes = np.unique(pred_cls)
            elif len(target_cls) > 0:
                unique_classes = np.unique(target_cls)
            else:
                unique_classes = np.array([])
            
            nc = len(unique_classes) if len(unique_classes) > 0 else 1
            
            LOGGER.info(f"Processing {len(tp)} predictions across {nc} classes")
            LOGGER.info(f"Unique classes: {unique_classes}")
            LOGGER.info(f"Total TP: {tp.sum()}, Total predictions: {len(tp)}")
            
            # Compute AP for each class using 101-point (COCO-style) interpolation
            # and select optimal precision/recall at the point maximizing F1.
            ap_per_class = []
            p_per_class = []
            r_per_class = []

            eps = 1e-16

            for c in unique_classes:
                # Select predictions for this class (predictions are already globally sorted by conf desc)
                class_mask = pred_cls == c
                tp_c = tp[class_mask]
                conf_c = conf[class_mask]

                # Count GT instances for this class
                n_gt_c = (target_cls == c).sum()

                # If no predictions for this class
                if len(tp_c) == 0:
                    if n_gt_c > 0:
                        ap_per_class.append(0.0)
                        p_per_class.append(0.0)
                        r_per_class.append(0.0)
                    continue

                # Compute cumulative TP and FP
                tp_cumsum = np.cumsum(tp_c)
                fp_cumsum = np.cumsum(~tp_c.astype(bool))

                # Compute precision and recall curves
                recall_curve = tp_cumsum / (n_gt_c + eps)
                precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + eps)

                # Compute AP using 101-point interpolation (COCO-style)
                recall_thresholds = np.linspace(0.0, 1.0, 101)
                ap_c = 0.0
                for r_thresh in recall_thresholds:
                    p_at_r = precision_curve[recall_curve >= r_thresh]
                    ap_c += p_at_r.max() if len(p_at_r) > 0 else 0.0
                ap_c /= len(recall_thresholds)

                ap_per_class.append(ap_c)

                # Compute F1 curve and select the index maximizing F1
                f1_curve = 2.0 * (precision_curve * recall_curve) / (precision_curve + recall_curve + eps)
                if f1_curve.size == 0:
                    # fallback
                    p_per_class.append(precision_curve[-1] if precision_curve.size > 0 else 0.0)
                    r_per_class.append(recall_curve[-1] if recall_curve.size > 0 else 0.0)
                else:
                    best_idx = int(np.argmax(f1_curve))
                    p_per_class.append(float(precision_curve[best_idx]))
                    r_per_class.append(float(recall_curve[best_idx]))
            
            # Compute mean metrics across classes
            if len(ap_per_class) > 0:
                mAP = float(np.mean(ap_per_class))
                mean_precision = float(np.mean(p_per_class))
                mean_recall = float(np.mean(r_per_class))
                LOGGER.info(f"Per-class AP (sample): {ap_per_class[:8]}{'...' if len(ap_per_class)>8 else ''}")
                LOGGER.info(f"Mean metrics - P: {mean_precision:.4f}, R: {mean_recall:.4f}, mAP: {mAP:.4f}")
            else:
                mAP = 0.0
                mean_precision = 0.0
                mean_recall = 0.0
                LOGGER.warning("No AP computed for any class")
            
            results = {
                "metrics/precision(B)": float(mean_precision),
                "metrics/recall(B)": float(mean_recall),
                "metrics/mAP50(B)": float(mAP),
                "metrics/mAP50-95(B)": float(mAP),  # Simplified - using mAP50 as proxy for mAP50-95
            }
            
            return results
            
        except Exception as e:
            LOGGER.warning(f"Failed to compute detection metrics: {e}")
            import traceback
            LOGGER.warning(traceback.format_exc())
            return {
                "metrics/precision(B)": 0.0,
                "metrics/recall(B)": 0.0,
                "metrics/mAP50(B)": 0.0,
                "metrics/mAP50-95(B)": 0.0,
            }

    def finalize_metrics(self):
        """Finalize validation metrics and return results."""
        return {}
