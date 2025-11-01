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
        Preprocess a batch of Siamese data (query and support images).

        Args:
            batch (dict): Batch dictionary containing 'query_img', 'support_img', and labels.

        Returns:
            (dict): Preprocessed batch with images normalized and moved to device.
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
        from pathlib import Path

        # Get images from batch
        query_imgs = batch.get("query_img", batch.get("img"))  # Fall back to img if query_img not present
        support_imgs = batch.get("support_img")

        if query_imgs is not None:
            # Prepare output directory
            output_dir = Path(self.save_dir) / "train_batch_plots"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get labels from batch
            bboxes = batch.get("bboxes")
            cls_labels = batch.get("cls")
            batch_idx = batch.get("batch_idx")
            
            # Ensure bboxes and cls are proper tensors, not None
            if bboxes is None:
                bboxes = torch.zeros((0, 4), dtype=torch.float32, device=query_imgs.device)
            if cls_labels is None:
                cls_labels = torch.zeros((0, 1), dtype=torch.float32, device=query_imgs.device)
            if batch_idx is None:
                batch_idx = torch.zeros(len(bboxes), dtype=torch.long, device=query_imgs.device)

            # Query batch: Has bounding boxes (what we're looking for)
            query_batch = {
                "img": query_imgs,
                "cls": cls_labels,
                "bboxes": bboxes,
                "batch_idx": batch_idx,
                "im_file": batch.get("im_file", []),
            }
            
            # Plot query images with bounding boxes
            try:
                plot_images(
                    labels=query_batch,
                    fname=str(output_dir / f"train_batch_query_{ni}.jpg"),
                    on_plot=self.on_plot,
                )
            except Exception as e:
                LOGGER.warning(f"Failed to plot query images: {e}")

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
                        on_plot=self.on_plot,
                    )
                except Exception as e:
                    LOGGER.warning(f"Failed to plot support images: {e}")

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
        Postprocess model predictions.

        Args:
            preds: Raw model predictions.

        Returns:
            Postprocessed predictions.
        """
        # This will be implemented based on specific postprocessing needs
        return preds

    def __call__(self, trainer=None, model=None):
        """
        Run inference on validation set with Siamese model support.
        
        Args:
            trainer: Training trainer instance.
            model: Model to validate.
            
        Returns:
            Validation metrics.
        """
        self.training = False
        self.device = next(trainer.model.parameters()).device
        self.model = trainer.model
        self.model.eval()

        # Reset metrics
        self.metrics.clear_stats()

        with torch.no_grad():
            for batch in self.dataloader:
                batch = self.preprocess_batch(batch)
                
                # Get predictions with Siamese support
                if "support_img" in batch and batch["support_img"] is not None:
                    preds = self.model(batch.get("query_img", batch["img"]), 
                                      support_img=batch["support_img"], 
                                      augment=False)
                else:
                    # Fallback to standard inference if support_img not available
                    preds = self.model(batch.get("query_img", batch["img"]), augment=False)
                
                preds = self.postprocess(preds)
                
        return {"fitness": 0.0}

    def update_metrics(self, preds, batch):
        """
        Update validation metrics with predictions and ground truth.

        Args:
            preds: Model predictions.
            batch (dict): Batch data with ground truth.
        """
        # Implement metric updates for Siamese detection
        pass

    def get_stats(self):
        """
        Get validation statistics.

        Returns:
            (dict): Validation metrics.
        """
        stats = {
            "fitness": 0.0,
        }
        return stats

    def finalize_metrics(self):
        """Finalize validation metrics and return results."""
        return {}
