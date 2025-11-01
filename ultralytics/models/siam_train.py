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

    def visualize_training_samples(self, batch, output_dir=None, epoch=0, max_samples=5):
        """
        Visualize a limited number of training samples (query and support images) and save to output folder.

        Args:
            batch (dict): Batch dictionary containing 'query_img', 'support_img', and labels.
            output_dir (str): Directory to save visualizations. Defaults to self.save_dir.
            epoch (int): Current epoch number (used for naming files).
            max_samples (int): Maximum number of samples to visualize.
        """
        import os
        import matplotlib.pyplot as plt

        # Use the training save directory if output_dir is not provided
        output_dir = output_dir or os.path.join(self.save_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        LOGGER.info(f"Saving visualizations to {output_dir}")

        query_imgs = batch.get("query_img")
        support_imgs = batch.get("support_img")

        if query_imgs is not None and support_imgs is not None:
            for i in range(min(len(query_imgs), max_samples)):
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                # Plot query image
                axes[0].imshow(query_imgs[i].cpu().numpy().transpose(1, 2, 0))
                axes[0].set_title("Query Image")
                axes[0].axis("off")

                # Plot support image
                axes[1].imshow(support_imgs[i].cpu().numpy().transpose(1, 2, 0))
                axes[1].set_title("Support Image")
                axes[1].axis("off")

                # Save the figure
                save_path = os.path.join(output_dir, f"epoch_{epoch}_sample_{i}.png")
                plt.savefig(save_path)
                plt.close(fig)
                LOGGER.info(f"Saved visualization: {save_path}")

    def train_epoch(self, epoch):
        """
        Train for a single epoch and log metrics with labels.

        Args:
            epoch (int): Current epoch number.
        """
        # Visualize training samples only at the beginning of training (epoch 0)
        if epoch == 0:
            batch = next(iter(self.train_loader))  # Example: Get a batch from the dataloader
            LOGGER.info("Visualizing training samples for epoch 0")
            self.visualize_training_samples(batch, epoch=epoch)

        # Example metrics dictionary (replace with actual metrics from training loop)
        metrics = {
            "iou_loss": self.tloss[0],
            "bce_loss": self.tloss[1],
            "rpl_loss": self.tloss[2],
            "dice_loss": self.tloss[3],
            "dfl_loss": self.tloss[4],
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }

        # Log metrics at the end of the epoch
        LOGGER.info(f"Logging metrics for epoch {epoch}")
        self.log_metrics(epoch, metrics)
        LOGGER.info(f"Metrics logged for epoch {epoch}")


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
