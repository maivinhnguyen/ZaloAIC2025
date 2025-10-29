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
            loss_dict = dict(zip(keys, loss_items))
        else:
            loss_dict = {k: v.item() if v is not None else 0 for k, v in zip(keys, self.tloss)}
        return loss_dict

    def progress_string(self):
        """Return a formatted training progress string."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))).format(
            "Epoch", "GPU_mem", *self.loss_names, "Instances", "Size"
        )


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
