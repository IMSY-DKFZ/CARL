"""Image classification trainer using CARL model with PyTorch Lightning.

This module implements a Lightning trainer for fine-tuning the CARL model's 
classification head for image classification tasks on spectral imagery.
"""

from typing import Tuple, Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from kornia import augmentation as K
from torch import Tensor

from carl.model.carl import CARLModel
from carl.config import ConfigAccessor
from carl.trainer.constants import DEFAULT_DROPOUT_PROB


class LinearTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for CARL image classification.
    
    This trainer:
    - Freezes the pre-trained CARL model parameters
    - Trains only a small classification head on top
    - Uses data augmentation for robustness
    - Logs accuracy as validation metric
    
    Attributes:
        model: Frozen pre-trained CARL model
        classifier: Trainable classification head
        accuracy: Accuracy metric for tracking
        loss_fun: Cross-entropy loss function
        transforms: Data augmentation pipeline
    """
    
    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        """Initialize the image classification trainer.
        
        Args:
            config: Configuration dictionary containing model_kwargs.
            *args: Additional positional arguments for LightningModule.
            **kwargs: Additional keyword arguments for LightningModule.
        """
        super().__init__(*args, **kwargs)

        # Use ConfigAccessor for safer configuration access
        self.config_accessor = ConfigAccessor(config)
        
        model_kwargs = self.config_accessor.get("model_kwargs", {})
        self.n_classes = model_kwargs["n_classes"]

        # Load pre-trained CARL model and freeze its parameters
        self.model = CARLModel(**model_kwargs)
        self._freeze_model_parameters()

        # Create trainable classification head
        # Global average pooling followed by linear layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(
            self.model.spatial_encoder.embed_dim,
            self.n_classes
        )

        # Metrics for validation
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.n_classes
        )
        if self.config_accessor.get("training_kwargs/multi_label", False):
            self.accuracy = torchmetrics.classification.MultilabelAveragePrecision(
                num_labels=self.n_classes,
                average="micro"
            )
            self.loss_fun = torch.nn.MultiLabelSoftMarginLoss()
        else:
            self.accuracy = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=self.n_classes
            )
            self.loss_fun = nn.CrossEntropyLoss(ignore_index=-1)

        # Data augmentation pipeline
        self.transforms = self._build_augmentation_pipeline()

    def _freeze_model_parameters(self) -> None:
        """Freeze all CARL model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_augmentation_pipeline(self) -> K.AugmentationSequential:
        """Build data augmentation pipeline for classification.
        
        Returns:
            Augmentation pipeline with resize and flip operations.
        """
        img_size = self.config_accessor.get("model_kwargs/image_size")
        transforms = K.AugmentationSequential(
            K.Resize(
                (img_size, img_size),
                p=1.0,
            ),
            K.RandomHorizontalFlip(p=DEFAULT_DROPOUT_PROB),
            K.RandomVerticalFlip(p=DEFAULT_DROPOUT_PROB),
            data_keys=["input"],
            same_on_batch=False,
        )
        
        # Set device and dtype for all augmentations
        for idx in range(len(transforms)):
            transforms[idx].set_rng_device_and_dtype(
                device="cuda", 
                dtype=torch.float64
            )
        
        return transforms

    def training_step(
        self, 
        batch: Tuple[Tensor, Tensor, Tensor], 
        batch_idx: int
    ) -> Tensor:
        """Training step for one batch.
        
        Args:
            batch: Tuple containing:
                - img: Input image tensor of shape (B, C, H, W)
                - wavelengths: Wavelength tensor of shape (B, N)
                - labels: Target class labels of shape (B,)
            batch_idx: Index of the current batch.
            
        Returns:
            Computed loss value.
        """
        img, wavelengths, labels = batch
        
        # Ensure correct dtype
        img = img.to(dtype=torch.float32)
        wavelengths = wavelengths.to(dtype=torch.float32)
        labels = labels.long()
        
        # Apply augmentations
        img = self.transforms(img)

        # Forward pass through CARL model
        spat_feat, spec_feat = self.model(img, wavelengths)
        
        # Global average pooling
        pooled_feat = self.global_avg_pool(spat_feat).squeeze(-1).squeeze(-1)
        
        # Classification
        predictions = self.classifier(pooled_feat)

        # Compute loss
        loss = self.loss_fun(predictions, labels)
        self.log('train/loss', loss, on_epoch=True, on_step=True)
        
        return loss
    
    def validation_step(
        self, 
        batch: Tuple[Tensor, Tensor, Tensor], 
        batch_idx: int
    ) -> None:
        """Validation step for one batch.
        
        Args:
            batch: Tuple containing:
                - img: Input image tensor of shape (B, C, H, W)
                - wavelengths: Wavelength tensor of shape (B, N)
                - labels: Target class labels of shape (B,)
            batch_idx: Index of the current batch.
        """
        img, wavelengths, labels = batch
        
        # Ensure correct dtype
        img = img.to(dtype=torch.float32)
        wavelengths = wavelengths.to(dtype=torch.float32)
        labels = labels.long()

        # Resize image to standard size
        img_size = self.config_accessor.get("model_kwargs/image_size")
        img = F.interpolate(
            img,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False
        )

        # Forward pass through CARL model
        spat_feat, spec_feat = self.model(img, wavelengths)
        
        # Global average pooling
        pooled_feat = self.global_avg_pool(spat_feat).squeeze(-1).squeeze(-1)
        
        # Classification
        predictions = self.classifier(pooled_feat)

        # Update metrics
        self.accuracy.update(predictions, labels)

    def on_validation_epoch_end(self) -> None:
        """Compute and log metrics at the end of validation epoch."""
        acc = self.accuracy.compute()
        self.log("val_accuracy", acc, on_epoch=True)
        self.accuracy.reset()

    def configure_optimizers(self) -> Tuple[list, list]:
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            Tuple of (optimizers, schedulers) for PyTorch Lightning.
        """
        params = self.classifier.parameters()
        lr = float(
            self.config_accessor.get("training_kwargs/learning_rate")
        )
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            fused=True,
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )

        return [optimizer], [lr_scheduler]