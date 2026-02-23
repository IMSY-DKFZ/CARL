# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0 
# found in the LICENSE file in the root directory of this source tree.


import copy
import logging
from typing import Tuple, Dict, Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from kornia import augmentation as K
from torch import Tensor

from carl.model.carl_ssl import CARLSSLModel
from carl.modules.ssl_modules.spectral_masking import SpectralMasking
from carl.modules.ssl_modules.spatial_masking import SpatialMasking
from carl.modules.utils.ssl_utils import apply_masks, repeat_interleave_batch
from carl.config import ConfigAccessor
from carl.trainer.constants import (
    DEFAULT_DROPOUT_PROB,
    EPSILON,
    KNN_K_NEIGHBORS,
    KNN_TEMPERATURE
)
from carl.trainer.ssl_utils.ssl_validator import KNNValidator
from carl.trainer.ssl_utils.ssl_loss import VICRegLoss


class Trainer(pl.LightningModule):
    """PyTorch Lightning trainer for CARL self-supervised learning.
    
    This trainer implements a joint embedding predictive architecture (JEPA) style
    self-supervised learning approach with:
    - Student and teacher networks (teacher updated via EMA)
    - Spatial and spectral masking strategies
    - VICReg-style loss for representation learning
    - KNN validation for monitoring representation quality
    
    Attributes:
        model: Student CARL SSL model with spatial/spectral encoders and predictors
        teacher: Teacher CARL SSL model (EMA updated, no predictors)
        spatial_masking: Module for generating spatial masks
        spectral_masking: Module for generating spectral masks
        loss_fun: VICReg-style loss function
        transforms: Data augmentation pipeline
    """
    
    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        """Initialize the semantic segmentation trainer.
        
        Args:
            config: Configuration dictionary containing model_kwargs.
            *args: Additional positional arguments for LightningModule.
            **kwargs: Additional keyword arguments for LightningModule.
        """
        super().__init__(*args, **kwargs)

        # Use ConfigAccessor for safer configuration access
        self.config_accessor = ConfigAccessor(config)
        
        model_kwargs = self.config_accessor.get("model_kwargs", {})

        # Load pre-trained CARL model and freeze its parameters
        self.model = CARLSSLModel(**model_kwargs)
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()

        # Remove predictors from teacher model (not needed for inference)
        self._remove_teacher_predictors()
        self._freeze_model_parameters(self.teacher)

        self._teacher_pairs = self._init_teacher_param_pairs()

        spatial_loss, spectral_loss = self.init_loss_functions()
        self.spatial_loss = spatial_loss
        self.spectral_loss = spectral_loss

        # Setup data augmentation pipeline
        self.transforms = self._build_augmentation_pipeline()

        # Initialize masking modules
        spectral_masking_kwargs = self.config_accessor.get("spectral_masking_kwargs", {})
        spatial_masking_kwargs = self.config_accessor.get("spatial_masking_kwargs", {})
        spatial_masking_kwargs["input_size"] = self.config_accessor.get("model_kwargs/image_size")
        spatial_masking_kwargs["patch_size"] = self.config_accessor.get("model_kwargs/patch_size")
        self.spectral_masking = SpectralMasking(**spectral_masking_kwargs)
        self.spatial_masking = SpatialMasking(**spatial_masking_kwargs)

        # Initialize KNN validator for validation
        num_classes = self.config_accessor.get("model_kwargs/n_classes")
        self.knn_validator = KNNValidator(
            num_classes=num_classes,
            k=KNN_K_NEIGHBORS,
            temperature=KNN_TEMPERATURE,
        )
        self.accuracies = []
        
        self.momentum_scheduler = None

    def _remove_teacher_predictors(self) -> None:
        """Remove predictor modules from teacher model."""
        if hasattr(self.teacher, "spectral_predictor"):
            del self.teacher.spectral_predictor
        if hasattr(self.teacher, "spatial_predictor"):
            del self.teacher.spatial_predictor
        if hasattr(self.teacher, "predictor"):
            del self.teacher.predictor
    
    def _freeze_model_parameters(self, module: nn.Module, freeze=True) -> None:
        """Freeze all model parameters.
        
        Args:
            module: The module to freeze.
        """
        for param in module.parameters():
            param.requires_grad = not freeze

    def _build_augmentation_pipeline(self) -> K.AugmentationSequential:
        """Build data augmentation pipeline.
        
        Returns:
            Augmentation pipeline with crop and flip operations.
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
                dtype=torch.float32
            )
        
        return transforms

    def training_step(
        self, 
        batch: Tuple[Tensor, Tensor], 
        batch_idx: int
    ) -> Tensor:
        """Training step for one batch.
        
        Args:
            batch: Tuple containing:
                - img: Input image tensor of shape (B, C, H, W)
                - wavelengths: Wavelength tensor of shape (B, N)
                - labels: Target labels of shape (B, H, W)
            batch_idx: Index of the current batch.
            
        Returns:
            Computed loss value.
        """
        if self.momentum_scheduler is None:
            self.init_momentum_scheduler()
        img, wavelengths = batch

        # Ensure correct dtype
        img = img.to(dtype=torch.float32)
        wavelengths = wavelengths.to(dtype=torch.float32)

        # Apply augmentations
        img = self.transforms(img)

        collated_masks_spat_enc, collated_masks_spat_pred = self.spatial_masking(
            img.size(0), device="cpu"
        )
        collated_masks_spat_enc = [mask.to(img.device) for mask in collated_masks_spat_enc]
        collated_masks_spat_pred = [mask.to(img.device) for mask in collated_masks_spat_pred]
        
        # Calculate spectral batch size
        image_size = self.config_accessor.get("model_kwargs/image_size", 224)
        spectral_batch_size = int(
            img.size(0) * (image_size // self.model.patch_size) ** 2
        )
        collated_masks_spec_enc, collated_masks_spec_pred = self.spectral_masking(
            spectral_batch_size, img.size(1), device="cpu"
        )
        collated_masks_spec_enc = [mask.to(img.device) for mask in collated_masks_spec_enc]
        collated_masks_spec_pred = [mask.to(img.device) for mask in collated_masks_spec_pred]

        (
            pred_token_spat,
            pred_token_spec,
            spec_idx_optim,
            collated_masks_spec_pred,
        ) = self.model.forward_student(
            img,
            wavelengths,
            collated_masks_spat_enc,
            collated_masks_spec_enc,
            collated_masks_spat_pred,
            collated_masks_spec_pred,
        )

        with torch.no_grad():
            spat_token, spec_token = self.teacher.forward_teacher(img, wavelengths)
            spat_token = F.layer_norm(spat_token, (spat_token.size(-1),))  # normalize over feature-dim
            spec_token = F.layer_norm(spec_token, (spec_token.size(-1),))  # normalize over feature-dim
            # # -- create targets (masked regions of h)
            spat_token_masked = apply_masks(spat_token, collated_masks_spat_pred)
            spat_teacher_token = repeat_interleave_batch(
                spat_token_masked, len(spat_token), repeat=len(collated_masks_spat_enc)
            )

            nh = int(spat_token.shape[1] ** 0.5)
            c, d = spec_token.shape[-2:]
            # Optimize spectral token processing - combine operations
            spec_token_masked = spec_token.view(len(img), nh * nh, c * d)
            spec_token_masked = apply_masks(spec_token_masked, collated_masks_spat_enc)
            spec_token_masked = spec_token_masked.view(-1, c, d)

            spec_token_masked = spec_token_masked[spec_idx_optim]
            spec_token_masked = apply_masks(spec_token_masked, collated_masks_spec_pred)


        spat_loss, spat_loss_inv, spat_pstd_z, spat_cov = self.spatial_loss(
            pred_token=pred_token_spat,
            teacher_token=spat_teacher_token,
        )

        spec_loss, spec_loss_inv, spec_pstd_z, spec_cov = self.spectral_loss(    
            pred_token=pred_token_spec,
            teacher_token=spec_token_masked,
        )

        # Combine spatial and spectral losses with coefficients
        spatial_coeff = self.config_accessor.get("loss_coeff/spatial_coeff/loss_weight", 1.0)
        spectral_coeff = self.config_accessor.get("loss_coeff/spectral_coeff/loss_weight", 1.0)
        
        loss = spatial_coeff * spat_loss + spectral_coeff * spec_loss
        weights = spatial_coeff + spectral_coeff
        loss = loss / weights

        assert not torch.isnan(loss), f"Loss is NaN: {loss.item()}"

        self.log("train/loss", loss, on_step=True)
        self.log("train/spat_loss", spat_loss, on_step=True, on_epoch=True)
        self.log("train/spec_loss", spec_loss, on_step=True, on_epoch=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.update_teacher()

    def on_validation_epoch_start(self) -> None:
        """Build feature bank from validation set for KNN evaluation."""
        self.knn_validator.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        """Perform KNN validation on a batch.
        
        Args:
            batch: Tuple of (features, labels, wavelengths).
            batch_idx: Index of the current batch.
            
        Returns:
            Validation accuracy for this batch.
        """

        if dataloader_idx == 0:
            image_size = self.config_accessor.get("model_kwargs/image_size", 224)
            
            self.knn_validator.add_to_feature_bank(
                batch,
                image_size=image_size,
                model=self.model,
            )
            return

        img, wavelengths, labels = batch
        image_size = self.config_accessor.get("model_kwargs/image_size", 224)
        
        with torch.no_grad():
            size = (image_size, image_size)
            if img.size(2) != size[0] or img.size(3) != size[1]:
                img = F.interpolate(
                    img, size=size, mode="bilinear", align_corners=False
                )
            
            # Extract and normalize spatial tokens
            spatial_tokens, spectral_tokens = self.model.forward(img, wavelengths)
            features = spatial_tokens.flatten(2).mean(-1)  # (B, D)

            # Perform KNN validation
            accuracy = self.knn_validator.validate(features, labels)

        if batch_idx == 0:
            grid = self.plot(img, spatial_tokens, spectral_tokens)
            self.logger.experiment.add_image(
                "val/knn_visualizations",
                grid,
                global_step=self.global_step,
            )
        
        self.accuracies.append(accuracy)
        return accuracy

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        """Clean up feature bank after validation epoch."""
        m_acc = sum(self.accuracies) / len(self.accuracies) if len(self.accuracies) > 0 else 0.0
        self.log("val_knn_acc", m_acc)
        self.accuracies = []
        self.knn_validator.clear()

    def init_loss_functions(self):
        inv_coeff = self.config_accessor.get(f"loss_coeff/spectral_coeff/inv_coeff", 1.0)
        var_coeff = self.config_accessor.get(f"loss_coeff/spectral_coeff/var_coeff", 1.0)
        cov_coeff = self.config_accessor.get(f"loss_coeff/spectral_coeff/cov_coeff", 1.0)
        var_param = self.config_accessor.get(f"loss_coeff/spectral_coeff/var_param", 1.0)
        spectral_loss = VICRegLoss(
            sim_coeff=inv_coeff,
            var_coeff=var_coeff,
            cov_coeff=cov_coeff,
            var_param=var_param,
            eps=EPSILON,
        )

        inv_coeff = self.config_accessor.get(f"loss_coeff/spatial_coeff/inv_coeff", 1.0)
        var_coeff = self.config_accessor.get(f"loss_coeff/spatial_coeff/var_coeff", 1.0)
        cov_coeff = self.config_accessor.get(f"loss_coeff/spatial_coeff/cov_coeff", 1.0)
        var_param = self.config_accessor.get(f"loss_coeff/spatial_coeff/var_param", 1.0)
        spatial_loss = VICRegLoss(
            sim_coeff=inv_coeff,
            var_coeff=var_coeff,
            cov_coeff=cov_coeff,
            var_param=var_param,
            eps=EPSILON,
        )

        return spatial_loss, spectral_loss

    def _init_teacher_param_pairs(self):
        # Call once after both models are created/loaded
        teacher_named = dict(self.teacher.named_parameters())
        teacher_pairs = [
            (s_param, teacher_named[name])
            for name, s_param in self.model.named_parameters()
            if name in teacher_named
        ]

        return teacher_pairs

    @torch.no_grad()
    def update_teacher(self):
        m = next(self.momentum_scheduler)
        for s_param, t_param in self._teacher_pairs:
            t_param.mul_(m).add_(s_param, alpha=1.0 - m)

    def init_momentum_scheduler(self) -> Iterator[float]:
        """Initialize the momentum scheduler for teacher updates."""
        train_loader = self.trainer.train_dataloader
        iterations_per_epoch = len(train_loader)
        num_epochs = self.trainer.max_epochs
        
        # Get EMA momentum values from config
        ema_values = self.config_accessor.get("ema", [0.996, 1.0])
        base_momentum, final_momentum = ema_values[0], ema_values[1]
        total_iterations = int(iterations_per_epoch * num_epochs)
    
        self.momentum_scheduler = \
            (
            base_momentum + i * (final_momentum - base_momentum) / total_iterations
            for i in range(total_iterations + 1)
        )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration.
        """
        params = self.model.parameters()
        lr = float(
            self.config_accessor.get("training_kwargs/learning_rate")
        )
        weight_decay = float(
            self.config_accessor.get("training_kwargs/weight_decay")
        )
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            fused=True,
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=float(
                self.config_accessor.get("training_kwargs/min_lr")
            ),
        )

        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                },
        }

    def plot(
            self,
            imgs,
            spatial_tokens,
            spectral_tokens
        ):
        
        img_stack = []
        for idx in range(len(imgs)):
            
            img = imgs[idx]
            spat_representations = spatial_tokens[idx]
            spec_representations = spectral_tokens[idx]
            
            d,h,w = spat_representations.shape
            spat_representations = spat_representations.permute(1,2,0).flatten(0,1)
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                u,s,v = torch.pca_lowrank(spat_representations, q=3)
            s = torch.diag(s)
            pca = torch.matmul(u, s)
            pca = pca.reshape(h, w, 3).permute(2,0,1).unsqueeze(0)
            pca = F.interpolate(pca, size=img.shape[1:], mode='bilinear', align_corners=False)
            pca = pca[0].permute(1,2,0).cpu().numpy()

            d,h,w = spec_representations.shape
            spec_representations = spec_representations.permute(1,2,0).flatten(0,1)
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                u,s,v = torch.pca_lowrank(spec_representations, q=3)
            s = torch.diag(s)
            pca_spec = torch.matmul(u, s)
            pca_spec = pca_spec.reshape(h, w, 3).permute(2,0,1).unsqueeze(0)
            pca_spec = F.interpolate(pca_spec, size=img.shape[1:], mode='bilinear', align_corners=False)
            pca_spec = pca_spec[0].permute(1,2,0).cpu().numpy()

            fig, ax = plt.subplots(1,3, figsize=(15,6))

            img = img.permute(1,2,0).cpu().numpy()
            rgb_channels = np.linspace(0, img.shape[-1]-1, 3)
            rgb_channels = rgb_channels.astype(int).tolist()
            img = img[..., rgb_channels]
            img = (img - img.min()) / (img.max() - img.min())
            ax[0].imshow(img)
            ax[0].set_title("Input Image")

            feats = (pca - pca.min()) / (pca.max() - pca.min())
            ax[1].imshow(feats)
            ax[1].set_title("Spatial Features")

            feats_spec = (pca_spec - pca_spec.min()) / (pca_spec.max() - pca_spec.min())
            ax[2].imshow(feats_spec)
            ax[2].set_title("Spectral Features")

            for a in ax:
                a.axis('off')

            # Convert to array
            fig.canvas.draw()
            rgba_buf = fig.canvas.buffer_rgba()
            w, h = fig.canvas.get_width_height()
            rgb_array = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))[..., :3]
            plt.close(fig)
            rgb_array = torch.from_numpy(rgb_array).permute(2, 0, 1).float() / 255.
            img_stack.append(rgb_array)

        img_stack = torch.stack(img_stack)
        grid = torchvision.utils.make_grid(img_stack, nrow=4)
        return grid

        

        