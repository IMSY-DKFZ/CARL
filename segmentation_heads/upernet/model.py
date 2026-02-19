# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0 
# found in the LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any
from einops import rearrange
from transformers.models.upernet.modeling_upernet import UperNetHead
from transformers.models.upernet.configuration_upernet import UperNetConfig

from carl.model.carl import CARLModel
from segmentation_heads.upernet.utils.vit_adapter import VitAdapter



class CARL_Adapter(CARLModel):
    """CARL model with Vision Transformer adapter for segmentation tasks.
    
    This model extends the CARLModel by adding a segmentation head that adapts
    the output of the spatial encoder for pixel-wise classification.
    
    Attributes:
        segmentation_head: A convolutional layer that maps spatial features to class logits.
    """
    def __init__(
            self, 
            spec_encoder_kwargs: Dict[str, Any] = None,
            spat_encoder_kwargs: Dict[str, Any] = None,
            patch_size: int = 8,
            vit_adapter_kwargs: Dict[str, Any] = {},
            upernet_kwargs: Dict[str, Any] = {},
            n_classes: int = 10,
            **kwargs) -> None:
        """Initialize the ViT adapter model.
        
        Args:
            n_classes: Number of segmentation classes.
            **kwargs: Additional keyword arguments for the CARLModel.
        """
        super().__init__(
            spec_encoder_kwargs,
            spat_encoder_kwargs,
            patch_size,
            **kwargs
        )
        
        vit_adapter_kwargs.update({
            "embed_dim": self.spatial_encoder.embed_dim,
            "patch_size": patch_size,
        })
        self.spatial_encoder = VitAdapter(
            spatial_backbone = self.spatial_encoder,
            **vit_adapter_kwargs
        )

        upernet_config = UperNetConfig(
            num_labels=n_classes,
            **upernet_kwargs
        )
        in_channels = self.spatial_encoder.spatial_backbone.embed_dim
        in_channels = [in_channels] * len(self.spatial_encoder.interaction_indexes)
        self.upernet_head = UperNetHead(
            upernet_config,
            in_channels=in_channels
        )

    def forward(
        self, 
        img: torch.Tensor, 
        wavelengths: torch.Tensor,
        cnn_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through CARL model.
        
        Args:
            img: Input image tensor of shape (B, C, H, W) where C is number of
                 spectral bands.
            wavelengths: Wavelength values tensor of shape (B, C).
            
        Returns:
            Tuple of:
                - spatial_representations: Spatial features of shape (B, D, H', W')
                - spectral_representations: Spectral features of shape (B, D, H', W')
                  where H'=H/patch_size, W'=W/patch_size, D is embedding dimension
        """
        batch_size, num_channels, height, width = img.shape
        
        # Reshape to process each spectral band separately
        # From (B, C, H, W) to (B*C, 1, H, W)
        img_reshaped = rearrange(
            img, 
            "b c h w -> (b c) 1 h w", 
            b=batch_size, 
            c=num_channels, 
            h=height, 
            w=width
        )

        # Embed patches: (B*C, 1, H, W) -> (B*C, D, H', W')
        patch_embeddings = self.embedder(img_reshaped)
        
        # Get spatial dimensions of patches
        patch_height, patch_width = patch_embeddings.shape[-2], patch_embeddings.shape[-1]
        embed_dim = patch_embeddings.shape[1]
        
        # Rearrange to sequence format for spectral encoder
        # (B*C, D, H', W') -> (B*H'*W', C, D)
        patch_seq = rearrange(
            patch_embeddings,
            "(b c) d nh nw -> (b nh nw) c d",
            b=batch_size,
            c=num_channels,
            nh=patch_height,
            nw=patch_width,
            d=embed_dim
        )

        # Spectral encoding: process wavelength information
        spectral_output = self.spectral_tf(patch_seq, wavelengths)
        spec_representations = spectral_output["queries"]
        
        # Aggregate spectral queries across wavelengths
        spec_representations = spec_representations.sum(dim=1)
        
        # Normalize spectral representations
        spec_representations = F.layer_norm(
            spec_representations,
            spec_representations.shape[-1:]
        )
        
        # Project spectral embeddings to spatial encoder dimension
        spec_representations = self.linear_connector(spec_representations)
        
        # Reshape back to spatial structure for spatial encoder
        # (B*H'*W', D_s) -> (B, H', W', D_s)
        spec_representations = rearrange(
            spec_representations,
            "(b nh nw) d -> b nh nw d",
            b=batch_size,
            nh=patch_height,
            nw=patch_width
        )

        # Spatial encoding: apply vision transformer
        # (B, H', W', D_s) -> (B, H'*W', D_s)
        multi_scale_features = self.spatial_encoder(
            spec_representations,
            cnn_image
        )
        
        prediction = self.upernet_head(multi_scale_features)

        return prediction