# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0 
# found in the LICENSE file in the root directory of this source tree.

from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from carl.modules.spectral_encoder import SpectralEncoder
from carl.modules.spatial_encoder import TimmWrapper


class CARLModel(nn.Module):
    """Collaborative Attentive Representation Learning model.
    
    Combines spectral and spatial encoders for multispectral image analysis.
    The model:
    1. Embeds input images into patch representations
    2. Encodes spectral information across wavelengths
    3. Connects spectral embeddings to spatial domain
    4. Applies spatial transformer to capture spatial relationships
    
    Attributes:
        spectral_tf: Spectral encoder for wavelength-aware feature extraction
        spatial_encoder: Vision transformer for spatial feature learning
        embedder: Convolutional patch embedder
        linear_connector: Linear projection between spectral and spatial spaces
    """
    
    def __init__(
        self,
        spec_encoder_kwargs: Dict[str, Any] = {},
        spat_encoder_kwargs: Dict[str, Any] = {},
        patch_size: int = 8,
        **kwargs
    ) -> None:
        """Initialize CARL model.
        
        Args:
            spec_encoder_kwargs: Configuration for spectral encoder.
            spat_encoder_kwargs: Configuration for spatial encoder.
            patch_size: Size of patches for embedding (e.g., 8 means 8x8 patches).
            **kwargs: Additional keyword arguments (e.g., n_classes, not used here).
            
        Raises:
            ValueError: If encoder kwargs are invalid.
        """
        super().__init__()

        # Initialize spectral and spatial encoders
        self.spectral_tf = SpectralEncoder(**spec_encoder_kwargs)

        # Patch embedding: convert images to patch embeddings
        self.patch_size = patch_size
        self.embedder = nn.Conv2d(
            in_channels=1,
            out_channels=self.spectral_tf.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        if spat_encoder_kwargs != {}:
            self.spatial_encoder = TimmWrapper(**spat_encoder_kwargs)

            # Projection layer to align spectral and spatial embedding dimensions
            self.linear_connector = nn.Linear(
                self.spectral_tf.embed_dim,
                self.spatial_encoder.embed_dim,
            )

    def forward(
        self, 
        img: torch.Tensor, 
        wavelengths: torch.Tensor
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
        spatial_representations = self.spatial_encoder(spec_representations)
        
        # Reshape to spatial tensor format
        # (B, H'*W', D_s) -> (B, D_s, H', W')
        spatial_representations = rearrange(
            spatial_representations,
            "b (h w) d -> b d h w",
            b=batch_size,
            d=spatial_representations.shape[-1],
            h=patch_height,
            w=patch_width
        )

        return spatial_representations, spec_representations.permute(0, 3, 1, 2)
