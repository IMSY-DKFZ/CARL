# Copyright (c) Shanghai AI Lab. All rights reserved.
# 
# This source code is licensed under the Apache License, Version 2.0.
# 
# References:
#   https://github.com/czczup/ViT-Adapter/blob/main/segmentation/mmseg_custom/models/backbones/vit_adapter.py


import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_
from torch.nn.init import normal_

from segmentation_heads.upernet.utils.layers import (
    InteractionBlock,
    SpatialPriorModule,
    deform_inputs,
)
from segmentation_heads.upernet.utils.ops.modules import MSDeformAttn
from carl.modules.spatial_encoder import TimmWrapper


class VitAdapter(nn.Module):
    """
    Vision Transformer Adapter with multi-scale deformable attention.

    This module integrates a spatial Vision Transformer backbone with a Spatial Prior Module (SPM)
    and interaction blocks to extract multi-scale features. It uses deformable attention mechanisms
    to efficiently attend to multi-resolution features.

    Args:
        spatial_backbone: The Vision Transformer backbone model
        embed_dim (int): Embedding dimension. Default: 384
        conv_inplane (int): Number of input planes for convolutional layers. Default: 64
        n_points (int): Number of reference points for deformable attention. Default: 4
        deform_num_heads (int): Number of attention heads for deformable layers. Default: 6
        init_values (float): Initial values for learnable scale parameters. Default: 0.0
        interaction_indexes (List[List[int]]): Indices of transformer blocks to interact with each level.
            Default: [[0, 2], [3, 5], [6, 8], [9, 11]]
        with_cffn (bool): Whether to use convolutional feed-forward networks. Default: True
        cffn_ratio (float): Ratio for hidden features in CFFN. Default: 0.25
        deform_ratio (float): Scaling ratio for deformable attention. Default: 1.0
        add_vit_feature (bool): Whether to add ViT features to output. Default: True
        use_extra_extractor (bool): Whether to use additional extractors. Default: True
        with_cp (bool): Whether to use gradient checkpointing. Default: False
        n_levels (int): Number of pyramid levels (3 or 4). Default: 4
        in_channels (int): Number of input channels (e.g., 3 for RGB). Default: 3
        patch_size (int): Vision Transformer patch size. Default: 14
        high_res (bool): Whether to use high-resolution (4x instead of 8x downsampling). Default: False
        track_running_stats (bool): Whether batch norm layers track running statistics. Default: True
    """

    def __init__(
        self,
        spatial_backbone: TimmWrapper,
        embed_dim: int = 384,
        conv_inplane: int = 64,
        n_points: int = 4,
        deform_num_heads: int = 6,
        init_values: float = 0.0,
        interaction_indexes: Optional[List[List[int]]] = None,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        deform_ratio: float = 1.0,
        add_vit_feature: bool = True,
        use_extra_extractor: bool = True,
        with_cp: bool = False,
        in_channels: int = 3,
        patch_size: int = None,
        n_levels: int = 4,
        high_res: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__()

        if interaction_indexes is None:
            n_blocks = len(spatial_backbone.net.blocks)
            interaction_len = n_blocks // 4
            interaction_indexes = [[i * interaction_len, i * interaction_len + interaction_len - 1] for i in range(4)]
        assert len(interaction_indexes) == 4, "interaction_indexes must contain 4 blocks"

        self.n_levels = n_levels
        self.interaction_indexes = interaction_indexes
        self.spatial_backbone = spatial_backbone
        self.add_vit_feature = add_vit_feature
        self.high_res = high_res
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Level embeddings for different feature scales
        self.level_embed = nn.Parameter(torch.zeros(4, embed_dim))

        # Spatial Prior Module to extract initial features from RGB
        self.spm = SpatialPriorModule(
            in_channels=in_channels,
            inplanes=conv_inplane,
            embed_dim=embed_dim,
        )

        # Build interaction blocks for each pyramid level
        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=(
                        (True if i == len(interaction_indexes) - 1 else False)
                        and use_extra_extractor
                    ),
                    with_cp=with_cp,
                    n_levels=n_levels,
                )
                for i in range(len(interaction_indexes))
            ]
        )

        # Batch normalization layers for final feature normalization
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        # Initialize weights
        self.apply(self._init_weights)
        normal_(self.level_embed)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Initialize module weights using appropriate strategies.

        Args:
            m (nn.Module): Module to initialize weights for
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add learnable level embeddings to features from different pyramid levels.

        Args:
            c1: Features from level 1
            c2: Features from level 2
            c3: Features from level 3
            c4: Features from level 4

        Returns:
            Tuple of features with added level embeddings
        """
        if self.n_levels == 4:
            c1 = c1 + self.level_embed[0]
        c2 = c2 + self.level_embed[1]
        c3 = c3 + self.level_embed[2]
        c4 = c4 + self.level_embed[3]
        return c1, c2, c3, c4

    def _concatenate_features(
        self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate multi-scale features based on pyramid level configuration.

        Args:
            c1-c4: Feature tensors from different pyramid levels

        Returns:
            Concatenated feature tensor
        """
        if self.n_levels == 4:
            return torch.cat([c1, c2, c3, c4], dim=1)
        
        if self.high_res:
            return torch.cat([c1, c2, c3], dim=1)
        
        return torch.cat([c2, c3, c4], dim=1)

    def _split_concatenated_features(
        self, c: torch.Tensor, c1_size: int, c2_size: int, c3_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split concatenated features back into individual pyramid levels.

        Args:
            c: Concatenated feature tensor
            c1_size: Size of level 1 features
            c2_size: Size of level 2 features
            c3_size: Size of level 3 features

        Returns:
            Tuple of split feature tensors (c1, c2, c3, c4)
        """
        if self.n_levels == 4:
            c1 = c[:, 0:c1_size, :]
            c2 = c[:, c1_size : c1_size + c2_size, :]
            c3 = c[:, c1_size + c2_size : c1_size + c2_size + c3_size, :]
            c4 = c[:, c1_size + c2_size + c3_size :, :]
        else:
            if self.high_res:
                c1 = c[:, 0:c1_size, :]
                c2 = c[:, c1_size : c1_size + c2_size, :]
                c3 = c[:, c1_size + c2_size :, :]
                c4 = None
            else:
                c1 = None
                c2 = c[:, 0:c2_size, :]
                c3 = c[:, c2_size : c2_size + c3_size, :]
                c4 = c[:, c2_size + c3_size :, :]

        return c1, c2, c3, c4

    def _reshape_features_to_spatial(
        self,
        c1: Optional[torch.Tensor],
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: Optional[torch.Tensor],
        deform_inputs1: List,
        bs: int,
        dim: int,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Reshape feature tensors from sequence format to spatial format.

        Args:
            c1-c4: Feature tensors (may be None if not used)
            deform_inputs1: Deformation input containing spatial shape information
            bs: Batch size
            dim: Feature dimension

        Returns:
            Tuple of reshaped spatial feature tensors
        """
        spatial_dims = deform_inputs1[1]

        if self.n_levels == 4:
            # Reshape all 4 levels
            a, b = spatial_dims[0]
            c1 = c1.transpose(1, 2).view(bs, dim, a, b).contiguous()
            a, b = spatial_dims[1]
            c2 = c2.transpose(1, 2).view(bs, dim, a, b).contiguous()
            a, b = spatial_dims[2]
            c3 = c3.transpose(1, 2).view(bs, dim, a, b).contiguous()
            a, b = spatial_dims[3]
            c4 = c4.transpose(1, 2).view(bs, dim, a, b).contiguous()
        else:
            if self.high_res:
                # High res: reshape c1, c2, c3 and interpolate c4
                a, b = spatial_dims[0]
                c1 = c1.transpose(1, 2).view(bs, dim, a, b).contiguous()
                a, b = spatial_dims[1]
                c2 = c2.transpose(1, 2).view(bs, dim, a, b).contiguous()
                a, b = spatial_dims[2]
                c3 = c3.transpose(1, 2).view(bs, dim, a, b).contiguous()
                # c4 is in sequence form, reshape it
                n = int(math.sqrt(c4.shape[1]))
                c4 = c4.permute(0, 2, 1).view(bs, dim, n, n).contiguous()
                c4 = F.interpolate(c3, size=(n, n)) + c4
            else:
                # Low res: reshape c2, c3, c4 and interpolate c1
                a, b = spatial_dims[0]
                c2 = c2.transpose(1, 2).view(bs, dim, a, b).contiguous()
                a, b = spatial_dims[1]
                c3 = c3.transpose(1, 2).view(bs, dim, a, b).contiguous()
                a, b = spatial_dims[2]
                c4 = c4.transpose(1, 2).view(bs, dim, a, b).contiguous()
                # c1 is in sequence form, reshape it
                n = int(math.sqrt(c1.shape[1]))
                c1 = c1.permute(0, 2, 1).view(bs, dim, n, n).contiguous()
                c1 = F.interpolate(c2, size=(n, n)) + c1

        return c1, c2, c3, c4

    def _fuse_vit_features(
        self,
        c1: Optional[torch.Tensor],
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: Optional[torch.Tensor],
        vit_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optionally fuse ViT features with deformed CNN features.

        Args:
            c1-c4: CNN features from multi-scale pyramid
            vit_features: List of ViT features for each level

        Returns:
            Tuple of fused features
        """
        if not self.add_vit_feature:
            return c1, c2, c3, c4

        x1, x2, x3, x4 = vit_features
        x1 = F.interpolate(x1, size=(c1.size(-2), c1.size(-1)), mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=(c2.size(-2), c2.size(-1)), mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=(c3.size(-2), c3.size(-1)), mode="bilinear", align_corners=False)
        x4 = F.interpolate(x4, size=(c4.size(-2), c4.size(-1)), mode="bilinear", align_corners=False)
        
        c1 = c1 + x1
        c2 = c2 + x2
        c3 = c3 + x3
        c4 = c4 + x4

        return c1, c2, c3, c4

    def forward(
        self, x: torch.Tensor, rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of VitAdapter.

        Args:
            x: Input tensor from Vision Transformer of shape (B, H*W, C)
            rgb: RGB image tensor of shape (B, C, H, W) used for spatial prior

        Returns:
            Tuple of four multi-scale feature maps (f1, f2, f3, f4)
        """
        h, w = rgb.size(2), rgb.size(3)
        H, W = x.shape[1], x.shape[2]

        # Extract spatial prior features from RGB image
        c1, c2, c3, c4, d1, d2, d3, d4 = self.spm(rgb)
        c1, c2, c3, c4 = self._add_level_embed(c1, c2, c3, c4)
        
        # Concatenate features for current level configuration
        c = self._concatenate_features(c1, c2, c3, c4)
        d = [d1, d2, d3, d4]
        sizes = [di.shape[-2:] for di in d]

        deform_inputs1, deform_inputs2= deform_inputs(
            h, w, device=x.device, patch_size=self.patch_size,
            n_levels=self.n_levels, high_res=self.high_res, c_sizes=sizes
        )

        # Extract and process ViT features
        num_reg_tokens = self.spatial_backbone.num_prefix_tokens
        pos_embed_out = self.spatial_backbone.net._pos_embed(x)
        
        # TODO - Handle case where positional embedding returns a tuple (e.g., for EVA or certain ViT variants)
        if isinstance(pos_embed_out, tuple):
            x, rot_pos_embed = pos_embed_out
        else:
            x = pos_embed_out
            rot_pos_embed = None

        bs, n, dim = x.shape
        if num_reg_tokens > 0:
            reg_tokens = x[:, :num_reg_tokens, :]
            x = x[:, num_reg_tokens:, :]
        else:
            reg_tokens = None
        # Apply interaction blocks for each pyramid level
        vit_features = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            blk = self.spatial_backbone.net.blocks[indexes[0] : indexes[-1] + 1]
            
            x, c = layer(
                x, c, blk, 
                deform_inputs1, 
                deform_inputs2,
                rot_pos_embed,
                reg_tokens
            )
            
            vit_features.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split concatenated features back to individual levels
        c1, c2, c3, c4 = self._split_concatenated_features(c, c1.size(1), c2.size(1), c3.size(1))

        # Reshape features to spatial format
        c1, c2, c3, c4 = self._reshape_features_to_spatial(
            c1, c2, c3, c4, deform_inputs1, bs, dim
        )

        # Optionally fuse ViT features
        c1, c2, c3, c4 = self._fuse_vit_features(c1, c2, c3, c4, vit_features)

        # Apply final layer normalization
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        return f1, f2, f3, f4
