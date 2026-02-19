# Copyright (c) Shanghai AI Lab. All rights reserved.
# 
# This source code is licensed under the Apache License, Version 2.0.
# 
# References:
#   https://github.com/czczup/ViT-Adapter/blob/main/segmentation/mmseg_custom/models/backbones/adapter_modules.py


import logging
from functools import partial
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath

from segmentation_heads.upernet.utils.ops.modules import MSDeformAttn


def get_reference_points(spatial_shapes: List[Tuple[int, int]], device: torch.device) -> torch.Tensor:
    """
    Generate normalized reference points for spatial shapes.

    Creates a grid of normalized coordinates (0-1) for each spatial shape,
    used as reference points in deformable attention mechanisms.

    Args:
        spatial_shapes: List of (height, width) tuples for each pyramid level
        device: Device to create tensors on

    Returns:
        Tensor of shape (1, total_points, 1, 2) containing normalized (x, y) coordinates
    """
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H_ - 0.5, H_, dtype=torch.float32, device=device
            ),
            torch.linspace(
                0.5, W_ - 0.5, W_, dtype=torch.float32, device=device
            ),
            indexing='ij',
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points

def deform_inputs(
    h: int, w: int,
    device: torch.device,
    patch_size: int = 14,
    n_levels: int = 4,
    high_res: bool = False,
    c_sizes: List[Tuple[int, int]] = None
) -> Tuple[List, List]:
    """Wrapper function to generate deformation inputs based on configuration. """
    # define strides
    if n_levels == 4:
        c_sizes = c_sizes
    elif n_levels == 3 and high_res:
        c_sizes = c_sizes[:-1]
    elif n_levels == 3 and not high_res:
        c_sizes = c_sizes[1:]

    # Reference points from patch-level ViT features
    spatial_shapes = torch.as_tensor(c_sizes, dtype=torch.long, device=device)
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(h // patch_size, w // patch_size)], device
    )
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    # Reference points from CNN pyramid features to ViT features
    spatial_shapes = torch.as_tensor(
        [(h // patch_size, w // patch_size)], dtype=torch.long, device=device
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        c_sizes,
        device,
    )
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2

class ConvFFN(nn.Module):
    """
    Convolutional Feed-Forward Network.

    A lightweight FFN that uses depthwise convolution for spatial interaction
    between tokens represented in multi-scale spatial format.

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden feature dimension (default: same as in_features)
        out_features: Output feature dimension (default: same as in_features)
        act_layer: Activation function class (default: GELU)
        drop: Dropout rate (default: 0.0)
        n_levels_c: Number of pyramid levels for spatial decomposition (default: 4)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
        n_levels_c: int = 4,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features, n_levels_c)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, spatial_shapes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, N, C) where N is sequence length
            spatial_shapes: Tensor containing spatial dimensions for each pyramid level

        Returns:
            Output tensor of same shape as input
        """
        x = self.fc1(x)
        x = self.dwconv(x, spatial_shapes)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    """
    Depthwise Convolutional layer for multi-level feature processing.

    Applies depthwise convolution independently to sequences from different
    pyramid levels to enable spatial interaction within each scale.

    Args:
        dim: Feature dimension
        n_levels_c: Number of pyramid levels to process
    """

    def __init__(self, dim: int = 768, n_levels_c: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.n_levels_c = n_levels_c

    def forward(
        self, x: torch.Tensor, spatial_shapes: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass applying depthwise convolution to multi-scale features.

        Args:
            x: Input tensor of shape (B, N, C) where N is total sequence length
            spatial_shapes: Tensor of shape (n_levels, 2) with spatial dims per level

        Returns:
            Output tensor of same shape as input
        """
        B, N, C = x.shape
        
        if self.n_levels_c == 4:
            # Process 4 pyramid levels
            x1, x2, x3, x4 = self._reshape_levels_4(x, spatial_shapes, B, C)
            x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
            x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
            x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
            x4 = self.dwconv(x4).flatten(2).transpose(1, 2)
            x = torch.cat([x1, x2, x3, x4], dim=1)
        else:
            # Process 3 pyramid levels
            x1, x2, x3 = self._reshape_levels_3(x, spatial_shapes, B, C)
            x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
            x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
            x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
            x = torch.cat([x1, x2, x3], dim=1)
        
        return x

    def _reshape_levels_4(
        self, x: torch.Tensor, spatial_shapes: torch.Tensor, B: int, C: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape 4-level sequence into spatial tensors."""
        a, b = spatial_shapes[0]
        x1 = x[:, 0 : a * b, :].transpose(1, 2).view(B, C, a, b).contiguous()
        
        cur_idx = a * b
        a, b = spatial_shapes[1]
        x2 = x[:, cur_idx : cur_idx + a * b, :].transpose(1, 2).view(B, C, a, b).contiguous()
        
        cur_idx += a * b
        a, b = spatial_shapes[2]
        x3 = x[:, cur_idx : cur_idx + a * b, :].transpose(1, 2).view(B, C, a, b).contiguous()
        
        cur_idx += a * b
        a, b = spatial_shapes[3]
        x4 = x[:, cur_idx : cur_idx + a * b, :].transpose(1, 2).view(B, C, a, b).contiguous()
        
        return x1, x2, x3, x4

    def _reshape_levels_3(
        self, x: torch.Tensor, spatial_shapes: torch.Tensor, B: int, C: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape 3-level sequence into spatial tensors."""
        a, b = spatial_shapes[0]
        x1 = x[:, 0 : a * b, :].transpose(1, 2).view(B, C, a, b).contiguous()
        
        cur_idx = a * b
        a, b = spatial_shapes[1]
        x2 = x[:, cur_idx : cur_idx + a * b, :].transpose(1, 2).view(B, C, a, b).contiguous()
        
        cur_idx += a * b
        a, b = spatial_shapes[2]
        x3 = x[:, cur_idx:, :].transpose(1, 2).view(B, C, a, b).contiguous()
        
        return x1, x2, x3


class Extractor(nn.Module):
    """
    Deformable attention-based feature extractor for multi-scale fusion.

    Extracts features from one pyramid level using deformable attention to query
    features from another scale, optionally followed by a convolutional FFN.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads (default: 6)
        n_points: Number of sampling points per attention head (default: 4)
        n_levels: Number of pyramid levels to attend to (default: 1)
        deform_ratio: Scaling factor for deformable attention (default: 1.0)
        with_cffn: Whether to use convolutional FFN (default: True)
        cffn_ratio: Hidden feature ratio in CFFN (default: 0.25)
        drop: Dropout rate (default: 0.0)
        drop_path: Stochastic depth rate (default: 0.0)
        norm_layer: Normalization layer class (default: LayerNorm)
        with_cp: Whether to use gradient checkpointing (default: False)
        n_levels_c: Number of CNN pyramid levels (default: 4)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        n_points: int = 4,
        n_levels: int = 1,
        deform_ratio: float = 1.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=None,
        with_cp: bool = False,
        n_levels_c: int = 4,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        
        if with_cffn:
            self.ffn = ConvFFN(
                in_features=dim,
                hidden_features=int(dim * cffn_ratio),
                drop=drop,
                n_levels_c=n_levels_c,
            )
            self.ffn_norm = norm_layer(dim)
            self.drop_path = (
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        feat: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        spatial_shapes_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply deformable attention and optional FFN.

        Args:
            query: Query features of shape (B, N, C)
            reference_points: Reference points for deformable attention
            feat: Feature tensor to attend to
            spatial_shapes: Spatial dimensions of feature pyramid levels
            level_start_index: Indices marking start of each level in concatenated features
            spatial_shapes_c: Spatial shapes for FFN processing

        Returns:
            Updated query features
        """

        def _inner_forward(query, feat):
            # Apply deformable attention with layer norm on inputs
            attn = self.attn(
                self.query_norm(query),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
            query = query + attn

            # Optional FFN with residual connection
            if self.with_cffn:
                query = query + self.drop_path(
                    self.ffn(self.ffn_norm(query), spatial_shapes_c)
                )
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    """
    Deformable attention-based feature injector with learnable scale.

    Injects features from one pyramid level into another using deformable attention
    with a learnable scaling parameter.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads (default: 6)
        n_points: Number of sampling points per attention head (default: 4)
        n_levels: Number of pyramid levels to attend to (default: 1)
        deform_ratio: Scaling factor for deformable attention (default: 1.0)
        norm_layer: Normalization layer class (default: LayerNorm)
        init_values: Initial value for learnable scale (default: 0.0)
        with_cp: Whether to use gradient checkpointing (default: False)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        n_points: int = 4,
        n_levels: int = 1,
        deform_ratio: float = 1.0,
        norm_layer=None,
        init_values: float = 0.0,
        with_cp: bool = False,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
        )
        # Learnable scale for attention output
        self.gamma = nn.Parameter(
            init_values * torch.ones(dim), requires_grad=True
        )

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        feat: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply scaled deformable attention.

        Args:
            query: Query features of shape (B, N, C)
            reference_points: Reference points for deformable attention
            feat: Feature tensor to attend to
            spatial_shapes: Spatial dimensions of feature pyramid levels
            level_start_index: Indices marking start of each level in concatenated features

        Returns:
            Updated query features with scaled attention
        """

        def _inner_forward(query, feat):
            # Apply scaled deformable attention with layer norm on inputs
            attn = self.attn(
                self.query_norm(query),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    """
    Interaction block combining injector and extractor for bidirectional feature flow.

    This block implements bidirectional interaction between ViT features and
    multi-scale CNN features through learnable deformable attention mechanisms.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads (default: 6)
        n_points: Number of sampling points per attention head (default: 4)
        norm_layer: Normalization layer class (default: LayerNorm)
        drop: Dropout rate (default: 0.0)
        drop_path: Stochastic depth rate (default: 0.0)
        with_cffn: Whether to use convolutional FFN (default: True)
        cffn_ratio: Hidden feature ratio in CFFN (default: 0.25)
        init_values: Initial value for learnable scales (default: 0.0)
        deform_ratio: Scaling factor for deformable attention (default: 1.0)
        extra_extractor: Whether to add extra extractor layers (default: False)
        with_cp: Whether to use gradient checkpointing (default: False)
        n_levels: Number of pyramid levels (default: 4)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        n_points: int = 4,
        norm_layer=None,
        drop: float = 0.0,
        drop_path: float = 0.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        init_values: float = 0.0,
        deform_ratio: float = 1.0,
        extra_extractor: bool = False,
        with_cp: bool = False,
        n_levels: int = 4,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.with_cp = with_cp
        
        # Inject CNN features into ViT features
        self.injector = Injector(
            dim=dim,
            n_levels=n_levels,
            num_heads=num_heads,
            init_values=init_values,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cp=with_cp,
        )
        
        # Extract ViT features to update CNN features
        self.extractor = Extractor(
                    dim=dim,
                    n_levels=1,
                    num_heads=num_heads,
                    n_points=n_points,
                    norm_layer=norm_layer,
                    deform_ratio=deform_ratio,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    drop=drop,
                    drop_path=drop_path,
                    with_cp=with_cp,
                    n_levels_c=n_levels,
                )        
        
        # Additional extractors for deeper interaction
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                        n_levels_c=n_levels,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        blocks: nn.Module,
        deform_inputs1: List,
        deform_inputs2: List,
        rot_pos_embed: Optional[torch.Tensor] = None,
        reg_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with bidirectional feature interaction.

        Args:
            x: ViT features of shape (B, N, C)
            c: CNN features of shape (B, M, C)
            blocks: ViT transformer blocks to apply
            deform_inputs1: Deformation inputs for injector
            deform_inputs2: Deformation inputs for extractors
            H: Height of feature map
            W: Width of feature map
            rot_pos_embed: Optional rotary position embeddings

        Returns:
            Tuple of (updated_vit_features, updated_cnn_features)
        """
        if reg_tokens is not None:
            num_reg_tokens = reg_tokens.shape[1]
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2],
        )
        
        # Apply ViT transformer blocks
        if blocks is not None:
            if reg_tokens is not None:
                x = torch.cat([reg_tokens, x], dim=1)
            for idx, blk in enumerate(blocks):
                if rot_pos_embed is not None:
                    x = blk(x, rope=rot_pos_embed)
                else:
                    x = blk(x)
            if reg_tokens is not None:
                reg_tokens = x[:, :num_reg_tokens]
                x = x[:, num_reg_tokens:]

        # Extract ViT features to update CNN features
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            spatial_shapes_c=deform_inputs1[1],
        )
        
        # Apply extra extractors for deeper interaction
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    spatial_shapes_c=deform_inputs1[1],
                )
        
        return x, c

class SpatialPriorModule(nn.Module):
    """
    Spatial Prior Module for extracting multi-scale features from RGB images.

    This module processes RGB input through a stem and progressive convolutional
    layers to extract a 4-level spatial pyramid. Features are projected to a
    common embedding dimension for later use in the adapter.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB). Default: 3
        inplanes: Number of channels in stem convolution. Default: 64
        embed_dim: Output embedding dimension for all levels. Default: 384
        with_cp: Whether to use gradient checkpointing. Default: False
        track_running_stats: Whether batch norm layers track running statistics. Default: True
    """

    def __init__(
        self,
        in_channels: int = 3,
        inplanes: int = 64,
        embed_dim: int = 384,
    ):
        super().__init__()

        # Initial stem: 3 conv blocks with 2x downsampling + maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Progressive downsampling layers: 2x, 4x, 4x strides
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                inplanes,
                2 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                2 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                4 * inplanes,
                4 * inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True),
        )
        
        # 1x1 projections to common embedding dimension
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Extract multi-scale features from input image.

        Args:
            x: Input image tensor of shape (B, C, H, W)

        Returns:
            Tuple of 8 tensors:
                - d1, d2, d3, d4: Flattened sequence features for each level (B, N_i, D)
                - c1, c2, c3, c4: Spatial features for each level (B, D, H_i, W_i)
        """

        c1 = self.stem(x)        # 4x downsampling
        c2 = self.conv2(c1)      # 8x downsampling
        c3 = self.conv3(c2)      # 16x downsampling
        c4 = self.conv4(c3)      # 32x downsampling
        
        # Project to embedding dimension
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        # Convert spatial features to sequence format
        bs, dim, _, _ = c1.shape
        d1 = c1.view(bs, dim, -1).transpose(1, 2)  # (B, N1, D)
        d2 = c2.view(bs, dim, -1).transpose(1, 2)  # (B, N2, D)
        d3 = c3.view(bs, dim, -1).transpose(1, 2)  # (B, N3, D)
        d4 = c4.view(bs, dim, -1).transpose(1, 2)  # (B, N4, D)

        return d1, d2, d3, d4, c1, c2, c3, c4