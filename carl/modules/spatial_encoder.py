"""Spatial encoder using Vision Transformer from TIMM.

This module wraps the TIMM EVA2 Vision Transformer model for spatial feature
encoding. It supports gradient checkpointing for memory efficiency.
"""

import logging
import math
from typing import Optional, Dict, Any, List, Union

import timm
from timm.models import Eva, VisionTransformer
import torch
import torch.nn as nn

from carl.modules.utils.ssl_utils import apply_masks


class TimmWrapper(nn.Module):
    """Wrapper for TIMM Vision Transformers.
    
    Provides a clean interface to the pre-trained TIMM models.
    
    Attributes:
        net: The underlying EVA2 Vision Transformer model.
        embed_dim: Embedding dimension of the model.
        num_prefix_tokens: Number of prefix tokens (e.g., cls token).
    """
    
    def __init__(
        self,
        model_name: str = "timm/eva02_base_patch14_224.mim_in22k",
        depth: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the spatial encoder.
        
        Args:
            model_name: Name of the TIMM model to load.
            depth: Number of transformer blocks to use (uses first `depth` blocks).
            grad_checkpointing: Whether to use gradient checkpointing for memory efficiency.
            model_kwargs: Additional keyword arguments to pass to the model.
            
        Raises:
            RuntimeError: If the model cannot be loaded from TIMM.
        """
        super().__init__()
        
        model_kwargs = model_kwargs or {}
        model_kwargs["dynamic_img_size"] = True
        
        # Load pre-trained model from TIMM
        try:
            timm_module: Union[Eva, VisionTransformer] = timm.create_model(
                model_name,
                pretrained=True,
                **model_kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}' from TIMM: {e}")
        
        # Truncate to specified depth
        if depth is not None:
            timm_module.blocks = timm_module.blocks[:depth]
        self.net = timm_module

        # Store configuration
        self.num_prefix_tokens = self.net.num_prefix_tokens
        self.embed_dim = self.net.embed_dim

    def forward(self, x: torch.Tensor, masks: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass through the spatial encoder.
        
        Args:
            x: Input tensor of shape (B, H, W, D) or (B, N, D) where:
               - B is batch size
               - H, W are spatial dimensions (for spatial input)
               - N is sequence length (for sequence input)
               - D is embedding dimension
            masks: Optional list of long tensors indicating kept token indices. Used in CARL-SSL.
               
        Returns:
            Encoded features of shape (B, N, D) where N includes prefix tokens.
        """
        # Apply positional embeddings
        out_pos_embed = self.net._pos_embed(x)
        # Eva and ViT support from Timm
        if isinstance(out_pos_embed, tuple):
            x, rot_pos_embed = out_pos_embed
        else:
            x = out_pos_embed
            rot_pos_embed = None
        hidden_states = x

        # Create attention mask if masks are provided
        attn_mask = self.create_attn_mask(x, masks)
        
        # Prepare kwargs for transformer blocks
        block_kwargs = {}
        if rot_pos_embed is not None:
            block_kwargs["rope"] = rot_pos_embed
        if attn_mask is not None:
            block_kwargs["attn_mask"] = attn_mask
        # Apply transformer blocks
        for block in self.net.blocks:
            hidden_states = block(hidden_states, **block_kwargs)
        # Remove prefix tokens (e.g., cls token) if present
        if self.num_prefix_tokens > 0:
            hidden_states = hidden_states[:, self.num_prefix_tokens:, :]
        
        if masks is not None:
            hidden_states = apply_masks(hidden_states, masks)

        return hidden_states
    
    def create_attn_mask(self, 
            x: torch.Tensor, 
            masks: Optional[List[torch.Tensor]] = None
        
        ) -> Optional[torch.Tensor]:
        """Create attention mask for input tensor.
        
        Args:
            x: Input tensor of shape (B, N, D).

        Returns:
            Attention mask tensor of shape (B, 1, N, N).
        """
        if masks is None:
            return None
        reg_tokens = self.num_prefix_tokens
        b, n, d = x.shape
        masks = [m.cpu() for m in masks]
        zeros = torch.zeros((b, reg_tokens), dtype=torch.bool)
        zeros = [zeros for _ in range(len(masks))]

        adjusted_mask = [m + reg_tokens for m in masks]
        adjusted_mask = [
            torch.cat([z, m], dim=1) for z, m in zip(zeros, adjusted_mask)
        ][0]

        attn_mask = torch.zeros((b, n, n), dtype=torch.bool)
        batch_indices = torch.arange(b).unsqueeze(1)  # Shape (B, 1)
        attn_mask[batch_indices, adjusted_mask] = torch.ones(
            (b, adjusted_mask.shape[1], n),
            dtype=torch.bool,
        )
        attn_mask = attn_mask.transpose(1, 2).unsqueeze(1).to(x.device)

        return attn_mask