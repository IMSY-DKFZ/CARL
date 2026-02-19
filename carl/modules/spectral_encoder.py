# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0 
# found in the LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import List, Tuple, Optional

import timm
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from carl.modules.utils.wavelength_pos_enc import PositionalEncoding
from carl.modules.utils.attention import SelfAttention, CrossAttention
from carl.modules.utils.mlp import Mlp
from carl.modules.utils.block import Block
from carl.modules.utils.utils import get_1d_sincos_pos_embed
from carl.modules.utils.ssl_utils import apply_masks


class SpectralEncoder(nn.Module):
    """Encoder that ingests spectral tokens and produces query embeddings.

    The encoder accepts a sequence of spectral tokens ``x`` and a corresponding
    wavelength tensor ``w``. It builds wavelength positional encodings (via
    ``PositionalEncoding``), maintains a small set of learned ``queries``
    (spectral representations), and alternates self- and cross-attention blocks
    so queries can attend to the spectral tokens and produce a compact spectral
    representation for downstream tasks.

    Args:
        embed_dim: Embedding dimension for tokens and queries.
        n_queries: Number of learned spectral representations.
        pos_enc_sigma: Scaling parameter for wavelength positional encoding.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for MLP hidden dimension.
        qkv_bias: Whether to include bias in query, key, value projections.
        ffn_bias: Whether to include bias in feed-forward network.
        proj_bias: Whether to include bias in projection layers.
        drop_path_rate: Stochastic depth rate.
        drop_path_uniform: Whether to use uniform stochastic depth schedule.
        act_layer: Activation function layer.
        proj_drop: Dropout rate for projections.
        drop: General dropout rate.
        attn_drop: Dropout rate for attention weights.
        layer_scale: Layer scaling factor for initialization.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_queries: int = 4,
        pos_enc_sigma: float = 3,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        act_layer=nn.GELU,
        proj_drop: float = 0.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        layer_scale: float = 1e-5,
    ):
        super().__init__()

        # Store configuration
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.n_queries = n_queries

        # Initialize learned query vectors and their positional encodings
        self.queries = nn.Parameter(
            torch.zeros(1, n_queries, embed_dim), requires_grad=True
        )
        self.queries_pos_enc = nn.Parameter(
            torch.zeros(1, n_queries, embed_dim), requires_grad=False
        )

        # Build stochastic depth schedule
        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # Build alternating attention blocks (self-attention, cross-attention, ...)
        blocks_list = []
        for i in range(depth):
            attn_class = SelfAttention if i % 2 == 0 else CrossAttention
            blocks_list.append(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=Mlp,
                    init_values=layer_scale,
                    attn_class=attn_class,
                    proj_drop=proj_drop,
                    drop=drop,
                    attn_drop=attn_drop,
                )
            )

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.positional_embedding = PositionalEncoding(embed_dim, sigma=pos_enc_sigma)

        # Set up random generator for weight initialization
        self.rng = torch.Generator()
        self.rng.manual_seed(0)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize queries
        torch.nn.init.trunc_normal_(self.queries, std=0.02, generator=self.rng)
        # Initialize positional encodings for queries
        queries_pos_enc = get_1d_sincos_pos_embed(
            self.embed_dim, self.n_queries, cls_token=False, theta=10000
        )
        self.queries_pos_enc.data.copy_(torch.from_numpy(queries_pos_enc).float().unsqueeze(0))

        if self.embed_dim == 384:
            model_init = "vit_small_patch14_dinov2.lvd142m"
        elif self.embed_dim == 768:
            model_init = "vit_base_patch14_dinov2.lvd142m"
        else:
            def _init_weights(module):
                if isinstance(module, nn.Linear):
                    trunc_normal_(module.weight, std=.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif hasattr(module, 'init_weights'):
                    module.init_weights()
            self.apply(_init_weights)
            return

        # Initialize spectral encoder blocks from DINOv2 weights
        dino = timm.create_model(model_init, pretrained=True)
        block_weights = dino.state_dict()
        block_weights = {k: v for k, v in block_weights.items() if 'blocks' in k}
        new_block_weights = {}
        for k, v in block_weights.items():
            block_id = int(k.split('.')[1])
            if block_id >= self.n_blocks:
                continue
            if "attn.qkv.weight" in k:
                w_q = v.data[:self.embed_dim]
                w_k = v.data[self.embed_dim:(2*self.embed_dim)]
                w_v = v.data[-self.embed_dim:]
                k_q = k.replace("attn.qkv.weight", "attn.q.weight")
                k_k = k.replace("attn.qkv.weight", "attn.k.weight")
                k_v = k.replace("attn.qkv.weight", "attn.v.weight")
                new_block_weights[k_q] = w_q
                new_block_weights[k_k] = w_k
                new_block_weights[k_v] = w_v
            elif "attn.qkv.bias" in k:
                b_q = v.data[:self.embed_dim]
                b_k = v.data[self.embed_dim:(2*self.embed_dim)]
                b_v = v.data[-self.embed_dim:]
                k_q = k.replace("attn.qkv.bias", "attn.q.bias")
                k_k = k.replace("attn.qkv.bias", "attn.k.bias")
                k_v = k.replace("attn.qkv.bias", "attn.v.bias")
                new_block_weights[k_q] = b_q
                new_block_weights[k_k] = b_k
                new_block_weights[k_v] = b_v
            else:
                new_block_weights[k] = v
        missing, unexpected = self.load_state_dict(new_block_weights, strict=False)
        logging.info(f"Spectral Encoder weight initialization from {model_init}: {missing} missing, {unexpected} unexpected")
        del dino

    def prepare_tokens(
        self, x: torch.Tensor, w: torch.Tensor, masks=Optional[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encodings to spectral tokens and apply optional masks.

        Args:
            x: Spectral token embeddings of shape (batch_size, seq_len, embed_dim).
            w: Wavelength values of shape (batch_size, seq_len).
            masks: Optional mask tensor to apply to tokens.

        Returns:
            Tuple of (encoded_tokens, masked_wavelengths):
                - encoded_tokens: Tokens with positional encodings added.
                - masked_wavelengths: Wavelength tensor with masks applied.
        """
        pos_enc = self.positional_embedding.forward(w)

        # Align batch dimensions if positional encoding is shared
        repeats = x.shape[0] // pos_enc.shape[0]
        assert pos_enc.shape[0] * repeats == x.shape[0], (
            "Batch size of x must be a multiple of positional encoding batch size."
            f" Got {x.shape[0]} and {pos_enc.shape[0]}."
        )
        if repeats > 1:
            pos_enc = pos_enc.repeat_interleave(repeats, dim=0)

        x = x + pos_enc
        # Apply masks if provided
        if masks is not None:
            x = apply_masks(x, masks)
 
        return x

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, masks: Optional[List[torch.Tensor]] = None
    ) -> dict[str, torch.Tensor]:
        """Encode spectral tokens and produce query embeddings.

        The network alternates between refining the spectral tokens with
        self-attention blocks and allowing the learned queries to attend to
        the tokens with cross-attention blocks.

        Args:
            x: Spectral token embeddings of shape (batch_size, seq_len, embed_dim).
            w: Wavelength values of shape (batch_size, seq_len).
            masks: Optional channel indices to keep, used in CARL-SSL for masked tokens.

        Returns:
            Dictionary:
                - queries: Learned query embeddings of shape (batch_size, n_queries, embed_dim).
                - spectral_tokens: Refined spectral tokens of shape (batch_size, seq_len, embed_dim).
        """

        # Prepare tokens with positional encodings and apply masks
        spectral_tokens = self.prepare_tokens(x, w, masks)

        # Initialize and encode query vectors
        queries = self.queries.expand(spectral_tokens.shape[0], -1, -1)
        queries_pos_enc = self.queries_pos_enc.expand_as(queries)
        queries = queries + queries_pos_enc

        # Apply alternating self- and cross-attention blocks
        for blk in self.blocks:
            if isinstance(blk.attn, CrossAttention):
                # Cross-attention: queries attend to spectral tokens
                queries = blk(queries, spectral_tokens, spectral_tokens)
            else:
                # Self-attention: refine spectral tokens
                spectral_tokens = blk(spectral_tokens)

        # Normalize query embeddings
        queries = self.norm(queries)

        return {"queries": queries, "spectral_tokens": spectral_tokens}
