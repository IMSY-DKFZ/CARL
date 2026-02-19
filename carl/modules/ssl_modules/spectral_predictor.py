# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0 
# found in the LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from carl.modules.utils.wavelength_pos_enc import PositionalEncoding
from carl.modules.utils.attention import CrossAttention
from carl.modules.utils.block import Block
from carl.modules.utils.ssl_utils import apply_masks, repeat_interleave_batch


class SpectralPredictor(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        pos_enc_sigma=3,
        n_queries=4,
    ):
        super().__init__()
        if embed_dim != predictor_embed_dim:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        else:
            self.predictor_embed = nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.positional_embedding = PositionalEncoding(predictor_embed_dim, sigma=pos_enc_sigma)
        self.queries_pos_enc = nn.Parameter(torch.zeros(1, n_queries, predictor_embed_dim), requires_grad=True)
    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim), requires_grad=True)
        
        # --
        self.predictor_blocks = nn.ModuleList([])
        for i in range(depth):
            self.predictor_blocks.append(
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attn_class=CrossAttention,
                )
            )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        
        # ------
        self.init_std = init_std
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.queries_pos_enc, std=self.init_std)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
            self,
            x: torch.Tensor,
            wlens: torch.Tensor,
            masks_x: List[torch.Tensor],
            masks: List[torch.Tensor]
    ) -> torch.Tensor:
        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)
        x = x + self.queries_pos_enc.expand(len(x), -1, -1)

        # # -- add positional embedding to x tokens
        pos_embed_masked = self.positional_embedding(wlens)
        pos_embed_masked = apply_masks(pos_embed_masked, masks)
        pos_embed_masked = repeat_interleave_batch(pos_embed_masked, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.expand(pos_embed_masked.size(0), pos_embed_masked.size(1), -1)
        pred_tokens = pred_tokens + pos_embed_masked

        x = x.repeat(len(masks), 1, 1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            keys = torch.cat([x, pred_tokens], dim=1)
            pred_tokens = blk(pred_tokens, keys, keys)

        pred_tokens = self.predictor_norm(pred_tokens)
        pred_tokens = self.predictor_proj(pred_tokens)

        return pred_tokens