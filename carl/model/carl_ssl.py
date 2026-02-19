# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0 
# found in the LICENSE file in the root directory of this source tree.

from typing import Dict, Any

import torch
import torch.nn.functional as F
from einops import rearrange

from carl.model.carl import CARLModel
from carl.modules.ssl_modules.spectral_predictor import SpectralPredictor
from carl.modules.ssl_modules.spatial_predictor import SpatialPredictor
from carl.modules.utils.ssl_utils import apply_masks


class CARLSSLModel(CARLModel):
    """CARL model for self-supervised learning (SSL) tasks.

    This class extends the base CARLModel to include additional
    functionalities specific to SSL, such as feature extraction
    and projection heads.

    Attributes:
        spatial_encoder: The spatial encoder module.
        spectral_encoder: The spectral encoder module.
        projection_head: A projection head for SSL tasks.
    """

    def __init__(
        self,
        spec_encoder_kwargs: Dict[str, Any] = None,
        spat_encoder_kwargs: Dict[str, Any] = None,
        spec_predictor_kwargs: Dict[str, Any] = None,
        spat_predictor_kwargs: Dict[str, Any] = None,
        patch_size: int = 8,
        image_size: int = 224,
        **kwargs
    ) -> None:
        """Initialize the CARL SSL model.

        Args:
            *args: Positional arguments for the base CARLModel.
            **kwargs: Keyword arguments for the base CARLModel.
        """
        super().__init__(
            spec_encoder_kwargs,
            spat_encoder_kwargs,
            patch_size, 
        )

        # Additional initialization for SSL can be added here
        spec_predictor_kwargs["embed_dim"] = spec_encoder_kwargs["embed_dim"]
        spec_predictor_kwargs["n_queries"] = self.spectral_tf.n_queries
        self.spectral_predictor = SpectralPredictor(**spec_predictor_kwargs)

        spat_predictor_kwargs["num_patches"] = (image_size // patch_size) ** 2
        spat_predictor_kwargs["embed_dim"] = self.spatial_encoder.embed_dim
        self.spatial_predictor = SpatialPredictor(**spat_predictor_kwargs)

        self.image_size = image_size

    def forward_student(
        self,
        img,
        wlens,
        masks_enc_spat,
        masks_enc_spec,
        masks_pred_spat,
        masks_pred_spec,
    ):

        b, c, h, w = img.shape
        device = img.device
        img = rearrange(img, "b c h w -> (b c) 1 h w", b=b, c=c, h=h, w=w)
        embedded_img = self.embedder(img)
        nh, nw = embedded_img.shape[-2:]
        d = embedded_img.shape[1]
        n_patches = nh * nw
        
        # Combine wavelength operations
        embedded_img = rearrange(embedded_img, "(b c) d nh nw -> (b nh nw) c d", b=b, c=c, nh=nh, nw=nw, d=d)

        # Optimize spatial mask expansion and application
        org_embedded_img = rearrange(embedded_img, "(b nh nw) c d -> (b c) (nh nw) d", b=b, c=c, nh=nh, nw=nw, d=d)
        masks_enc_spat_expanded = masks_enc_spat[0].unsqueeze(1).expand(b, c, -1).reshape(b * c, -1)
        masks_enc_spat_expanded = [masks_enc_spat_expanded]
        masked_embedded_img = apply_masks(org_embedded_img, masks_enc_spat_expanded)
        n_masked = masked_embedded_img.shape[1]
        masked_embedded_img = rearrange(masked_embedded_img, "(b c) n d -> (b n) c d", b=b, c=c, n=n_masked, d=d)
        wavelengths_rearranged = wlens.unsqueeze(1).expand(b, n_masked, c).reshape(b * n_masked, c)

        # More efficient mask_2d creation
        mask_2d = torch.zeros((b, n_patches), dtype=torch.bool, device=device)
        mask_2d.scatter_(1, masks_enc_spat[0], True)
        mask_2d = mask_2d.view(b, nh, nw)

        # Optimize spectral mask processing
        masks_enc_spec = [m.view(b, n_patches, -1) for m in masks_enc_spec]
        masks_enc_spec = [apply_masks(m, masks_enc_spat) for m in masks_enc_spec]
        masks_enc_spec = [m.view(b * n_masked, -1) for m in masks_enc_spec]
        masks_pred_spec = [m.view(b, n_patches, -1) for m in masks_pred_spec]
        masks_pred_spec = [apply_masks(m, masks_enc_spat) for m in masks_pred_spec]
        masks_pred_spec = [m.view(b * n_masked, -1) for m in masks_pred_spec]

        assert masks_enc_spec[0].shape[0] == masked_embedded_img.shape[0], f"{masks_enc_spec[0].shape[0]} vs {masked_embedded_img.shape[0]}"
        assert masked_embedded_img.shape[:2] == wavelengths_rearranged.shape, f"{masked_embedded_img.shape} vs {wavelengths_rearranged.shape}"

        out_tf = self.spectral_tf(masked_embedded_img, wavelengths_rearranged, masks_enc_spec)
        queries = out_tf["queries"]

        spec_idx_optim = torch.randperm(queries.shape[0])[:b]
        queries_optim = queries[spec_idx_optim]
        wlens_optim = wavelengths_rearranged[spec_idx_optim]
        masks_pred_spec = [m[spec_idx_optim] for m in masks_pred_spec]
        masks_enc_spec = [m[spec_idx_optim] for m in masks_enc_spec]
        pred_spectral_token = self.spectral_predictor(queries_optim, wlens_optim, masks_enc_spec, masks_pred_spec)

        query_readout = queries.sum(1)
        query_readout = F.layer_norm(query_readout, (query_readout.shape[-1],))
        queries_proj = self.linear_connector(query_readout)
        
        # More efficient spatial input creation using scatter
        embed_dim = queries_proj.shape[-1]
        spatial_input = torch.zeros((b, n_patches, embed_dim), dtype=queries_proj.dtype, device=device)
        mask_2d_flat = mask_2d.view(b, -1)
        spatial_input[mask_2d_flat] = queries_proj
        spatial_input = spatial_input.view(b, nh, nw, embed_dim)

        last_hidden_state = self.spatial_encoder(spatial_input, masks=masks_enc_spat)
        last_hidden_state = F.layer_norm(last_hidden_state, (last_hidden_state.size(-1),))
        
        pred_token = self.spatial_predictor(last_hidden_state, masks_enc_spat, masks_pred_spat)

        pred_spectral_token = F.layer_norm(pred_spectral_token, (pred_spectral_token.size(-1),))
        pred_token = F.layer_norm(pred_token, (pred_token.size(-1),))

        return (
            pred_token,
            pred_spectral_token,
            spec_idx_optim,
            masks_pred_spec,
        )

    def forward_teacher(self, img, wlens):
        """
        Args:
            x (torch.Tensor): (batch_size, channels)
            w (torch.Tensor): (batch_size, channels)
        """
        b, c, h, w = img.shape
        input_tokens = rearrange(img, "b c h w -> (b c) 1 h w", b=b, c=c, h=h, w=w)
        input_tokens = self.embedder(input_tokens)
        nh, nw = input_tokens.shape[-2:]
        d = input_tokens.shape[1]
        n_patches = nh * nw
        
        # Optimize wavelength and token rearrangement
        wlens_rear = wlens.unsqueeze(1).expand(b, n_patches, c).reshape(b * n_patches, c)
        input_tokens = rearrange(input_tokens, "(b c) d nh nw -> (b nh nw) c d", b=b, c=c, nh=nh, nw=nw, d=d)
        assert input_tokens.ndim == 3, "Input tensor must have 3 dimensions"
        assert wlens_rear.ndim == 2, f"Wavelength tensor must have 2 dimensions, but has {wlens_rear.ndim} dimensions"

        out_tf = self.spectral_tf(input_tokens, wlens_rear)
        queries = out_tf["queries"]
        spectral_tokens = out_tf["spectral_tokens"]
        cls_tokens = queries.sum(1)
        cls_tokens = F.layer_norm(cls_tokens, (cls_tokens.shape[-1],))
        
        cls_tokens = self.linear_connector(cls_tokens)
        cls_tokens = rearrange(cls_tokens, "(b nh nw) d -> b nh nw d", b=b, nh=nh, nw=nw)
        last_hidden_state = self.spatial_encoder(cls_tokens)

        return last_hidden_state, spectral_tokens