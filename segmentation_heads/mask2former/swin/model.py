# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0 
# found in the LICENSE file in the root directory of this source tree.


import logging
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentation, Mask2FormerForUniversalSegmentationOutput
from transformers.models.mask2former.configuration_mask2former import Mask2FormerConfig


from carl.model.carl import CARLModel
from segmentation_heads.mask2former.utils import Swinv2EmbeddingsCustom



class CARL_Mask2Former(CARLModel):
    """CARL model with Vision Transformer adapter for segmentation tasks.
    
    This model extends the CARLModel by adding a segmentation head that adapts
    the output of the spatial encoder for pixel-wise classification.
    
    Attributes:
        segmentation_head: A convolutional layer that maps spatial features to class logits.
    """
    def __init__(
            self, 
            spec_encoder_kwargs: Dict[str, Any] = None,
            mask2former_kwargs: Dict[str, Any] = {},
            patch_size: int = 8,
            n_classes: int = 10,
            **kwargs) -> None:
        """Initialize the ViT adapter model.
        
        Args:
            n_classes: Number of segmentation classes.
            **kwargs: Additional keyword arguments for the CARLModel.
        """
        super().__init__(
            spec_encoder_kwargs=spec_encoder_kwargs,
            patch_size=patch_size,
            **kwargs
        )
        
        model_name = mask2former_kwargs.pop("model_name")
        if model_name is not None:
            assert "swin" in model_name.lower(), "Implementation only supports Swin backbone."
            self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name
            ).train()

            logging.info(f"Mask2former config: {self.mask2former.config}")

            config = self.mask2former.model.pixel_level_module.encoder.embeddings.config

            logging.info(f"Swin embeddings config: {config}")
            state_dict = self.mask2former.model.pixel_level_module.encoder.embeddings.state_dict()
            self.mask2former.model.pixel_level_module.encoder.embeddings = Swinv2EmbeddingsCustom(config)
            missing = self.mask2former.model.pixel_level_module.encoder.embeddings.load_state_dict(state_dict)

        else:
            mask2former_kwargs["num_labels"] = n_classes
            mask2former_config = Mask2FormerConfig(**mask2former_kwargs)
            self.mask2former = Mask2FormerForUniversalSegmentation(mask2former_config)
            config = self.mask2former.model.pixel_level_module.encoder.embeddings.config
            self.mask2former.model.pixel_level_module.encoder.embeddings = Swinv2EmbeddingsCustom(config)
            self.mask2former.post_init()

        self.linear_connector = nn.Linear(
            self.spectral_tf.embed_dim,
            config.embed_dim,
        )

    def forward(
        self, 
        img: torch.Tensor, 
        wavelengths: torch.Tensor,
        batch_instances: Optional[list] = None,
        batch_classes: Optional[list] = None,
    ) -> Mask2FormerForUniversalSegmentationOutput:
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
            "(b nh nw) d -> b d nh nw",
            b=batch_size,
            nh=patch_height,
            nw=patch_width
        )
        out: Mask2FormerForUniversalSegmentationOutput = self.mask2former(
            pixel_values=spec_representations,
            mask_labels=batch_instances,
            class_labels=batch_classes,
            return_dict=True,
        )

        return out


    @torch.no_grad()
    def post_process_semantic_segmentation(
            self, 
            outputs: Mask2FormerForUniversalSegmentationOutput,
            target_sizes = None
        ):
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                with torch.amp.autocast(device_type='cuda'):
                    resized_logits = torch.nn.functional.interpolate(
                        segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                    )
                resized_logits = resized_logits.cpu()
                semantic_map = resized_logits[0]
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]
        semantic_segmentation = torch.stack(semantic_segmentation, dim=0).cuda()
        return semantic_segmentation

