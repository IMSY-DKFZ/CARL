# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# This source code is licensed under the Apache License, Version 2.0.
# 
# References:
#   https://github.com/huggingface/transformers/blob/main/src/transformers/models/mask2former/modeling_mask2former.py


from typing import List
from torch import Tensor
import torch.nn as nn

from transformers.models.mask2former.modeling_mask2former import (
        Mask2FormerConfig,
        Mask2FormerPixelLevelModuleOutput,
        Mask2FormerModelOutput,
        Mask2FormerPixelDecoder
)


class Mask2FormerPixelLevelModuleCustom(nn.Module):
    def __init__(self, config: Mask2FormerConfig, feature_channels: List[int] = [512, 256, 128, 64]) -> None:
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://huggingface.co/papers/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`Mask2FormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels=feature_channels)

    def forward(self, backbone_features: list[Tensor], output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )


def Mask2FormerModel_CustomForward(
    self,
    pixel_values: List[Tensor],
    pixel_mask: Tensor | None = None,
    output_hidden_states: bool | None = None,
    output_attentions: bool | None = None,
    return_dict: bool | None = None,
    **kwargs,
) -> Mask2FormerModelOutput:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    pixel_level_module_output = self.pixel_level_module(
        backbone_features=pixel_values, output_hidden_states=output_hidden_states
    )

    transformer_module_output = self.transformer_module(
        multi_scale_features=pixel_level_module_output.decoder_hidden_states,
        mask_features=pixel_level_module_output.decoder_last_hidden_state,
        output_hidden_states=True,
        output_attentions=output_attentions,
    )

    encoder_hidden_states = None
    pixel_decoder_hidden_states = None
    transformer_decoder_hidden_states = None
    transformer_decoder_intermediate_states = None

    if output_hidden_states:
        encoder_hidden_states = pixel_level_module_output.encoder_hidden_states
        pixel_decoder_hidden_states = pixel_level_module_output.decoder_hidden_states
        transformer_decoder_hidden_states = transformer_module_output.hidden_states
        transformer_decoder_intermediate_states = transformer_module_output.intermediate_hidden_states

    output = Mask2FormerModelOutput(
        encoder_last_hidden_state=pixel_level_module_output.encoder_last_hidden_state,
        pixel_decoder_last_hidden_state=pixel_level_module_output.decoder_last_hidden_state,
        transformer_decoder_last_hidden_state=transformer_module_output.last_hidden_state,
        encoder_hidden_states=encoder_hidden_states,
        pixel_decoder_hidden_states=pixel_decoder_hidden_states,
        transformer_decoder_hidden_states=transformer_decoder_hidden_states,
        transformer_decoder_intermediate_states=transformer_decoder_intermediate_states,
        attentions=transformer_module_output.attentions,
        masks_queries_logits=transformer_module_output.masks_queries_logits,
    )

    if not return_dict:
        output = tuple(v for v in output.values() if v is not None)

    return output