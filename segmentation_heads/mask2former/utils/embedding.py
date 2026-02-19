# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.

# This source code is licensed under the Apache License, Version 2.0.
# 
# References:
#   https://github.com/huggingface/transformers/blob/main/src/transformers/models/swinv2/modeling_swinv2.py


from transformers.models.swinv2.modeling_swinv2 import Swinv2Embeddings


class Swinv2EmbeddingsCustom(Swinv2Embeddings):
    """
    Swinv2 embedding layer without the actual patch embedding, as we already have
    embeddings from the CARL model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        pixel_values,
        bool_masked_pos = None,
        interpolate_pos_encoding: bool = True,
    ):
        _, num_channels, height, width = pixel_values.shape
        (embeddings, output_dimensions) = pixel_values.flatten(2).transpose(1, 2), (height, width)
        embeddings = self.norm(embeddings)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            if interpolate_pos_encoding:
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions