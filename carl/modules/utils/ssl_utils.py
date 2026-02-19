# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE-THIRD-PARTY file in the root directory of this source tree.
#
# References:
#   https://github.com/facebookresearch/vjepa2/blob/main/src/masks/utils.py
#   https://github.com/facebookresearch/vjepa2/blob/main/src/utils/tensors.py

import torch

def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    if len(masks) == 1:
        # Optimize single mask case
        m = masks[0]
        if x.ndim == 4:
            mask_keep = m.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(-2), x.size(-1))
        elif x.ndim == 3:
            mask_keep = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        return torch.gather(x, dim=1, index=mask_keep)
    
    all_x = []
    for m in masks:
        if x.ndim == 4:
            mask_keep = m.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(-2), x.size(-1))
        elif x.ndim == 3:
            mask_keep = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, B, repeat):
    # Optimized version using repeat_interleave
    N = len(x) // B
    if N == 1:
        return x.repeat(repeat, 1, 1) if x.ndim == 3 else x.repeat(repeat, 1, 1, 1)
    x = torch.cat([torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)], dim=0)
    return x