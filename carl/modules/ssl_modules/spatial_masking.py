# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE-THIRD-PARTY file in the root directory of this source tree.
#
# References:
#   https://github.com/facebookresearch/vjepa2/blob/main/src/masks/multiseq_multiblock3d.py

import math

import torch


class SpatialMasking:
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=14,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
    ):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = -1

    def step(self):
        i = self._itr_counter
        i += 1
        self._itr_counter = i
        return i

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None, sample_pred_mask=False, device='cpu'):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """Helper to restrict given mask to a set of acceptable regions"""
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        sequence_length = self.height * self.width
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h + 1, (1,))
            left = torch.randint(0, self.width - w + 1, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32, device=device)
            mask[top : top + h, left : left + w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep if not sample_pred_mask else sequence_length - len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                if tries == 15:
                    raise ValueError(
                        f"Mask generator failed to find a valid mask. Image size: {self.height}x{self.width}, mask"
                        f" size: {h}x{w}, min_keep: {self.min_keep},                                     mask length:"
                        f" {len(mask)}"
                    )
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32, device=device)
        mask_complement[top : top + h, left : left + w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch_size, device="cpu"):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        """
        B = batch_size

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(generator=g, scale=self.pred_mask_scale, aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(generator=g, scale=self.enc_mask_scale, aspect_ratio_scale=(1.0, 1.0))
        # e_size = (16,16)
        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size, sample_pred_mask=True, device=device)
                mask = torch.sort(mask).values
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C

            mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions, sample_pred_mask=False, device=device)
            assert mask.max() < self.height * self.width, \
                f"{mask.max()} >= {self.height * self.width}"
            mask = torch.sort(mask).values
            min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append([mask])

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred
