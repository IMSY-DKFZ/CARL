# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0 
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.utils.data


class SpectralMasking:
    def __init__(
        self, 
        pred_mask_scale=(0.2, 0.8),
        npred=2,
        min_keep=0.05
    ):
        super().__init__()
        self.pred_mask_scale = pred_mask_scale
        self.npred = npred
        self.min_keep = min_keep  # minimum ratio of spectral channels to keep unmasked
        self._itr_counter = -1

    def step(self):
        i = self._itr_counter
        i += 1
        self._itr_counter = i
        return i

    def _sample_block_mask(self, b_size, sequence_length, generator, bounds=None, sample_pred_mask=False, min_keep=None, device='cpu'):
        h = b_size
        min_keep = max(int(min_keep * sequence_length), 3)

        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            if bounds is not None:
                top = torch.randint(bounds[0], bounds[1] - h + 1, (1,), generator=generator)
            else:
                top = torch.randint(0, sequence_length - h + 1, (1,), generator=generator)
            mask = torch.zeros(sequence_length, dtype=torch.int32, device=device)
            mask[top : top + h] = 1
            # -- Constrain mask to a set of acceptable regions
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > min_keep if not sample_pred_mask else sequence_length - len(mask) > min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                if tries > 15:
                    raise ValueError(
                        f"Mask generator failed to find a valid mask. Block size: {h}, sequence length:"
                        f" {sequence_length}, min_keep: {min_keep}, tries: {tries}"
                    )
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones(sequence_length, dtype=torch.int32, device=device)
        mask_complement[top : top + h] = 0
        # --
        return mask, mask_complement
    
    def sample(self, generator, batch_size, sequence_length, device='cpu'):
        n_pred = self.npred

        # For multispectral images, we do not mask too much.
        if sequence_length < 20:
            _rand = torch.rand(1, generator=generator).item() * 2 + 2
            p_size = int(_rand)
            n_pred = 1
        else:
            a,b = self.pred_mask_scale
            b = b - a
            _rand = a * sequence_length
            _rand += torch.rand(1, generator=generator).item() * b * sequence_length
            p_size = int(_rand)

        collated_masks_pred, collated_masks_enc = [], []

        total_mask_indices = torch.arange(sequence_length, device=device)
        min_keep_pred = sequence_length
        min_keep_enc = sequence_length
        for i in range(batch_size):
            masks_p, masks_C = [], []
            for _ in range(n_pred):
                mask, mask_C = self._sample_block_mask(
                    p_size,
                    sequence_length,
                    min_keep=self.min_keep,
                    sample_pred_mask=True,
                    generator=generator,
                    device=device
                )
                mask = torch.sort(mask).values
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            mask_C_multiplied = torch.ones_like(mask_C)
            for mask_C in masks_C:
                mask_C_multiplied *= mask_C

            total_mask = torch.ones(sequence_length, dtype=torch.int32, device=device).bool()
            
            encoder_mask = total_mask ^ ~(mask_C_multiplied.bool())
            masks_e = torch.nonzero(encoder_mask.flatten()).squeeze()
            masks_e = torch.sort(masks_e).values

            try:
                assert len(masks_e) > 0, f"Error in mask generation, {masks_e}, {p_size}, {_rand} \
                    {mask_C_multiplied}, {total_mask}, {encoder_mask}"
                assert len(masks_p[0]) > 0, f"Error in mask generation, {masks_p[0]}, {p_size}, {_rand}"
                assert set(list(masks_e.numpy()) + list(torch.cat(masks_p).numpy())) == set(list(total_mask_indices.numpy())), (
                    f"Error in mask generation, {set(list(masks_e.numpy()) + list(torch.cat(masks_p).numpy()))} !="
                    f" {set(list(total_mask_indices.numpy()))}"
                )
            except Exception as e:
                logging.info(f"Error in mask generation, {masks_e}, {p_size}, {_rand} \
                    {mask_C_multiplied}, {total_mask}, {encoder_mask}")

            min_keep_enc = min(min_keep_enc, len(masks_e))
            collated_masks_enc.append([masks_e])
            
        return collated_masks_enc, collated_masks_pred, min_keep_enc, min_keep_pred


    def __call__(self, batch_size, sequence_length, device="cpu"):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        min_keep_pred = sequence_length
        min_keep_enc = sequence_length
        collated_masks_enc, collated_masks_pred, min_keep_enc, min_keep_pred = \
            self.sample(g, batch_size, sequence_length, device=device)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_masks_enc, collated_masks_pred
