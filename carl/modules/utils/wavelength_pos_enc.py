# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0 
# found in the LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, sigma):
        super().__init__()
        self.d_hid = d_hid
        self.sigma = sigma
   
        rng_state = torch.get_rng_state()
        temp_gen = torch.Generator()
        seed = 0
        temp_gen.manual_seed(seed)  # Fixed seed for reproducible sampling    
        B_gauss = 2 * torch.tensor(math.pi) * (torch.randn((d_hid // 2), generator=temp_gen) * sigma + 0)
        torch.set_rng_state(rng_state)
        B_gauss = torch.stack([B_gauss[i//2] for i in range(d_hid)], dim=0)
        self.register_buffer("dims", B_gauss, persistent=True)

    def get_position_angle_vec(self, position):
        position = position.unsqueeze(-1)
        dims = self.dims.unsqueeze(0).unsqueeze(0)
        out = position * dims
        return out

    def forward(self, w) -> torch.Tensor:
        if self.sigma == 0:
            b, c = w.shape
            d = self.d_hid
            return torch.zeros(b, c, d, device=w.device, dtype=w.dtype)
        wlens_encoded = self.get_position_angle_vec(w)  # No need to clone here initially

        sin_encoded = torch.sin(wlens_encoded[..., 0::2])  # Compute sine out-of-place
        cos_encoded = torch.cos(wlens_encoded[..., 1::2])  # Compute cosine out-of-place

        # Construct the output tensor by combining sine and cosine results
        wlens_encoded_out = torch.empty_like(wlens_encoded)  # Create a new tensor with the same shape
        wlens_encoded_out[..., 0::2] = sin_encoded
        wlens_encoded_out[..., 1::2] = cos_encoded

        return wlens_encoded_out