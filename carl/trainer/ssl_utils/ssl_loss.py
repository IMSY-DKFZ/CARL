# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE-THIRD-PARTY file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

import torch
import torch.nn.functional as F

class VICRegLoss:
    def __init__(
            self, 
            sim_coeff=1.0,
            var_coeff=1.0,
            cov_coeff=0.05, 
            var_param=1.0,
            eps=1e-5
        ):
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.var_param = var_param
        self.eps = eps

    def covariance_loss(self, x):
        x = x - x.mean(dim=0)
        batch_size = x.size(0)
        dim = x.size(-1)
        
        # Create mask for non-diagonal entries
        nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)
        
        # Compute covariance matrix with shape (..., dim, dim)
        cov = torch.einsum("b...c,b...d->...cd", x, x) / (batch_size - 1)
    
        loss = cov[..., nondiag_mask].pow(2).sum(-1) / dim
        return loss.mean()

    def __call__(self, pred_token, teacher_token):
        sim_loss = F.mse_loss(pred_token, teacher_token).mean()

        z_centered = pred_token - pred_token.mean(0, keepdim=True)
        pstd_z = torch.sqrt(z_centered.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(self.var_param - pstd_z))

        cov_loss = self.covariance_loss(pred_token)

        total_loss = (
            self.sim_coeff * sim_loss +
            self.var_coeff * var_loss +
            self.cov_coeff * cov_loss
        )
        weights = self.sim_coeff + self.var_coeff + self.cov_coeff
        total_loss = total_loss / weights

        return total_loss, sim_loss, var_loss, cov_loss