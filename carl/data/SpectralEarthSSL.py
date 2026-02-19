# Code adapted from:
# https://github.com/AABNassim/spectral_earth/blob/main/src/datasets/spectral_earth.py

import os
import random
from typing import Tuple, Optional

import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset

DEFAULT_DTYPE_OUTPUT = torch.float16

class SpectralEarthSSL(Dataset):
    """Spectral Earth Dataset for hyperspectral foundation model pretraining.
    
    This dataset handles the large-scale hyperspectral data collection from EnMAP
    for self-supervised pretraining. It supports both single and multi-view loading
    modes for contrastive learning approaches.
    
    Each patch has the following properties:
    - 128 x 128 pixels
    - Single multispectral GeoTIFF file
    - 202 spectral bands (EnMAP)
    - 30 m spatial resolution
    """

    def __init__(
        self,
        root_dir: str,
        wlens_path: str,
        n_channels: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a new SpectralEarthDataset instance.

        """
        self.root = root_dir
        self.patch_paths = {}

        # Build the patch_paths dictionary using os.scandir for better performance
        with os.scandir(root_dir) as patch_entries:
            for patch_entry in patch_entries:
                if patch_entry.is_dir():
                    image_files = [
                        entry.path
                        for entry in os.scandir(patch_entry.path)
                        if entry.is_file() and entry.name.endswith(".tif")
                    ]
                    if image_files:
                        self.patch_paths[patch_entry.name] = image_files

        wlens = np.loadtxt(wlens_path).astype(np.float32)
        self.wavelengths = torch.from_numpy(wlens)
        self.n_channels = n_channels

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Return a data sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A tuple containing:
            - image: Tensor of shape (C, H, W) with spectral image data.
            - wlens: Tensor of shape (C,) with corresponding wavelengths in micrometers.
        """
        patch_id = list(self.patch_paths.keys())[index]
        image_paths = self.patch_paths[patch_id]

        chosen_path = random.choice(image_paths)
        with rasterio.open(chosen_path) as f:
            image = torch.from_numpy(f.read().astype(np.int16))

        image = image.permute(2, 0, 1)  # (C, H, W)
        image = image.to(torch.float32)
        image = image / 10000.0
        image = (image - image.mean()) / (image.std() + 1e-6)

        wlens = self.wavelengths / 1000.0  # Convert to micrometers

        if self.n_channels is not None:
            random_indices = torch.randperm(image.shape[0])[: self.n_channels]
            image = image[random_indices]
            wlens = wlens[random_indices]

        # Convert to specified output dtype
        image = image.to(dtype=DEFAULT_DTYPE_OUTPUT)
        wlens = wlens.to(dtype=DEFAULT_DTYPE_OUTPUT)

        return image, wlens

    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return len(self.patch_paths)