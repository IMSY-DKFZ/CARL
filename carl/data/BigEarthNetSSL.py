import os
from typing import Tuple

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

WLENS_DICT = {
    "B01": 442.7,
    "B02": 492.4,
    "B03": 559.8,
    "B04": 664.6,
    "B05": 704.1,
    "B06": 740.5,
    "B07": 782.8,
    "B08": 832.8,
    "B8A": 864.7,
    "B09": 945.1,
    "B10": 1373.5,
    "B11": 1613.7,
    "B12": 2202.4,
}


class BigEarthNetSSL(Dataset):
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
        **kwargs,
    ) -> None:
        """
        Initialize a new SpectralEarthDataset instance.

        """
        self.root = root_dir
        self.patch_paths =  []

        # Build the patch_paths dictionary
        folders = os.listdir(root_dir)
        for folder in folders:
            folder_path = os.path.join(root_dir, folder)
            subfolders = os.listdir(folder_path)
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder)
                self.patch_paths.append(subfolder_path)

    def load_patch(self, index: int) -> Tensor:
        """
        Load and preprocess a single patch.

        Args:
            index: Index of the patch to load.
        Returns:
            image: Tensor of shape (C, H, W) with spectral image data.
        """

        folder_path = self.patch_paths[index]
        suffixes = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
                    "B09", "B11", "B12"]
        prefix = os.path.basename(folder_path)
        band_files = [
            os.path.join(folder_path, f"{prefix}_{suffix}.tif")
            for suffix in suffixes
        ]
        bands = []
        wavelengths = []
        for i, band_file in enumerate(band_files):
            try:
                with rasterio.open(band_file) as f:
                    band_data = f.read(1)
            except Exception as e:
                continue
            band_tensor = torch.from_numpy(band_data.astype(np.uint16))
            band_tensor = band_tensor / 10000.0  # Scale reflectance values
            band_tensor = band_tensor.unsqueeze(0)
            band_files = F.interpolate(
                band_tensor.unsqueeze(0),
                size=(128, 128),
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            bands.append(band_files)

            wlens = WLENS_DICT[suffixes[i]]
            wavelengths.append(wlens)
        image = torch.cat(bands, dim=0)  # (C, H, W)
        wavelengths = torch.tensor(wavelengths, dtype=torch.float32)  # (C,)
        return image, wavelengths

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
        image, wlens = self.load_patch(index)

        image = (image - image.mean()) / (image.std() + 1e-6)

        wlens = wlens / 1000.0  # Convert to micrometers

        return image, wlens

    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return len(self.patch_paths)