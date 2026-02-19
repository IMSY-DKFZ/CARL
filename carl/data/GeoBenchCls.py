"""GeoBench semantic segmentation dataset for spectral imagery.

This module provides a PyTorch Dataset implementation for loading GeoBench
spectral imagery data with associated wavelength information and labels.
"""

from typing import Tuple, Optional, Dict, Any

import geobench as gb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# Data normalization constants
NORMALIZATION_EPSILON = 1e-6


class GeoBenchCls(Dataset):
    """GeoBench classification dataset.
    
    Loads spectral imagery from GeoBench with associated wavelength information
    and labels. Implements lazy loading with caching.
    
    Attributes:
        gb_data: GeoBench dataset instance.
        n_classes: Number of semantic classes.
        files: Cache for lazy-loaded samples.
    """
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train', 
        cfg: Optional[Dict[str, Any]] = None
    ):
        """Initialize the GeoBench dataset.
        
        Args:
            path: Path to the GeoBench dataset.
            split: Dataset split ('train', 'val', or 'test').
            cfg: Configuration dictionary containing model_kwargs with n_classes.
            
        Raises:
            KeyError: If cfg doesn't contain required configuration keys.
        """
        super().__init__()
        
        if cfg is None:
            raise ValueError("Configuration (cfg) is required")
        
        self.gb_data = gb.dataset.GeobenchDataset(root_dir, split=split)
        self.cfg = cfg
        self.n_classes = cfg['model_kwargs']['n_classes']
        
        # Lazy loading: cache for loaded samples
        self.files = [None] * len(self.gb_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a dataset sample.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Tuple of (image, wavelengths, label) tensors.
        """
        if self.files[idx] is None:
            self.files[idx] = self._load_sample(idx)
        return self.files[idx]

    def _load_sample(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and preprocess a single sample.
        
        Args:
            index: Index of the sample to load.
            
        Returns:
            Tuple of (image, wavelengths, label) tensors.
            
        Raises:
            ValueError: If labels are outside the valid range [0, n_classes-1].
            AttributeError: If label doesn't have 'data' attribute.
        """
        sample = self.gb_data[index]

        # Extract spectral bands and wavelengths
        images, wavelengths = self._extract_spectral_bands(sample)
        
        # Stack bands into a single image tensor
        img = np.stack(images, axis=-1).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))  # (C, H, W)
        wavelengths = np.array(wavelengths).astype(np.float32)

        # Extract and validate labels
        label = self._extract_and_validate_label(sample)

        # Normalize image
        img = self._normalize_image(img)

        # Convert to tensors
        img_tensor = torch.from_numpy(img).float()
        wavelengths_tensor = torch.from_numpy(wavelengths).float()
        label_tensor = torch.tensor(label, dtype=torch.int64)

        return img_tensor, wavelengths_tensor, label_tensor

    def _extract_spectral_bands(
        self, 
        sample: Any
    ) -> Tuple[list, list]:
        """Extract spectral bands and wavelengths from a sample.
        
        Args:
            sample: A sample from GeoBench dataset.
            
        Returns:
            Tuple of (image_list, wavelength_list).
        """
        images = []
        wavelengths = []

        for band in sample.bands:
            if hasattr(band.band_info, 'wavelength'):
                wavelength = band.band_info.wavelength
                if wavelength is not None:
                    images.append(band.data)
                    wavelengths.append(wavelength)

        return images, wavelengths

    def _extract_and_validate_label(self, sample: Any) -> np.ndarray:
        """Extract and validate labels.
        
        Args:
            sample: A sample from GeoBench dataset.
            
        Returns:
            Label array.
            
        Raises:
            AttributeError: If label doesn't have 'data' attribute.
            ValueError: If labels are outside valid range [0, n_classes-1].
        """
        label = sample.label
        
        # Validate label range
        min_label, max_label = np.min(label), np.max(label)
        if min_label < 0 or max_label >= self.n_classes:
            raise ValueError(
                f"Labels should be in range [0, {self.n_classes - 1}]. "
                f"Found labels in range [{min_label}, {max_label}]."
            )
        
        return label

    @staticmethod
    def _normalize_image(img: np.ndarray) -> np.ndarray:
        """Normalize image using mean and standard deviation.
        
        Args:
            img: Image array of shape (C, H, W).
            
        Returns:
            Normalized image array.
        """
        mean = img.mean()
        std = img.std()
        return (img - mean) / (std + NORMALIZATION_EPSILON)

    def __len__(self) -> int:
        """Get dataset length.
        
        Returns:
            Number of samples in the dataset.
        """
        return len(self.gb_data)