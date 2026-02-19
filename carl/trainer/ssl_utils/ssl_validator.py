"""Utility functions for CARL SSL trainer.

This module provides helper functions for self-supervised learning tasks,
including KNN validation, momentum scheduling, and loss computation.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from carl.trainer.constants import (
    KNN_K_NEIGHBORS,
    KNN_TEMPERATURE,
)


class KNNValidator:
    """Helper class for k-nearest neighbors validation.
    
    This class manages the feature bank construction and performs
    weighted voting for KNN-based validation.
    """
    
    def __init__(
        self,
        num_classes: int,
        k: int = KNN_K_NEIGHBORS,
        temperature: float = KNN_TEMPERATURE,
    ):
        """Initialize the KNN validator.
        
        Args:
            k: Number of nearest neighbors to consider.
            temperature: Temperature for similarity weighting.
            num_classes: Number of classes for classification.
        """
        self.k = k
        self.temperature = temperature
        self.num_classes = num_classes
        self.feature_bank = []
        self.feature_labels = []
    
    def add_to_feature_bank(
        self,
        batch,
        image_size: int,
        model: torch.nn.Module,
    ) -> None:

        img, wavelengths, labels = batch
        size = (image_size, image_size)
        
        with torch.no_grad():
            if img.size(2) != size[0] or img.size(3) != size[1]:
                img = F.interpolate(
                    img, size=size, mode="bilinear", align_corners=False
                )
            
            # Extract spatial tokens
            spatial_token, _ = model.forward_teacher(img, wavelengths)
            spatial_token = spatial_token.mean(1)
            spatial_token = F.normalize(spatial_token, dim=-1)
            
            self.feature_bank.append(spatial_token)
            self.feature_labels.append(labels)
    
    def validate(self, features: Tensor, labels: Tensor) -> float:
        """Perform KNN validation on features.
        
        Args:
            features: Feature vectors to validate.
            labels: Ground truth labels.
            
        Returns:
            Validation accuracy.
        """
        if self.feature_bank is None:
            raise RuntimeError("Feature bank not built. Call build_feature_bank first.")
        
        if isinstance(self.feature_bank, list):
            self.feature_bank = torch.cat(self.feature_bank, dim=0)
            self.feature_labels = torch.cat(self.feature_labels, dim=0)

        # Normalize features
        features = F.normalize(features, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = features @ self.feature_bank.T
        
        # Get top-k similar samples
        similarity_weights, similarity_indices = similarity_matrix.topk(k=self.k, dim=-1)
        similarity_labels = self.feature_labels[similarity_indices]
        
        # Weighted voting
        similarity_weights = (similarity_weights / self.temperature).exp()
        one_hot = torch.zeros(
            features.size(0) * self.k,
            self.num_classes,
            device=features.device
        )
        one_hot.scatter_(1, similarity_labels.view(-1, 1), 1)
        one_hot = one_hot.view(features.size(0), self.k, -1)
        
        prediction_scores = (one_hot * similarity_weights.unsqueeze(-1)).sum(1)
        predictions = prediction_scores.argmax(dim=-1)
        
        accuracy = (predictions == labels).float().mean()
        return accuracy.item()
    
    def clear(self) -> None:
        """Clear the feature bank to free memory."""
        self.feature_bank = []
        self.feature_labels = []
