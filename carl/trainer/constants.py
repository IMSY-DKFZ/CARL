"""Constants for CARL self-supervised learning trainer.

This module contains configuration constants used throughout the SSL training process.
"""

# Augmentation parameters
DEFAULT_CROP_SCALE = (0.8, 1.0)
DEFAULT_DROPOUT_PROB = 0.5

# Numerical stability
EPSILON = 1e-5

# KNN validation parameters
KNN_K_NEIGHBORS = 3
KNN_TEMPERATURE = 0.1
