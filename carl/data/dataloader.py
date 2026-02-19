# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0 
# found in the LICENSE file in the root directory of this source tree.

import random
from typing import List


class MultiDataLoader:
    """
    Combines multiple dataloaders, truncating all to the smallest length,
    with randomized sampling order.
    """
    
    def __init__(
        self, 
        dataloaders: List,
        seed: int = None,
        min_lengths: bool = True,
    ):
        """
        Args:
            dataloaders: List of PyTorch DataLoader objects
            mode: Sampling strategy
                - 'shuffle_once': Pre-shuffle the order at epoch start (recommended)
                - 'random': Randomly pick a dataloader for each batch
            seed: Random seed for reproducibility
            min_lengths: Whether to truncate all dataloaders to the length of the
                         shortest one (default: True)
        """
        if not dataloaders:
            raise ValueError("Must provide at least one dataloader")
        
        self.dataloaders = dataloaders
        self.num_loaders = len(dataloaders)
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Calculate minimum length (truncation point)
        lengths = [len(dl) for dl in self.dataloaders]
        if min_lengths:
            self.lengths = [min(lengths)] * self.num_loaders
        else:
            self.lengths = lengths
        
    def __len__(self):
        """
        Total batches = min_length * num_loaders
        Each loader contributes exactly min_length batches
        """
        return sum(self.lengths)
        
    def __iter__(self):
        """
        Create a shuffled schedule of which loader to sample from.
        Each loader appears exactly min_length times.
        """
        # Create schedule: each loader index appears min_length times
        # e.g., [0,0,0,1,1,1,2,2,2] for 3 loaders with min_length=3
        dataloader_schedule = []
        for loader_idx in range(self.num_loaders):
            dataloader_schedule.extend([loader_idx] * self.lengths[loader_idx])
        
        # Shuffle the schedule for randomized order
        self.rng.shuffle(dataloader_schedule)
        
        # Create iterators for all dataloaders
        iterators = [iter(dl) for dl in self.dataloaders]
        
        # Follow the shuffled schedule
        for loader_idx in dataloader_schedule:
            try:
                batch = next(iterators[loader_idx])
            except Exception as e:
                raise RuntimeError(f"Error fetching batch from dataloader {loader_idx}: {e}")
            yield batch
