"""Data utilities for CARL training."""

import importlib
from typing import Tuple, Dict, Any

from torch.utils.data import DataLoader, Dataset

from carl.data.dataloader import MultiDataLoader

def load_dataset_class(dataset_name: str) -> type:
    """Dynamically load a dataset class by name.
    
    Args:
        dataset_name: Name of the dataset class (e.g., 'GeoBenchSeg').
        
    Returns:
        The dataset class.
        
    Raises:
        ModuleNotFoundError: If the dataset module cannot be found.
        AttributeError: If the dataset class cannot be found in the module.
    """
    try:
        module = importlib.import_module(f"carl.data.{dataset_name}")
        dataset_class = getattr(module, dataset_name)
        return dataset_class
    except (ModuleNotFoundError, AttributeError) as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")


def create_datasets(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
    """Create train, validation, and test datasets from configuration.
    
    Args:
        config: Configuration dictionary containing data_kwargs.
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    data_config = config["data_kwargs"]
    
    if isinstance(data_config["train_dataset"], dict):
        kwargs = data_config["train_dataset"]
        dataset_name = kwargs.pop("name")
        dataset_class = load_dataset_class(dataset_name)
        train_dataset = dataset_class(
            cfg=config,
            **kwargs,
        )
    elif isinstance(data_config["train_dataset"], list):
        train_dataset = []
        for ds_cfg in data_config["train_dataset"]:
            dataset_name = ds_cfg.pop("name")
            dataset_class = load_dataset_class(dataset_name)
            ds = dataset_class(
                cfg=config,
                **ds_cfg,
            )
            train_dataset.append(ds)

    if isinstance(data_config["val_dataset"], dict):
        kwargs = data_config["val_dataset"]
        dataset_name = kwargs.pop("name")
        dataset_class = load_dataset_class(dataset_name)
        val_dataset = dataset_class(
            cfg=config,
            **kwargs,
        )
    elif isinstance(data_config["val_dataset"], list):
        val_dataset = []
        for ds_cfg in data_config["val_dataset"]:
            dataset_name = ds_cfg.pop("name")
            dataset_class = load_dataset_class(dataset_name)
            ds = dataset_class(
                cfg=config,
                **ds_cfg,
            )
            val_dataset.append(ds)
    
    if "test_dataset" in data_config:
        kwargs = data_config["test_dataset"]
        dataset_name = kwargs.pop("name")
        dataset_class = load_dataset_class(dataset_name)
        test_dataset = dataset_class(
            cfg=config,
            **kwargs,
        )
    else:
        test_dataset = None
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.
    
    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        config: Configuration dictionary containing training_kwargs.
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader).
    """
    training_config = config["training_kwargs"]
    batch_size = training_config["batch_size"]
    num_workers = training_config["num_workers"]
    
    if isinstance(train_dataset, list):
        train_dataloader = MultiDataLoader(
            [
                DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                )
                for ds in train_dataset
            ],
            seed=42,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    val_dataset = [val_dataset] if not isinstance(val_dataset, list) else val_dataset

    val_dataloader = [
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ) for ds in val_dataset
    ]
    
    if test_dataset is None:
        test_dataloader = val_dataloader
    else:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_dataloader, val_dataloader, test_dataloader
