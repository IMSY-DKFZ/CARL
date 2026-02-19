"""CARL model training script for self-supervised learning.

This script trains the CARL (Collaborative Attentive Representation Learning) model
using self-supervised learning on spectral imagery with PyTorch Lightning.
"""

import datetime
import logging
from pathlib import Path

import torch
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from carl.trainer.ssl_trainer import Trainer
from carl.config import load_config, save_config
from carl.data_utils import create_datasets, create_dataloaders


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_checkpoint(model: Trainer, checkpoint_path: str) -> None:
    """Load a pretrained checkpoint into the model.
    
    Args:
        model: The model to load the checkpoint into.
        checkpoint_path: Path to the checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['state_dict'], 
        strict=False
    )
    if missing_keys:
        logging.info(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logging.info(f"Unexpected keys: {unexpected_keys}")


def setup_logging(config: dict) -> tuple:
    """Setup logging directory and TensorBoard logger.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (save_dir, logger, timestamp).
    """
    log_dir = config["training_kwargs"].get("log_dir", "logs/")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(log_dir) / f"carl_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration to logging directory
    save_config(config, save_dir / "config.yaml")
    
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=f"carl_{timestamp}",
        version=""
    )
    
    return save_dir, tb_logger, timestamp


def create_checkpoint_callback(config: dict, save_dir: Path) -> ModelCheckpoint:
    """Create model checkpoint callback.
    
    Args:
        config: Configuration dictionary.
        save_dir: Directory to save checkpoints.
        
    Returns:
        ModelCheckpoint callback.
    """
    metric = config["training_kwargs"]["monitor_metric"]
    filename = f"{{epoch:02d}}_{{{metric}:.4f}}"
    return ModelCheckpoint(
        monitor=config["training_kwargs"]["monitor_metric"],
        dirpath=save_dir,
        filename=filename,
        save_top_k=1,
        mode="max",
        save_last=True,
    )


def main(config_path: str) -> None:
    """Main training loop.
    
    Args:
        config_path: Path to the configuration YAML file.
    """
    # Load configuration
    config = load_config(config_path)
    logger.info("Configuration loaded successfully")
    logger.info(f"Config: {config}")
    
    # Initialize model
    model = Trainer(config)
    model.to(dtype=torch.float32)
    
    # Load checkpoint if provided
    checkpoint_path = config["training_kwargs"].get("ckpt_path", None)
    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        load_checkpoint(model, checkpoint_path)
    
    # Create datasets and dataloaders
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
    # Setup logging
    save_dir, tb_logger, timestamp = setup_logging(config)
    logger.info(f"Logging to {save_dir}")
    
    # Create callbacks
    checkpoint_callback = create_checkpoint_callback(config, save_dir)
    
    # Configure PyTorch Lightning trainer
    lightning_kwargs = config.get("lightning_kwargs", {})
    lightning_kwargs.update({
        "logger": tb_logger,
        "callbacks": [checkpoint_callback]
    })
    
    trainer = PLTrainer(**lightning_kwargs)

    logging.info("Starting training...")
    logging.info(f"Training configuration: {config}")
    logging.info(f"Training dataset size: {len(train_dataloader)}")
    logging.info(f"Validation dataset size: {len(val_dataloader)}")
    
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train CARL model for semantic segmentation."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()
    
    main(args.config)