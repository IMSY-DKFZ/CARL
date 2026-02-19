"""Configuration utilities for CARL training."""

from pathlib import Path
from typing import Dict, Any

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration YAML file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save.
        output_path: Path where the configuration should be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f)


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a nested value from the configuration dictionary.
    
    Args:
        config: Configuration dictionary.
        key_path: Path to the value (e.g., 'training_kwargs.batch_size' or 'training_kwargs/batch_size').
        default: Default value if the key is not found.
        
    Returns:
        Configuration value or default.
    """
    # Support both dot and slash separators for backwards compatibility
    if '/' in key_path:
        keys = key_path.split('/')
    else:
        keys = key_path.split('.')
    
    value = config
    
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
            if value is default:
                return default
        else:
            return default
    
    return value


def require_config_value(config: Dict[str, Any], key_path: str) -> Any:
    """Get a required nested value from the configuration dictionary.
    
    Raises ValueError if the key is not found.
    
    Args:
        config: Configuration dictionary.
        key_path: Path to the value (supports '/' or '.' separators).
        
    Returns:
        Configuration value.
        
    Raises:
        ValueError: If the key is not found.
    """
    sentinel = object()
    value = get_config_value(config, key_path, default=sentinel)
    if value is sentinel:
        raise ValueError(f"Required configuration key '{key_path}' not found")
    return value


class ConfigAccessor:
    """Helper class for safer configuration access with IDE autocomplete support.
    
    This class provides convenient access to nested configuration values
    with better error messages and optional default values.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the configuration accessor.
        
        Args:
            config: Configuration dictionary.
        """
        self._config = config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value with optional default.
        
        Args:
            key_path: Path to the configuration value.
            default: Default value if key is not found.
            
        Returns:
            Configuration value or default.
        """
        return get_config_value(self._config, key_path, default)
    
    def require(self, key_path: str) -> Any:
        """Get a required configuration value.
        
        Args:
            key_path: Path to the configuration value.
            
        Returns:
            Configuration value.
            
        Raises:
            ValueError: If the key is not found.
        """
        return require_config_value(self._config, key_path)
    
    @property
    def raw(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary.
        
        Returns:
            Raw configuration dictionary.
        """
        return self._config
