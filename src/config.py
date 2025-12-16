"""
Configuration system for the Superconductor Analysis and Prediction project.

This module provides a centralized configuration system that can be used
throughout the project. It loads configuration from environment variables,
config files, and command-line arguments.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the project."""
    
    DEFAULT_CONFIG = {
        # API keys and external services
        "mp_api_key": "DjKx0q7YivC5u73uKFIVPif813v7InYq",  # Materials Project API key
        
        # Data collection settings
        "data_dir": "data",
        "ti_compounds_file": "ti_compounds.csv",
        "ti_structures_file": "ti_compounds_structures.json",
        "superconductors_file": "superconductors.csv",
        "superconductors_structures_file": "superconductors_structures.json",
        
        # Structure visualization settings
        "structures_dir": "structures",
        "visualization_dir": "visualization",
        "generate_3d_views": True,
        "image_format": "png",
        "image_dpi": 300,
        
        # Model settings
        "models_dir": "models",
        "model_file": "superconductor_gnn.pt",
        "hidden_dim": 128,
        "learning_rate": 0.0003,
        "batch_size": 16,
        "num_epochs": 500,
        "test_split": 0.2,
        "val_split": 0.1,
        "early_stopping_patience": 50,
        
        # GPU settings
        "use_gpu": torch.cuda.is_available(),
        "gpu_memory_fraction": 0.9,
        "gpu_device": 0,  # Default to first GPU
        "mixed_precision": True,  # Use mixed precision training for faster computation
        "cudnn_benchmark": True,  # Use cuDNN benchmark for faster training
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration system.
        
        Args:
            config_file: Path to a JSON configuration file (optional)
        """
        # Start with default configuration
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Update from config file if provided
        if config_file:
            self._load_from_file(config_file)
        
        # Update from environment variables
        self._load_from_env()
        
        # Ensure paths are Path objects
        self._process_paths()
        
        # Configure GPU settings
        self._configure_gpu()
        
        logger.info("Configuration loaded successfully")
    
    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from a JSON file."""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_file}")
                return
                
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                
            # Update configuration with file values
            self.config.update(file_config)
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Map of environment variables to config keys
        env_mapping = {
            "MP_API_KEY": "mp_api_key",
            "SUPERCONDUCTOR_DATA_DIR": "data_dir",
            "SUPERCONDUCTOR_MODELS_DIR": "models_dir",
            "SUPERCONDUCTOR_USE_GPU": "use_gpu",
            "SUPERCONDUCTOR_GPU_DEVICE": "gpu_device",
            "SUPERCONDUCTOR_MIXED_PRECISION": "mixed_precision",
            "SUPERCONDUCTOR_GPU_MEMORY_FRACTION": "gpu_memory_fraction",
        }
        
        # Update config from environment variables
        for env_var, config_key in env_mapping.items():
            if env_var in os.environ:
                # Handle boolean values
                if config_key in ("use_gpu", "mixed_precision", "cudnn_benchmark"):
                    self.config[config_key] = os.environ[env_var].lower() in ('true', '1', 'yes')
                # Handle numeric values
                elif config_key in ("gpu_memory_fraction", "learning_rate", "batch_size", "num_epochs", "gpu_device"):
                    try:
                        if config_key in ("gpu_device"):
                            self.config[config_key] = int(os.environ[env_var])
                        else:
                            self.config[config_key] = float(os.environ[env_var])
                    except ValueError:
                        logger.warning(f"Invalid value for {env_var}: {os.environ[env_var]}")
                # Handle string values
                else:
                    self.config[config_key] = os.environ[env_var]
    
    def _process_paths(self) -> None:
        """Convert path strings to Path objects and ensure directories exist."""
        path_keys = ["data_dir", "structures_dir", "visualization_dir", "models_dir"]
        
        for key in path_keys:
            if key in self.config and self.config[key]:
                # Convert to Path object
                self.config[key] = Path(self.config[key])
    
    def _configure_gpu(self) -> None:
        """Configure GPU settings based on configuration."""
        use_gpu = self.config.get("use_gpu", False)
        
        if use_gpu:
            if not torch.cuda.is_available():
                logger.warning("GPU usage requested but CUDA is not available. Falling back to CPU.")
                self.config["use_gpu"] = False
                return
            
            # Set the GPU device
            gpu_device = self.config.get("gpu_device", 0)
            if gpu_device >= torch.cuda.device_count():
                logger.warning(f"Requested GPU device {gpu_device} is not available. "
                              f"Using device 0 instead.")
                gpu_device = 0
                self.config["gpu_device"] = 0
            
            # Set the device
            torch.cuda.set_device(gpu_device)
            
            # Configure cuDNN for better performance
            if self.config.get("cudnn_benchmark", True):
                torch.backends.cudnn.benchmark = True
            
            # Log GPU information
            gpu_name = torch.cuda.get_device_name(gpu_device)
            gpu_memory = torch.cuda.get_device_properties(gpu_device).total_memory / (1024**3)  # GB
            logger.info(f"Using GPU {gpu_device}: {gpu_name} with {gpu_memory:.2f} GB memory")
            
            # Set memory fraction if using CUDA
            memory_fraction = self.config.get("gpu_memory_fraction", 0.9)
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                logger.info(f"Set GPU memory fraction to {memory_fraction}")
            
            # Enable mixed precision if requested
            if self.config.get("mixed_precision", True) and torch.cuda.is_available():
                if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                    logger.info("Mixed precision training enabled")
                else:
                    logger.warning("Mixed precision requested but not supported by PyTorch version")
        else:
            logger.info("GPU usage is disabled. Using CPU for computation.")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key
            default: Default value if key is not found
            
        Returns:
            The configuration value or default
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key
            value: The value to set
        """
        self.config[key] = value
        
        # Reconfigure GPU settings if relevant
        if key in ("use_gpu", "gpu_device", "cudnn_benchmark", "mixed_precision", "gpu_memory_fraction"):
            self._configure_gpu()
    
    def save(self, config_file: str) -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            config_file: Path to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert Path objects to strings for JSON serialization
            serializable_config = {}
            for key, value in self.config.items():
                if isinstance(value, Path):
                    serializable_config[key] = str(value)
                else:
                    serializable_config[key] = value
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
            
            # Write to file
            with open(config_file, 'w') as f:
                json.dump(serializable_config, f, indent=2)
                
            logger.info(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary.
        
        Returns:
            A copy of the configuration dictionary
        """
        return self.config.copy()
    
    def get_device(self) -> torch.device:
        """
        Get the PyTorch device based on configuration.
        
        Returns:
            PyTorch device (cuda or cpu)
        """
        if self.config.get("use_gpu", False) and torch.cuda.is_available():
            return torch.device(f"cuda:{self.config.get('gpu_device', 0)}")
        else:
            return torch.device("cpu")

# Create a global configuration instance
config = Config()

def load_config(config_file: Optional[str] = None) -> Config:
    """
    Load configuration from a file and return a Config instance.
    
    Args:
        config_file: Path to a JSON configuration file
        
    Returns:
        A Config instance
    """
    global config
    config = Config(config_file)
    return config