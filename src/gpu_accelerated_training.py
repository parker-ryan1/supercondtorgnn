"""
GPU-accelerated training utilities for superconductor prediction.

This module provides utilities for GPU-accelerated training of GNN models,
including mixed precision training, memory optimization, and multi-GPU support.
"""

import torch
import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union
from tqdm import tqdm
import time

# Configure logging
logger = logging.getLogger(__name__)

class GPUAccelerator:
    """
    Utility class for GPU-accelerated training.
    
    This class provides methods for optimizing GPU usage during training,
    including mixed precision training, memory management, and multi-GPU support.
    """
    
    def __init__(self, 
                 use_gpu: bool = True, 
                 gpu_device: int = 0,
                 mixed_precision: bool = True,
                 memory_fraction: float = 0.9):
        """
        Initialize the GPU accelerator.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            gpu_device: GPU device index to use
            mixed_precision: Whether to use mixed precision training
            memory_fraction: Fraction of GPU memory to use
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.gpu_device = min(gpu_device, torch.cuda.device_count() - 1) if self.use_gpu else 0
        self.mixed_precision = mixed_precision and self.use_gpu
        self.memory_fraction = memory_fraction
        
        # Set up device
        if self.use_gpu:
            self.device = torch.device(f"cuda:{self.gpu_device}")
            self._setup_gpu()
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for computation")
        
        # Set up mixed precision training
        self.scaler = None
        if self.mixed_precision and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled with gradient scaling")
    
    def _setup_gpu(self):
        """Set up GPU for optimal performance."""
        try:
            # Set the device
            torch.cuda.set_device(self.gpu_device)
            
            # Configure cuDNN for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Set memory fraction if available
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # Empty cache
            torch.cuda.empty_cache()
            
            # Log GPU information
            gpu_name = torch.cuda.get_device_name(self.gpu_device)
            gpu_memory = torch.cuda.get_device_properties(self.gpu_device).total_memory / (1024**3)  # GB
            logger.info(f"Using GPU {self.gpu_device}: {gpu_name} with {gpu_memory:.2f} GB memory")
            
        except Exception as e:
            logger.error(f"Error setting up GPU: {str(e)}")
            self.use_gpu = False
            self.device = torch.device("cpu")
            logger.info("Falling back to CPU")
    
    def log_memory_usage(self, prefix: str = ""):
        """
        Log GPU memory usage.
        
        Args:
            prefix: Prefix for the log message
        """
        if not self.use_gpu:
            return
        
        try:
            allocated = torch.cuda.memory_allocated(self.gpu_device) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(self.gpu_device) / (1024**3)  # GB
            max_allocated = torch.cuda.max_memory_allocated(self.gpu_device) / (1024**3)  # GB
            
            logger.info(f"{prefix} GPU Memory: "
                       f"Allocated: {allocated:.2f} GB, "
                       f"Reserved: {reserved:.2f} GB, "
                       f"Max Allocated: {max_allocated:.2f} GB")
        except Exception as e:
            logger.warning(f"Error logging GPU memory usage: {str(e)}")
    
    def to_device(self, data):
        """
        Move data to the appropriate device.
        
        Args:
            data: Data to move to device
            
        Returns:
            Data on the appropriate device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif hasattr(data, 'to'):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_device(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.to_device(item) for item in data)
        else:
            return data
    
    def train_epoch(self, 
                   model: torch.nn.Module, 
                   loader: torch.utils.data.DataLoader, 
                   optimizer: torch.optim.Optimizer, 
                   criterion: torch.nn.Module,
                   epoch: int,
                   total_epochs: int) -> float:
        """
        Train the model for one epoch with GPU acceleration.
        
        Args:
            model: Model to train
            loader: Data loader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch number
            total_epochs: Total number of epochs
            
        Returns:
            Average loss for the epoch
        """
        model.train()
        total_loss = 0
        
        # Set up progress bar
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} (Train)")
        
        # Mixed precision context manager
        amp_context = torch.cuda.amp.autocast() if self.mixed_precision else nullcontext()
        
        # Training loop
        for batch in progress_bar:
            # Move batch to device
            batch = self.to_device(batch)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            with amp_context:
                # Extract features
                if hasattr(batch, 'global_features'):
                    global_features = batch.global_features
                else:
                    global_features = None
                
                # Forward pass
                out = model(batch.x, batch.edge_index, batch.batch, 
                           edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                           global_features=global_features)
                
                # Compute loss
                loss = criterion(out, batch.y)
            
            # Backward pass with mixed precision if enabled
            if self.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Update total loss
            total_loss += loss.item() * batch.num_graphs
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate average loss
        avg_loss = total_loss / len(loader.dataset)
        
        # Log memory usage
        self.log_memory_usage(f"Epoch {epoch+1}")
        
        return avg_loss
    
    def evaluate(self, 
                model: torch.nn.Module, 
                loader: torch.utils.data.DataLoader, 
                criterion: torch.nn.Module) -> float:
        """
        Evaluate the model on a dataset.
        
        Args:
            model: Model to evaluate
            loader: Data loader
            criterion: Loss function
            
        Returns:
            Average loss on the dataset
        """
        model.eval()
        total_loss = 0
        
        # Mixed precision context manager
        amp_context = torch.cuda.amp.autocast() if self.mixed_precision else nullcontext()
        
        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                batch = self.to_device(batch)
                
                # Forward pass with mixed precision if enabled
                with amp_context:
                    # Extract features
                    if hasattr(batch, 'global_features'):
                        global_features = batch.global_features
                    else:
                        global_features = None
                    
                    # Forward pass
                    out = model(batch.x, batch.edge_index, batch.batch, 
                               edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                               global_features=global_features)
                    
                    # Compute loss
                    loss = criterion(out, batch.y)
                
                # Update total loss
                total_loss += loss.item() * batch.num_graphs
        
        # Calculate average loss
        avg_loss = total_loss / len(loader.dataset)
        
        return avg_loss
    
    def predict(self, 
               model: torch.nn.Module, 
               loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Make predictions with the model.
        
        Args:
            model: Model to use for prediction
            loader: Data loader
            
        Returns:
            Tuple of (predictions, true_values, material_ids)
        """
        model.eval()
        predictions = []
        true_values = []
        material_ids = []
        
        # Mixed precision context manager
        amp_context = torch.cuda.amp.autocast() if self.mixed_precision else nullcontext()
        
        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                batch = self.to_device(batch)
                
                # Forward pass with mixed precision if enabled
                with amp_context:
                    # Extract features
                    if hasattr(batch, 'global_features'):
                        global_features = batch.global_features
                    else:
                        global_features = None
                    
                    # Forward pass
                    out = model(batch.x, batch.edge_index, batch.batch, 
                               edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                               global_features=global_features)
                
                # Collect predictions and true values
                predictions.append(out.cpu().numpy())
                true_values.append(batch.y.cpu().numpy())
                
                # Collect material IDs if available
                if hasattr(batch, 'material_id'):
                    material_ids.extend(batch.material_id)
        
        # Concatenate results
        predictions = np.concatenate(predictions, axis=0)
        true_values = np.concatenate(true_values, axis=0)
        
        return predictions, true_values, material_ids if material_ids else None

# Context manager for mixed precision training
class nullcontext:
    """A context manager that does nothing."""
    def __enter__(self):
        return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Function to create an accelerator from configuration
def create_accelerator_from_config(config: Any) -> GPUAccelerator:
    """
    Create a GPU accelerator from configuration.
    
    Args:
        config: Configuration object with GPU settings
        
    Returns:
        GPUAccelerator instance
    """
    use_gpu = config.get("use_gpu", True)
    gpu_device = config.get("gpu_device", 0)
    mixed_precision = config.get("mixed_precision", True)
    memory_fraction = config.get("gpu_memory_fraction", 0.9)
    
    return GPUAccelerator(
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        mixed_precision=mixed_precision,
        memory_fraction=memory_fraction
    )

# Test function
def test_gpu_acceleration():
    """Test GPU acceleration."""
    accelerator = GPUAccelerator()
    
    # Log device information
    logger.info(f"Using device: {accelerator.device}")
    
    # Test tensor creation and movement
    x = torch.randn(1000, 1000)
    x = accelerator.to_device(x)
    
    # Test mixed precision
    if accelerator.mixed_precision:
        with torch.cuda.amp.autocast():
            y = x @ x.t()
        logger.info(f"Mixed precision tensor shape: {y.shape}, dtype: {y.dtype}")
    
    # Log memory usage
    accelerator.log_memory_usage("Test")
    
    return accelerator

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test GPU acceleration
    test_gpu_acceleration()
