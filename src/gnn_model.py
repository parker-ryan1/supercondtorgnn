"""
Graph Neural Network models for superconductor analysis and prediction.

This module provides GNN models for predicting superconducting properties
of materials based on their crystal structure and composition.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GATConv, global_add_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import os
import logging
from typing import List, Tuple, Dict, Optional, Union, Any
import time
from pathlib import Path
from tqdm import tqdm

# Try to import optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Import configuration
try:
    from config import config
except ImportError:
    # Fallback if config is not available
    config = {
        "models_dir": "models",
        "hidden_dim": 64,
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 100,
        "use_gpu": True
    }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedCrystalGNN(torch.nn.Module):
    """
    Improved Graph Neural Network for crystal property prediction with better architecture.
    """
    
    def __init__(self, 
                 num_node_features: int, 
                 num_global_features: int = 0,
                 hidden_dim: int = 128,
                 num_conv_layers: int = 4,
                 dropout_rate: float = 0.3,
                 use_attention: bool = True,
                 output_dim: int = 1):
        """
        Initialize the improved GNN model.
        """
        super(ImprovedCrystalGNN, self).__init__()
        
        self.num_node_features = num_node_features
        self.num_global_features = num_global_features
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # Initialize graph convolution layers with residual connections
        self.conv_layers = torch.nn.ModuleList()
        
        # First layer (input to hidden)
        if use_attention:
            self.conv_layers.append(GATConv(num_node_features, hidden_dim, heads=4, concat=False))
        else:
            self.conv_layers.append(GCNConv(num_node_features, hidden_dim))
        
        # Additional layers (hidden to hidden)
        for _ in range(num_conv_layers - 1):
            if use_attention:
                self.conv_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # Batch normalization
        self.batch_norm = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_conv_layers)
        ])
        
        # Layer normalization for better training stability
        self.layer_norm = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_dim) for _ in range(num_conv_layers)
        ])
        
        # Multi-head pooling (mean, max, sum)
        self.pool_weight = torch.nn.Parameter(torch.ones(3) / 3)
        
        # Final prediction layers with residual connections
        combined_dim = hidden_dim * 3  # 3x because of multi-head pooling
        self.fc1 = torch.nn.Linear(combined_dim, hidden_dim * 2)
        self.fc2 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = torch.nn.Linear(hidden_dim // 2, output_dim)
        
        # Additional batch norms for FC layers
        self.fc_bn1 = torch.nn.BatchNorm1d(hidden_dim * 2)
        self.fc_bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc_bn3 = torch.nn.BatchNorm1d(hidden_dim // 2)
    
    def forward(self, x, edge_index, batch, global_features=None):
        """
        Forward pass through the network.
        """
        # Store initial features for potential residual connections
        h_list = []
        
        # Graph convolution layers with residual connections
        h = x
        for i, conv in enumerate(self.conv_layers):
            h_new = conv(h, edge_index)
            
            # Apply batch normalization
            if h_new.size(0) > 1:  # Only apply if batch size > 1
                h_new = self.batch_norm[i](h_new)
            
            # Apply activation
            h_new = F.relu(h_new)
            
            # Apply dropout
            h_new = self.dropout(h_new)
            
            # Residual connection (if dimensions match)
            if i > 0 and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
            
            h_list.append(h)
        
        # Multi-head global pooling (mean, max, sum)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        
        # Weighted combination of pooling methods
        w = F.softmax(self.pool_weight, dim=0)
        h_pooled = torch.cat([
            w[0] * h_mean,
            w[1] * h_max,
            w[2] * h_sum
        ], dim=1)
        
        # Final prediction layers with residual connections
        h = self.fc1(h_pooled)
        if h.size(0) > 1:
            h = self.fc_bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.fc2(h)
        if h.size(0) > 1:
            h = self.fc_bn2(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.fc3(h)
        if h.size(0) > 1:
            h = self.fc_bn3(h)
        h = F.relu(h)
        
        # Final output (no activation for regression)
        out = self.fc4(h)
        
        return out

class SuperconductorTcPredictor:
    """
    Predictor for superconductor critical temperature (Tc).
    
    This class handles training, evaluation, and prediction using the GNN model.
    """
    
    def __init__(self, 
                 model_dir: Optional[str] = None,
                 hidden_dim: Optional[int] = None,
                 learning_rate: Optional[float] = None,
                 device: Optional[str] = None):
        """
        Initialize the Tc predictor.
        """
        # Set parameters from arguments or config
        self.model_dir = model_dir or config.get("models_dir", "models")
        self.hidden_dim = hidden_dim or config.get("hidden_dim", 128)  # Increased from 64
        self.learning_rate = learning_rate or config.get("learning_rate", 0.0005)  # Reduced for stability
        
        # Set device
        self.device = self._setup_device(device)
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model to None (will be created/loaded later)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        logger.info(f"SuperconductorTcPredictor initialized on {self.device}")
    
    def _setup_device(self, device: Optional[str] = None) -> str:
        """
        Set up the device for training and inference.
        """
        if device is not None:
            return device
        
        use_gpu = config.get("use_gpu", True)
        
        if not use_gpu or not torch.cuda.is_available():
            logger.info("Using CPU for computation")
            return 'cpu'
        
        # Get GPU information
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s)")
        
        if gpu_count == 0:
            logger.info("No GPUs available, using CPU")
            return 'cpu'
        
        # Use the first GPU
        device = f'cuda:0'
        torch.cuda.set_device(0)
        
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
        
        return device
    
    def create_model(self, 
                    num_node_features: int, 
                    num_global_features: int = 0,
                    use_attention: bool = True) -> ImprovedCrystalGNN:
        """
        Create a new improved GNN model.
        """
        model = ImprovedCrystalGNN(
            num_node_features=num_node_features,
            num_global_features=num_global_features,
            hidden_dim=self.hidden_dim,
            num_conv_layers=4,  # Increased depth
            dropout_rate=0.3,
            use_attention=use_attention
        )
        
        model = model.to(self.device)
        
        # Set up optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Set up scheduler with cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.model = model
        logger.info(f"Created improved model with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def load_model(self, model_path: Optional[str] = None) -> ImprovedCrystalGNN:
        """
        Load a saved model.
        """
        if model_path is None:
            model_path = os.path.join(self.model_dir, "superconductor_gnn.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model state
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Create model with the same architecture
        model = ImprovedCrystalGNN(
            num_node_features=state_dict["num_node_features"],
            num_global_features=state_dict["num_global_features"],
            hidden_dim=state_dict["hidden_dim"],
            num_conv_layers=state_dict["num_conv_layers"],
            dropout_rate=state_dict["dropout_rate"],
            use_attention=state_dict["use_attention"]
        )
        
        # Load weights
        model.load_state_dict(state_dict["model_state_dict"])
        model = model.to(self.device)
        
        # Set up optimizer and scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        if "optimizer_state_dict" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        self.model = model
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save the current model.
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if model_path is None:
            model_path = os.path.join(self.model_dir, "superconductor_gnn.pt")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        
        # Save model state with architecture parameters
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "num_node_features": self.model.num_node_features,
            "num_global_features": self.model.num_global_features,
            "hidden_dim": self.model.hidden_dim,
            "num_conv_layers": self.model.num_conv_layers,
            "dropout_rate": self.model.dropout_rate,
            "use_attention": self.model.use_attention
        }
        
        torch.save(state_dict, model_path)
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def train(self, 
             train_loader: DataLoader, 
             val_loader: Optional[DataLoader] = None,
             num_epochs: int = 300,
             early_stopping_patience: int = 30,
             log_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model with improved training procedure.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call create_model() first.")
        
        # Set up TensorBoard if available
        writer = None
        if TENSORBOARD_AVAILABLE and log_dir:
            writer = SummaryWriter(log_dir)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        # Check if mixed precision is available and enabled
        use_amp = False
        device_type = 'cuda' if self.device.startswith('cuda') else 'cpu'
        if device_type == 'cuda' and hasattr(torch, 'amp') and config.get("mixed_precision", True):
            use_amp = True
            scaler = torch.amp.GradScaler()
            logger.info("Using mixed precision training")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False):
                # Move batch to device
                batch = batch.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if use_amp:
                    with torch.amp.autocast(device_type=device_type):
                        # Forward pass
                        out = self.model(batch.x, batch.edge_index, batch.batch)
                        
                        # Reshape targets to match output
                        targets = batch.y.view(-1, 1)
                        
                        # Compute loss with Huber loss for robustness
                        loss = F.smooth_l1_loss(out, targets)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping for stability
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Forward pass
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    
                    # Reshape targets to match output
                    targets = batch.y.view(-1, 1)
                    
                    # Compute loss
                    loss = F.smooth_l1_loss(out, targets)
                    
                    # Backward pass and optimization
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                
                total_loss += loss.item() * batch.num_graphs
            
            # Calculate average loss
            avg_train_loss = total_loss / len(train_loader.dataset)
            history["train_loss"].append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                val_loss = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                
                # Update learning rate scheduler
                self.scheduler.step()
                
                # Log current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                history["learning_rate"].append(current_lr)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    self.save_model(os.path.join(self.model_dir, "best_model.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                               f"Train Loss = {avg_train_loss:.4f}, "
                               f"Val Loss = {val_loss:.4f}, "
                               f"LR = {current_lr:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                               f"Train Loss = {avg_train_loss:.4f}")
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Loss/train', avg_train_loss, epoch)
                if val_loader:
                    writer.add_scalar('Loss/val', val_loss, epoch)
                    writer.add_scalar('Learning Rate', current_lr, epoch)
        
        # Training complete
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Close TensorBoard writer
        if writer:
            writer.close()
        
        return history
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate the model on a dataset.
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        total_loss = 0
        
        # Check if mixed precision is available and enabled
        use_amp = False
        device_type = 'cuda' if self.device.startswith('cuda') else 'cpu'
        if device_type == 'cuda' and hasattr(torch, 'amp') and config.get("mixed_precision", True):
            use_amp = True
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass with mixed precision if enabled
                if use_amp:
                    with torch.amp.autocast(device_type=device_type):
                        # Forward pass
                        out = self.model(batch.x, batch.edge_index, batch.batch)
                        
                        # Reshape targets
                        targets = batch.y.view(-1, 1)
                        
                        # Compute loss
                        loss = F.smooth_l1_loss(out, targets)
                else:
                    # Forward pass
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    
                    # Reshape targets
                    targets = batch.y.view(-1, 1)
                    
                    # Compute loss
                    loss = F.smooth_l1_loss(out, targets)
                
                total_loss += loss.item() * batch.num_graphs
        
        # Calculate average loss
        avg_loss = total_loss / len(data_loader.dataset)
        
        return avg_loss
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Make predictions on a dataset.
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        predictions = []
        true_values = []
        material_ids = []
        
        # Check if mixed precision is available and enabled
        use_amp = False
        device_type = 'cuda' if self.device.startswith('cuda') else 'cpu'
        if device_type == 'cuda' and hasattr(torch, 'amp') and config.get("mixed_precision", True):
            use_amp = True
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass with mixed precision if enabled
                if use_amp:
                    with torch.amp.autocast(device_type=device_type):
                        # Forward pass
                        out = self.model(batch.x, batch.edge_index, batch.batch)
                else:
                    # Forward pass
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                
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
    
    def predict_single(self, data: Data) -> float:
        """
        Make a prediction on a single structure.
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            data = data.to(self.device)
            
            # Add batch dimension
            if not hasattr(data, 'batch'):
                data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
            
            # Forward pass
            out = self.model(data.x, data.edge_index, data.batch)
            
            # Get prediction
            prediction = out.item()
        
        return prediction

def main():
    """Main function for testing the module."""
    logger.info("Testing improved GNN model module")
    
    # Create a simple predictor
    predictor = SuperconductorTcPredictor()
    
    # Create a simple model
    model = predictor.create_model(num_node_features=4, num_global_features=6)
    logger.info(f"Created improved model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Save the model
    model_path = predictor.save_model()
    logger.info(f"Saved model to {model_path}")
    
    # Load the model
    loaded_model = predictor.load_model(model_path)
    logger.info("Successfully loaded the model")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
