"""
Enhanced Graph Neural Network models for superconductor critical temperature prediction.

This module provides specialized GNN architectures designed specifically for
predicting superconducting transition temperatures (Tc) with high accuracy.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool, 
    global_max_pool, MessagePassing, GraphNorm
)
from torch_geometric.nn.models import MLP
from torch_geometric.data import Data, Batch
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeConv(MessagePassing):
    """
    Edge convolutional layer that considers both node and edge features.
    
    This is an implementation of the edge convolution operation that explicitly
    incorporates edge features into the message passing.
    """
    
    def __init__(self, in_channels, edge_channels, out_channels):
        """
        Initialize the edge convolutional layer.
        
        Args:
            in_channels: Number of input node features
            edge_channels: Number of edge features
            out_channels: Number of output node features
        """
        super().__init__(aggr='add')  # Use "add" aggregation
        
        # MLP for transforming node features
        self.node_mlp = Sequential(
            Linear(in_channels * 2 + edge_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the edge convolutional layer.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_channels]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """
        Construct messages based on source, target, and edge features.
        
        Args:
            x_i: Target node features
            x_j: Source node features
            edge_attr: Edge features
            
        Returns:
            Messages to be aggregated
        """
        # Concatenate source, target, and edge features
        message_inputs = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # Transform with MLP
        return self.node_mlp(message_inputs)

class SuperconductorGNN(torch.nn.Module):
    """
    Enhanced Graph Neural Network for superconductor Tc prediction.
    
    This model incorporates multiple GNN layers, edge features, attention,
    and global features to achieve high accuracy in Tc prediction.
    """
    
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: Optional[int] = None,
                 global_dim: Optional[int] = None,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 num_conv_layers: int = 4,
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True,
                 use_edge_features: bool = True,
                 pool_method: str = "combined"):
        """
        Initialize the superconductor GNN model.
        
        Args:
            node_dim: Number of input node features
            edge_dim: Number of input edge features (None if no edge features)
            global_dim: Number of global features (None if no global features)
            hidden_dim: Hidden dimension size
            output_dim: Output dimension (1 for Tc prediction)
            num_conv_layers: Number of graph convolutional layers
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            use_edge_features: Whether to use edge features
            pool_method: Pooling method ("mean", "max", "add", or "combined")
        """
        super(SuperconductorGNN, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_edge_features = use_edge_features and edge_dim is not None
        self.pool_method = pool_method
        
        # Initial node embedding
        self.node_embedding = Linear(node_dim, hidden_dim)
        
        # Edge embedding if using edge features
        if self.use_edge_features:
            self.edge_embedding = Linear(edge_dim, hidden_dim // 2)
        
        # Graph convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if use_batch_norm else None
        self.graph_norms = torch.nn.ModuleList()
        
        # Create layers based on whether edge features are used
        for i in range(num_conv_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            
            if self.use_edge_features:
                # Use EdgeConv that incorporates edge features
                self.conv_layers.append(
                    EdgeConv(in_dim, hidden_dim // 2, hidden_dim)
                )
            else:
                # Use GAT for attention-based message passing
                self.conv_layers.append(
                    GATConv(in_dim, hidden_dim // 4, heads=4, concat=True)
                )
            
            # Add batch normalization if enabled
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_dim))
            
            # Add graph normalization
            self.graph_norms.append(GraphNorm(hidden_dim))
        
        # Dropout layer
        self.dropout = Dropout(dropout_rate)
        
        # Calculate pooled dimension based on pooling method
        if pool_method == "combined":
            pool_dim = hidden_dim * 3  # mean + max + add
        else:
            pool_dim = hidden_dim
        
        # Calculate final MLP input dimension
        mlp_input_dim = pool_dim
        if global_dim is not None:
            mlp_input_dim += global_dim
        
        # Final prediction MLP
        self.mlp = MLP(
            channel_list=[mlp_input_dim, hidden_dim, hidden_dim // 2, output_dim],
            dropout=dropout_rate,
            norm="batch_norm"
        )
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None,
                global_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            edge_attr: Edge features [num_edges, edge_dim]
            global_features: Global features [batch_size, global_dim]
            
        Returns:
            Predicted Tc values [batch_size, output_dim]
        """
        # Initial node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # Process edge features if used
        if self.use_edge_features and edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)
            edge_attr = F.relu(edge_attr)
        
        # Graph convolutional layers
        for i, conv in enumerate(self.conv_layers):
            # Apply convolution based on whether edge features are used
            if self.use_edge_features and edge_attr is not None:
                x_new = conv(x, edge_index, edge_attr)
            else:
                x_new = conv(x, edge_index)
            
            # Residual connection
            if i > 0:  # Skip first layer for residual
                x = x_new + x  # Residual connection
            else:
                x = x_new
            
            # Apply batch normalization if enabled
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            # Apply graph normalization
            x = self.graph_norms[i](x, batch)
            
            # Apply activation and dropout
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling based on method
        if self.pool_method == "mean":
            x_pool = global_mean_pool(x, batch)
        elif self.pool_method == "max":
            x_pool = global_max_pool(x, batch)
        elif self.pool_method == "add":
            x_pool = global_add_pool(x, batch)
        elif self.pool_method == "combined":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_add = global_add_pool(x, batch)
            x_pool = torch.cat([x_mean, x_max, x_add], dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pool_method}")
        
        # Combine with global features if provided
        if global_features is not None and self.global_dim is not None:
            x_pool = torch.cat([x_pool, global_features], dim=1)
        
        # Final prediction
        out = self.mlp(x_pool)
        
        return out

class ResidualGNN(torch.nn.Module):
    """
    Residual Graph Neural Network for superconductor Tc prediction.
    
    This model uses residual connections between GNN layers to enable
    training of deeper networks for better performance.
    """
    
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: Optional[int] = None,
                 global_dim: Optional[int] = None,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 num_blocks: int = 3,
                 layers_per_block: int = 2,
                 dropout_rate: float = 0.2):
        """
        Initialize the residual GNN model.
        
        Args:
            node_dim: Number of input node features
            edge_dim: Number of input edge features (None if no edge features)
            global_dim: Number of global features (None if no global features)
            hidden_dim: Hidden dimension size
            output_dim: Output dimension (1 for Tc prediction)
            num_blocks: Number of residual blocks
            layers_per_block: Number of layers per residual block
            dropout_rate: Dropout rate for regularization
        """
        super(ResidualGNN, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.layers_per_block = layers_per_block
        
        # Initial node embedding
        self.node_embedding = Linear(node_dim, hidden_dim)
        
        # Edge embedding if using edge features
        if edge_dim is not None:
            self.edge_embedding = Linear(edge_dim, hidden_dim // 2)
        
        # Residual blocks
        self.blocks = torch.nn.ModuleList()
        
        for _ in range(num_blocks):
            block = ResidualBlock(
                hidden_dim=hidden_dim,
                layers_per_block=layers_per_block,
                dropout_rate=dropout_rate,
                use_edge_features=edge_dim is not None
            )
            self.blocks.append(block)
        
        # Batch normalization
        self.batch_norm = BatchNorm1d(hidden_dim)
        
        # Calculate final MLP input dimension
        mlp_input_dim = hidden_dim * 3  # mean + max + add pooling
        if global_dim is not None:
            mlp_input_dim += global_dim
        
        # Final prediction MLP
        self.mlp = MLP(
            channel_list=[mlp_input_dim, hidden_dim, hidden_dim // 2, output_dim],
            dropout=dropout_rate,
            norm="batch_norm"
        )
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None,
                global_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            edge_attr: Edge features [num_edges, edge_dim]
            global_features: Global features [batch_size, global_dim]
            
        Returns:
            Predicted Tc values [batch_size, output_dim]
        """
        # Initial node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # Process edge features if available
        if edge_attr is not None and self.edge_dim is not None:
            edge_attr = self.edge_embedding(edge_attr)
            edge_attr = F.relu(edge_attr)
        
        # Process through residual blocks
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        
        # Final batch normalization
        x = self.batch_norm(x)
        x = F.relu(x)
        
        # Global pooling (combine mean, max, and add)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x_pool = torch.cat([x_mean, x_max, x_add], dim=1)
        
        # Combine with global features if provided
        if global_features is not None and self.global_dim is not None:
            x_pool = torch.cat([x_pool, global_features], dim=1)
        
        # Final prediction
        out = self.mlp(x_pool)
        
        return out

class ResidualBlock(torch.nn.Module):
    """
    Residual block for the ResidualGNN model.
    
    This block contains multiple GNN layers with a residual connection.
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 layers_per_block: int = 2,
                 dropout_rate: float = 0.2,
                 use_edge_features: bool = False):
        """
        Initialize the residual block.
        
        Args:
            hidden_dim: Hidden dimension size
            layers_per_block: Number of layers in the block
            dropout_rate: Dropout rate for regularization
            use_edge_features: Whether to use edge features
        """
        super(ResidualBlock, self).__init__()
        
        self.use_edge_features = use_edge_features
        
        # Create layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for _ in range(layers_per_block):
            if use_edge_features:
                # Use EdgeConv that incorporates edge features
                self.layers.append(
                    EdgeConv(hidden_dim, hidden_dim // 2, hidden_dim)
                )
            else:
                # Use GAT for attention-based message passing
                self.layers.append(
                    GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
                )
            
            # Add batch normalization
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Dropout layer
        self.dropout = Dropout(dropout_rate)
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Store original input for residual connection
        identity = x
        
        # Process through layers
        for i, (layer, batch_norm) in enumerate(zip(self.layers, self.batch_norms)):
            # Apply convolution based on whether edge features are used
            if self.use_edge_features and edge_attr is not None:
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)
            
            # Apply batch normalization
            x = batch_norm(x)
            
            # Apply activation and dropout (except last layer)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Add residual connection
        x = x + identity
        
        # Final activation
        x = F.relu(x)
        
        return x

def test_model():
    """Test the model with random data."""
    # Create random data
    num_nodes = 20
    batch_size = 2
    node_dim = 12
    edge_dim = 4
    global_dim = 15
    
    # Create node features
    x = torch.randn(num_nodes, node_dim)
    
    # Create edge index (random graph)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    
    # Create edge features
    edge_attr = torch.randn(edge_index.size(1), edge_dim)
    
    # Create batch assignment
    batch = torch.zeros(num_nodes, dtype=torch.long)
    batch[num_nodes // 2:] = 1  # Second half belongs to second graph
    
    # Create global features
    global_features = torch.randn(batch_size, global_dim)
    
    # Create model
    model = SuperconductorGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=64,
        num_conv_layers=3
    )
    
    # Forward pass
    out = model(x, edge_index, batch, edge_attr, global_features)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shapes: x={x.shape}, edge_index={edge_index.shape}, "
          f"edge_attr={edge_attr.shape}, global_features={global_features.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test residual model
    res_model = ResidualGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=64,
        num_blocks=2,
        layers_per_block=2
    )
    
    # Forward pass
    res_out = res_model(x, edge_index, batch, edge_attr, global_features)
    
    print(f"\nModel: {res_model.__class__.__name__}")
    print(f"Output shape: {res_out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in res_model.parameters())}")
    
    return model, res_model

if __name__ == "__main__":
    test_model()
