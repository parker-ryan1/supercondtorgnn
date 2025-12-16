#!/usr/bin/env python3
"""
Quick test script to verify the GNN model works correctly.
Tests model creation, forward pass, and basic training.
"""

import torch
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from gnn_model import SuperconductorTcPredictor, EnhancedCrystalTcGNN, CrystalTcGNN
from torch_geometric.data import Data, DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_creation():
    """Test that models can be created successfully"""
    logger.info("Testing model creation...")
    
    try:
        predictor = SuperconductorTcPredictor()
        logger.info(f"✓ Predictor created on device: {predictor.device}")
        
        # Test basic model
        model_basic = CrystalTcGNN(num_node_features=20, num_material_features=24, hidden_dim=64)
        logger.info(f"✓ Basic CrystalTcGNN created with {sum(p.numel() for p in model_basic.parameters())} parameters")
        
        # Test enhanced model
        model_enhanced = EnhancedCrystalTcGNN(num_node_features=20, num_material_features=24, hidden_dim=128)
        logger.info(f"✓ Enhanced CrystalTcGNN created with {sum(p.numel() for p in model_enhanced.parameters())} parameters")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test that forward pass works with dummy data"""
    logger.info("\nTesting forward pass...")
    
    try:
        predictor = SuperconductorTcPredictor()
        device = predictor.device
        
        # Create dummy graph data
        num_nodes = 10
        num_edges = 20
        
        x = torch.randn(num_nodes, 20, device=device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        material_props = torch.randn(24, device=device)
        
        # Test basic model
        model_basic = CrystalTcGNN(num_node_features=20, num_material_features=24, hidden_dim=64).to(device)
        output_basic = model_basic(x, edge_index, batch, material_props)
        logger.info(f"✓ Basic model forward pass successful, output shape: {output_basic.shape}")
        
        # Test enhanced model
        model_enhanced = EnhancedCrystalTcGNN(num_node_features=20, num_material_features=24, hidden_dim=128).to(device)
        output_enhanced = model_enhanced(x, edge_index, batch, material_props)
        logger.info(f"✓ Enhanced model forward pass successful, output shape: {output_enhanced.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test that training step works"""
    logger.info("\nTesting training step...")
    
    try:
        predictor = SuperconductorTcPredictor()
        device = predictor.device
        
        # Create dummy dataset
        dataset = []
        for _ in range(10):
            num_nodes = torch.randint(5, 15, (1,)).item()
            num_edges = torch.randint(10, 30, (1,)).item()
            
            x = torch.randn(num_nodes, 20, device=device)
            edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
            material_props = torch.randn(24, device=device)
            y = torch.tensor([torch.rand(1).item() * 100], device=device)
            
            data = Data(x=x, edge_index=edge_index, material_props=material_props, y=y)
            dataset.append(data)
        
        # Create data loader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Create model and optimizer
        model = EnhancedCrystalTcGNN(num_node_features=20, num_material_features=24, hidden_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Training step
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
            loss = criterion(output.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            
            logger.info(f"✓ Training step successful, loss: {loss.item():.4f}")
            break  # Just test one batch
        
        return True
    except Exception as e:
        logger.error(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("SUPERCONDUCTOR GNN MODEL QUICK TEST")
    logger.info("="*60)
    
    results = []
    
    results.append(("Model Creation", test_model_creation()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Training Step", test_training_step()))
    
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! Model is ready for training.")
    else:
        logger.info("\n❌ SOME TESTS FAILED! Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


