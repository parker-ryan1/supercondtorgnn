#!/usr/bin/env python3
"""
Simplified demo training script that completes quickly.
Demonstrates all components working together.
"""

import torch
import sys
from pathlib import Path
import logging
import numpy as np

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from gnn_model import SuperconductorTcPredictor, EnhancedCrystalTcGNN
from torch_geometric.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("SUPERCONDUCTOR GNN - DEMO TRAINING")
    logger.info("="*70)
    
    # Initialize predictor
    logger.info("\n1. Initializing predictor...")
    predictor = SuperconductorTcPredictor()
    
    # Process a small subset of data
    logger.info("\n2. Processing structures...")
    csv_file = 'data/superconductors.csv'
    structures_dir = 'structures/superconductors'
    
    dataset = predictor.process_structures_for_tc(
        csv_file=csv_file,
        structures_dir=structures_dir,
        max_structures=50  # Small subset for demo
    )
    
    logger.info(f"   Loaded {len(dataset)} graph samples")
    
    if len(dataset) < 10:
        logger.error("Not enough data loaded!")
        return False
    
    # Split dataset
    logger.info("\n3. Splitting dataset...")
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, temp_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size + test_size]
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size]
    )
    
    logger.info(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Initialize model
    logger.info("\n4. Creating model...")
    num_node_features = dataset[0].x.size(1)
    num_material_features = dataset[0].material_props.size(0)
    
    logger.info(f"   Node features: {num_node_features}")
    logger.info(f"   Material features: {num_material_features}")
    
    model = EnhancedCrystalTcGNN(
        num_node_features=num_node_features,
        num_material_features=num_material_features,
        hidden_dim=64  # Smaller for faster training
    ).to(predictor.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Model parameters: {num_params:,}")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    logger.info("\n5. Training model (10 epochs)...")
    best_val_loss = float('inf')
    
    for epoch in range(10):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            try:
                batch = batch.to(predictor.device)
                optimizer.zero_grad()
                
                output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
                loss = criterion(output.squeeze(), batch.y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                logger.warning(f"Training batch error: {e}")
                continue
        
        if num_batches == 0:
            logger.warning("No successful training batches!")
            continue
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = batch.to(predictor.device)
                    output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
                    loss = criterion(output.squeeze(), batch.y)
                    val_loss += loss.item()
                    val_batches += 1
                    
                    predictions.extend(output.cpu().numpy().flatten())
                    targets.extend(batch.y.cpu().numpy())
                except Exception as e:
                    continue
        
        if val_batches == 0:
            logger.warning("No successful validation batches!")
            continue
        
        avg_val_loss = val_loss / val_batches
        scheduler.step(avg_val_loss)
        
        # Calculate metrics
        if len(predictions) > 1:
            from sklearn.metrics import r2_score, mean_absolute_error
            r2 = r2_score(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
        else:
            r2 = 0.0
            mae = 0.0
        
        logger.info(f"   Epoch {epoch+1:2d}/10: Train Loss={avg_train_loss:7.2f}, "
                   f"Val Loss={avg_val_loss:7.2f}, R²={r2:6.3f}, MAE={mae:6.2f}K")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/demo_best_model.pt')
    
    # Final evaluation
    logger.info("\n6. Final evaluation on test set...")
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                batch = batch.to(predictor.device)
                output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
                test_predictions.extend(output.cpu().numpy().flatten())
                test_targets.extend(batch.y.cpu().numpy())
            except:
                continue
    
    if len(test_predictions) > 1:
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        test_r2 = r2_score(test_targets, test_predictions)
        test_mae = mean_absolute_error(test_targets, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
        
        logger.info("="*70)
        logger.info("FINAL RESULTS")
        logger.info("="*70)
        logger.info(f"   Test R² Score:  {test_r2:.4f}")
        logger.info(f"   Test MAE:       {test_mae:.2f} K")
        logger.info(f"   Test RMSE:      {test_rmse:.2f} K")
        logger.info(f"   Sample size:    {len(test_predictions)} materials")
        logger.info("="*70)
        
        # Show sample predictions
        logger.info("\nSample Predictions:")
        for i in range(min(5, len(test_predictions))):
            error = abs(test_predictions[i] - test_targets[i])
            error_pct = (error / test_targets[i]) * 100
            logger.info(f"   {i+1}. Target: {test_targets[i]:6.2f}K  →  "
                       f"Predicted: {test_predictions[i]:6.2f}K  "
                       f"(Error: {error:5.2f}K, {error_pct:5.1f}%)")
        
        logger.info("\n✅ Demo training completed successfully!")
        logger.info(f"   Model saved to: models/demo_best_model.pt")
        return True
    else:
        logger.error("Not enough test predictions!")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


