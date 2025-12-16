#!/usr/bin/env python3
"""
Large-scale training for 10,000+ materials with memory optimization.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import argparse
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch_geometric.data import DataLoader
import os
import gc

from train_stable_enhanced import StableEnhancedPredictor, EnhancedDataLoader
from scripts.gnn_model import EnhancedCrystalTcGNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_large_scale(
    max_structures=10000,
    num_epochs=50,
    batch_size=32,
    hidden_dim=128,
    learning_rate=0.0005
):
    """Train on large-scale dataset with memory optimization"""
    
    logger.info("="*70)
    logger.info("LARGE-SCALE TRAINING (10K+ MATERIALS)")
    logger.info("="*70)
    
    # Initialize predictor
    predictor = StableEnhancedPredictor()
    
    logger.info(f"\n[CONFIG] Training Configuration:")
    logger.info(f"  Max structures: {max_structures:,}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Device: {predictor.device}")
    
    # Check GPU memory
    if predictor.device.startswith('cuda'):
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"  GPU memory: {total_mem:.1f} GB")
    
    # Process data
    logger.info(f"\n[DATA] Processing structures...")
    dataset = predictor.process_with_enhanced_features(
        csv_file='data/superconductors.csv',
        structures_dir='structures/superconductors',
        max_structures=max_structures
    )
    
    if len(dataset) < 100:
        logger.error("[X] Insufficient data")
        return None
    
    # Check data
    sample = dataset[0]
    num_node_features = sample.x.size(1)
    num_material_features = sample.material_props.size(0)
    
    logger.info(f"\n[DATA] Dataset Ready:")
    logger.info(f"  Total samples: {len(dataset):,}")
    logger.info(f"  Node features: {num_node_features}")
    logger.info(f"  Material features: {num_material_features}")
    logger.info(f"  Enhanced coverage: {predictor.enhanced_count}/{predictor.total_count} "
                f"({predictor.enhanced_count/predictor.total_count*100:.1f}%)")
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_data, temp = torch.utils.data.random_split(dataset, [train_size, val_size + test_size])
    val_data, test_data = torch.utils.data.random_split(temp, [val_size, test_size])
    
    logger.info(f"\n[SPLIT] Data Split:")
    logger.info(f"  Train: {len(train_data):,} ({train_size/len(dataset)*100:.1f}%)")
    logger.info(f"  Val: {len(val_data):,} ({val_size/len(dataset)*100:.1f}%)")
    logger.info(f"  Test: {len(test_data):,} ({test_size/len(dataset)*100:.1f}%)")
    
    # Create loaders with memory optimization
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues on Windows
        pin_memory=False  # Data already on GPU
    )
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Create model
    logger.info(f"\n[MODEL] Creating model...")
    model = EnhancedCrystalTcGNN(
        num_node_features,
        num_material_features,
        hidden_dim=hidden_dim
    ).to(predictor.device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {total_params:,}")
    
    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    criterion = torch.nn.MSELoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )
    
    # Training loop
    logger.info(f"\n[TRAIN] Starting training...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 20
    
    train_history = []
    val_history = []
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in train_loader:
            batch = batch.to(predictor.device)
            optimizer.zero_grad()
            
            output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
            loss = criterion(output.squeeze(), batch.y.squeeze())
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"[!] NaN/Inf loss, skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        if train_batches == 0:
            logger.error("[X] No valid training batches")
            break
        
        avg_train_loss = train_loss / train_batches
        train_history.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(predictor.device)
                output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
                loss = criterion(output.squeeze(), batch.y.squeeze())
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_batches += 1
                    predictions.extend(output.squeeze().cpu().numpy())
                    targets.extend(batch.y.squeeze().cpu().numpy())
        
        if val_batches == 0:
            logger.error("[X] No valid validation batches")
            break
        
        avg_val_loss = val_loss / val_batches
        val_history.append(avg_val_loss)
        
        # Metrics
        r2 = r2_score(targets, predictions) if len(predictions) > 1 else 0.0
        mae = mean_absolute_error(targets, predictions) if len(predictions) > 0 else 0.0
        
        # LR scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/large_scale_model.pt')
            if epoch % 10 == 0:
                logger.info(f"[SAVE] Best model saved")
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"[{epoch:3d}/{num_epochs}] Train: {avg_train_loss:7.2f} | "
                       f"Val: {avg_val_loss:7.2f} | R²: {r2:.3f} | MAE: {mae:.2f}K")
        
        if patience_counter >= patience_limit:
            logger.info(f"[STOP] Early stopping at epoch {epoch}")
            break
        
        # Memory cleanup every 10 epochs
        if epoch % 10 == 0 and predictor.device.startswith('cuda'):
            torch.cuda.empty_cache()
            gc.collect()
    
    # Final evaluation
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION")
    logger.info("="*70)
    
    model.load_state_dict(torch.load('models/large_scale_model.pt'))
    model.eval()
    
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(predictor.device)
            output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
            test_predictions.extend(output.squeeze().cpu().numpy())
            test_targets.extend(batch.y.squeeze().cpu().numpy())
    
    # Metrics
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    test_r2 = r2_score(test_targets, test_predictions)
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    
    logger.info(f"\n[RESULTS] Overall Performance:")
    logger.info(f"  R² Score:  {test_r2:.4f}")
    logger.info(f"  MAE:       {test_mae:.2f} K")
    logger.info(f"  RMSE:      {test_rmse:.2f} K")
    
    # By Tc range
    low_mask = test_targets < 10
    med_mask = (test_targets >= 10) & (test_targets < 50)
    high_mask = test_targets >= 50
    
    logger.info(f"\n[RESULTS] Performance by Tc Range:")
    
    if np.sum(low_mask) > 1:
        low_r2 = r2_score(test_targets[low_mask], test_predictions[low_mask])
        low_mae = mean_absolute_error(test_targets[low_mask], test_predictions[low_mask])
        logger.info(f"  Low (<10K):    R²={low_r2:.4f}, MAE={low_mae:.2f}K ({np.sum(low_mask)} samples)")
    
    if np.sum(med_mask) > 1:
        med_r2 = r2_score(test_targets[med_mask], test_predictions[med_mask])
        med_mae = mean_absolute_error(test_targets[med_mask], test_predictions[med_mask])
        logger.info(f"  Medium (10-50K): R²={med_r2:.4f}, MAE={med_mae:.2f}K ({np.sum(med_mask)} samples)")
    
    if np.sum(high_mask) > 1:
        high_r2 = r2_score(test_targets[high_mask], test_predictions[high_mask])
        high_mae = mean_absolute_error(test_targets[high_mask], test_predictions[high_mask])
        logger.info(f"  High (>50K):   R²={high_r2:.4f}, MAE={high_mae:.2f}K ({np.sum(high_mask)} samples)")
    
    # Save results
    results = {
        'model': 'large_scale',
        'dataset_size': len(dataset),
        'enhanced_coverage': predictor.enhanced_count / predictor.total_count if predictor.total_count > 0 else 0,
        'test_r2': float(test_r2),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'train_history': train_history,
        'val_history': val_history,
        'num_epochs_trained': len(train_history),
        'hidden_dim': hidden_dim,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    
    with open('results/large_scale_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n[SAVE] Results saved")
    logger.info(f"  Model: models/large_scale_model.pt")
    logger.info(f"  Results: results/large_scale_results.json")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nDataset: {len(dataset):,} materials")
    logger.info(f"R² Score: {test_r2:.4f}")
    logger.info(f"MAE: {test_mae:.2f}K")
    logger.info(f"Enhanced Coverage: {results['enhanced_coverage']*100:.1f}%")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-structures', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0005)
    
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    try:
        results = train_large_scale(
            max_structures=args.max_structures,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            learning_rate=args.lr
        )
        
        if results:
            print("\n" + "="*70)
            print("SUCCESS!")
            print(f"Trained on {results['dataset_size']:,} materials")
            print(f"Final R²: {results['test_r2']:.4f}")
            print(f"Final MAE: {results['test_mae']:.2f}K")
            print("="*70)
    
    except Exception as e:
        logger.error(f"[X] Training failed: {e}")
        import traceback
        traceback.print_exc()

