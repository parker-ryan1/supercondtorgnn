#!/usr/bin/env python3
"""
STABLE training pipeline with enhanced Materials Project data.
Fixes NaN issues while integrating all enhanced features.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from pymatgen.core import Structure
import logging
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import os
from pathlib import Path
from scripts.gnn_model import SuperconductorTcPredictor, EnhancedCrystalTcGNN
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDataLoader:
    """Loads and integrates enhanced Materials Project data"""
    
    def __init__(self):
        self.enhanced_df = None
        self._load_enhanced_data()
    
    def _load_enhanced_data(self):
        """Load enhanced data from any available source"""
        # Try multiple sources
        sources = [
            'data/enhanced_superconductors_full.csv',
            'data/enhanced_for_structures.csv',
            'data/enhanced_superconductors.csv'
        ]
        
        for source in sources:
            if Path(source).exists():
                self.enhanced_df = pd.read_csv(source)
                logger.info(f"[OK] Loaded enhanced data from {source}: {len(self.enhanced_df)} materials")
                break
        
        if self.enhanced_df is None:
            logger.warning("[!] No enhanced data found - using baseline features only")
    
    def get_features(self, material_id: str) -> Dict:
        """Get enhanced features for a material"""
        if self.enhanced_df is None:
            return {}
        
        row_data = self.enhanced_df[self.enhanced_df['material_id'] == material_id]
        if row_data.empty:
            return {}
        
        row = row_data.iloc[0]
        
        # Safe float conversion
        def safe_float(val, default=0.0):
            if pd.isna(val) or val is None:
                return default
            try:
                return float(val)
            except:
                return default
        
        # Extract all enhanced features
        features = {
            # Electronic (CRITICAL)
            'fermi_energy': safe_float(row.get('efermi'), 0.0),
            'is_gap_direct': float(row.get('is_gap_direct', False) == True),
            'is_magnetic': float(row.get('is_magnetic', False) == True),
            'total_magnetization': safe_float(row.get('total_magnetization'), 0.0),
            
            # Mechanical
            'bulk_modulus': safe_float(row.get('bulk_modulus_vrh'), 0.0) / 100.0,  # Normalize
            'shear_modulus': safe_float(row.get('shear_modulus_vrh'), 0.0) / 100.0,
            'poisson_ratio': safe_float(row.get('poisson_ratio'), 0.3),
            
            # Structural
            'avg_coordination': safe_float(row.get('avg_coordination_number'), 6.0) / 12.0,  # Normalize
            'coord_variance': (safe_float(row.get('max_coordination_number'), 12.0) - 
                             safe_float(row.get('min_coordination_number'), 4.0)) / 12.0,
            
            # Bond lengths
            'avg_bond_length': safe_float(row.get('avg_bond_length'), 3.0) / 5.0,  # Normalize
            'bond_length_std': safe_float(row.get('bond_length_std'), 0.5) / 2.0,
            
            # Element composition
            'num_transition_metals': safe_float(row.get('num_transition_metals'), 0) / 5.0,
            'num_rare_earths': safe_float(row.get('num_rare_earths'), 0) / 3.0,
            'num_noble_metals': safe_float(row.get('num_noble_metals'), 0) / 3.0,
            
            # Electronegativity
            'avg_electronegativity': safe_float(row.get('avg_electronegativity'), 2.0) / 4.0,
            'electronegativity_variance': safe_float(row.get('electronegativity_variance'), 0.1),
            
            # Thermal (CRITICAL when available)
            'debye_temperature': safe_float(row.get('debye_temperature'), 200.0) / 500.0,  # Normalize
            'has_debye_data': float(pd.notna(row.get('debye_temperature'))),
            
            # Stability
            'energy_above_hull': safe_float(row.get('energy_above_hull'), 0.0),
            'is_stable': float(row.get('is_stable', True) == True),
        }
        
        return features

class StableEnhancedPredictor(SuperconductorTcPredictor):
    """Stable predictor with enhanced features"""
    
    def __init__(self, device: str = None):
        super().__init__(device)
        self.data_loader = EnhancedDataLoader()
        self.enhanced_count = 0
        self.total_count = 0
    
    def process_with_enhanced_features(
        self,
        csv_file: str,
        structures_dir: str,
        max_structures: int = 500
    ) -> List[Data]:
        """Process structures with enhanced features"""
        
        logger.info(f"[>>>] Processing structures with enhanced MP features...")
        
        csv_data = pd.read_csv(csv_file)
        structure_files = list(Path(structures_dir).glob("*.cif"))[:max_structures]
        
        logger.info(f"[>>>] Found {len(structure_files)} structures to process")
        
        dataset = []
        errors = 0
        
        for i, structure_file in enumerate(structure_files):
            try:
                material_id = structure_file.stem
                self.total_count += 1
                
                # Get CSV data
                csv_match = csv_data[csv_data['material_id'] == material_id]
                if not csv_match.empty:
                    csv_row = csv_match.iloc[0]
                    formation_energy = csv_row.get('formation_energy_per_atom', -1.0)
                    band_gap = csv_row.get('band_gap', 0.0)
                    is_metal = csv_row.get('is_metal', True)
                else:
                    formation_energy, band_gap, is_metal = -1.0, 0.0, True
                
                # Load structure
                structure = Structure.from_file(str(structure_file))
                
                # Calculate baseline features
                material_props = self._calculate_advanced_features(structure)
                material_props.update({
                    'formation_energy_per_atom': formation_energy,
                    'band_gap': band_gap,
                    'density': structure.density,
                    'is_metal': is_metal
                })
                
                # Add enhanced features
                enhanced_features = self.data_loader.get_features(material_id)
                if enhanced_features:
                    material_props.update(enhanced_features)
                    self.enhanced_count += 1
                
                # Estimate Tc
                target_tc = self._estimate_tc(structure, material_props)
                
                # Create graph
                graph = self.structure_to_graph(structure, material_props)
                if graph is not None:
                    graph.y = torch.tensor([target_tc], dtype=torch.float32)
                    dataset.append(graph)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"[...] Processed {i+1}/{len(structure_files)} structures")
                
            except Exception as e:
                errors += 1
                if errors < 5:
                    logger.warning(f"[!] Error processing {structure_file.stem}: {e}")
                continue
        
        coverage = (self.enhanced_count / self.total_count * 100) if self.total_count > 0 else 0
        logger.info(f"[OK] Processed {len(dataset)} structures successfully")
        logger.info(f"[OK] Enhanced features: {self.enhanced_count}/{self.total_count} ({coverage:.1f}%)")
        logger.info(f"[!] Errors: {errors}")
        
        # Analyze Tc distribution
        if dataset:
            tc_values = [float(d.y.item()) for d in dataset]
            logger.info(f"\n[STATS] Tc Distribution:")
            logger.info(f"  Mean: {np.mean(tc_values):.2f}K")
            logger.info(f"  Std: {np.std(tc_values):.2f}K")
            logger.info(f"  Range: {np.min(tc_values):.2f}K - {np.max(tc_values):.2f}K")
        
        return dataset

def train_stable_model(max_structures=500, num_epochs=50):
    """Train with stable settings and enhanced features"""
    
    logger.info("="*70)
    logger.info("STABLE ENHANCED TRAINING PIPELINE")
    logger.info("="*70)
    
    # Initialize predictor
    predictor = StableEnhancedPredictor()
    
    # Process data
    dataset = predictor.process_with_enhanced_features(
        csv_file='data/superconductors.csv',
        structures_dir='structures/superconductors',
        max_structures=max_structures
    )
    
    if len(dataset) < 50:
        logger.error("[X] Insufficient data")
        return None
    
    # Check dimensions
    sample = dataset[0]
    num_node_features = sample.x.size(1)
    num_material_features = sample.material_props.size(0)
    
    logger.info(f"\n[CONFIG] Model Configuration:")
    logger.info(f"  Node features: {num_node_features}")
    logger.info(f"  Material features: {num_material_features}")
    logger.info(f"  Dataset size: {len(dataset)}")
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_data, temp = torch.utils.data.random_split(dataset, [train_size, val_size + test_size])
    val_data, test_data = torch.utils.data.random_split(temp, [val_size, test_size])
    
    logger.info(f"\n[SPLIT] Data Split:")
    logger.info(f"  Train: {len(train_data)} ({train_size/len(dataset)*100:.1f}%)")
    logger.info(f"  Val: {len(val_data)} ({val_size/len(dataset)*100:.1f}%)")
    logger.info(f"  Test: {len(test_data)} ({test_size/len(dataset)*100:.1f}%)")
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)
    
    # Create model
    model = EnhancedCrystalTcGNN(
        num_node_features,
        num_material_features,
        hidden_dim=128
    ).to(predictor.device)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training with STABLE settings
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005,  # Lower LR for stability
        weight_decay=1e-4
    )
    
    # Use SIMPLE MSE loss (not physics-aware)
    criterion = torch.nn.MSELoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7
    )
    
    # Training loop
    logger.info(f"\n[TRAIN] Training for {num_epochs} epochs...")
    
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
            
            # NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"[!] NaN/Inf loss at epoch {epoch}, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping
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
            torch.save(model.state_dict(), 'models/stable_enhanced_model.pt')
            if epoch % 5 == 0:
                logger.info(f"[SAVE] Best model saved at epoch {epoch}")
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 5 == 0:
            logger.info(f"[{epoch:3d}/{num_epochs}] Train Loss: {avg_train_loss:.3f} | "
                       f"Val Loss: {avg_val_loss:.3f} | R²: {r2:.3f} | MAE: {mae:.2f}K")
        
        if patience_counter >= patience_limit:
            logger.info(f"[STOP] Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION")
    logger.info("="*70)
    
    model.load_state_dict(torch.load('models/stable_enhanced_model.pt'))
    model.eval()
    
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(predictor.device)
            output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
            test_predictions.extend(output.squeeze().cpu().numpy())
            test_targets.extend(batch.y.squeeze().cpu().numpy())
    
    # Calculate comprehensive metrics
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    test_r2 = r2_score(test_targets, test_predictions)
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    
    # Avoid division by zero in MAPE
    non_zero_mask = test_targets > 0.1
    if np.sum(non_zero_mask) > 0:
        test_mape = mean_absolute_percentage_error(test_targets[non_zero_mask], 
                                                   test_predictions[non_zero_mask])
    else:
        test_mape = 0.0
    
    logger.info(f"\n[RESULTS] Overall Performance:")
    logger.info(f"  R² Score:  {test_r2:.4f}")
    logger.info(f"  MAE:       {test_mae:.2f} K")
    logger.info(f"  RMSE:      {test_rmse:.2f} K")
    logger.info(f"  MAPE:      {test_mape:.2f}%")
    
    # By Tc range
    low_mask = test_targets < 10
    med_mask = (test_targets >= 10) & (test_targets < 50)
    high_mask = test_targets >= 50
    
    logger.info(f"\n[RESULTS] Performance by Tc Range:")
    
    if np.sum(low_mask) > 1:
        low_r2 = r2_score(test_targets[low_mask], test_predictions[low_mask])
        low_mae = mean_absolute_error(test_targets[low_mask], test_predictions[low_mask])
        logger.info(f"  Low Tc (<10K):     R²={low_r2:.4f}, MAE={low_mae:.2f}K ({np.sum(low_mask)} samples)")
    
    if np.sum(med_mask) > 1:
        med_r2 = r2_score(test_targets[med_mask], test_predictions[med_mask])
        med_mae = mean_absolute_error(test_targets[med_mask], test_predictions[med_mask])
        logger.info(f"  Medium Tc (10-50K): R²={med_r2:.4f}, MAE={med_mae:.2f}K ({np.sum(med_mask)} samples)")
    
    if np.sum(high_mask) > 1:
        high_r2 = r2_score(test_targets[high_mask], test_predictions[high_mask])
        high_mae = mean_absolute_error(test_targets[high_mask], test_predictions[high_mask])
        logger.info(f"  High Tc (>50K):     R²={high_r2:.4f}, MAE={high_mae:.2f}K ({np.sum(high_mask)} samples)")
    
    # Sample predictions
    logger.info(f"\n[SAMPLES] Sample Predictions:")
    indices = np.random.choice(len(test_targets), min(10, len(test_targets)), replace=False)
    for idx in indices:
        error_pct = abs(test_predictions[idx] - test_targets[idx]) / test_targets[idx] * 100
        logger.info(f"  Target: {test_targets[idx]:6.2f}K -> Predicted: {test_predictions[idx]:6.2f}K "
                   f"(Error: {error_pct:5.1f}%)")
    
    # Save results
    results = {
        'model': 'stable_enhanced',
        'dataset_size': len(dataset),
        'enhanced_coverage': predictor.enhanced_count / predictor.total_count if predictor.total_count > 0 else 0,
        'test_r2': float(test_r2),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_mape': float(test_mape),
        'train_history': train_history,
        'val_history': val_history,
        'num_epochs_trained': len(train_history)
    }
    
    with open('results/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n[SAVE] Results saved to results/training_results.json")
    logger.info(f"[SAVE] Model saved to models/stable_enhanced_model.pt")
    
    logger.info("\n" + "="*70)
    logger.info("[COMPLETE] Training Complete!")
    logger.info("="*70)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stable training with enhanced MP data')
    parser.add_argument('--max-structures', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    try:
        results = train_stable_model(
            max_structures=args.max_structures,
            num_epochs=args.epochs
        )
        
        if results:
            print("\n" + "="*70)
            print("TRAINING SUCCESSFUL!")
            print(f"Final R² Score: {results['test_r2']:.4f}")
            print(f"Final MAE: {results['test_mae']:.2f}K")
            print(f"Enhanced Feature Coverage: {results['enhanced_coverage']*100:.1f}%")
            print("="*70)
    except Exception as e:
        logger.error(f"[X] Training failed: {e}")
        import traceback
        traceback.print_exc()


