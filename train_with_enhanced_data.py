#!/usr/bin/env python3
"""
Enhanced training pipeline using detailed Materials Project data.
Integrates Fermi energy, coordination numbers, elastic properties, and more.
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
from pathlib import Path
from scripts.gnn_model import (
    SuperconductorTcPredictor, 
    EnhancedCrystalTcGNN,
    PhysicsAwareTcLoss
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataIntegrator:
    """
    Integrates enhanced Materials Project data into the training pipeline
    """
    
    def __init__(self, enhanced_csv_path='data/enhanced_superconductors.csv'):
        """Load enhanced data from Materials Project"""
        self.enhanced_df = None
        
        if Path(enhanced_csv_path).exists():
            self.enhanced_df = pd.read_csv(enhanced_csv_path)
            logger.info(f"✅ Loaded enhanced data for {len(self.enhanced_df)} materials")
            
            # Check for full dataset
            full_path = 'data/enhanced_superconductors_full.csv'
            if Path(full_path).exists():
                full_df = pd.read_csv(full_path)
                logger.info(f"✅ Found full enhanced dataset with {len(full_df)} materials")
                self.enhanced_df = full_df
        else:
            logger.warning(f"⚠️ Enhanced data not found at {enhanced_csv_path}")
            logger.info("   Using baseline features only")
    
    def get_enhanced_features(self, material_id: str) -> Dict:
        """
        Get enhanced features for a material from Materials Project data
        """
        if self.enhanced_df is None:
            return {}
        
        # Find material in enhanced dataset
        material_row = self.enhanced_df[self.enhanced_df['material_id'] == material_id]
        
        if material_row.empty:
            return {}
        
        row = material_row.iloc[0]
        
        # Extract enhanced features with safe defaults
        enhanced_features = {
            # Electronic structure (CRITICAL for Tc!)
            'fermi_energy': self._safe_float(row.get('efermi'), 0.0),
            'is_gap_direct': float(row.get('is_gap_direct', False) == True),
            'is_magnetic': float(row.get('is_magnetic', False) == True),
            'total_magnetization': self._safe_float(row.get('total_magnetization'), 0.0),
            
            # Mechanical properties (related to phonons)
            'bulk_modulus': self._safe_float(row.get('bulk_modulus_vrh'), 0.0),
            'shear_modulus': self._safe_float(row.get('shear_modulus_vrh'), 0.0),
            'poisson_ratio': self._safe_float(row.get('poisson_ratio'), 0.3),
            'elastic_anisotropy': self._safe_float(row.get('elastic_anisotropy'), 1.0),
            
            # Structural properties (affects electron-phonon coupling)
            'avg_coordination': self._safe_float(row.get('avg_coordination_number'), 6.0),
            'min_coordination': self._safe_float(row.get('min_coordination_number'), 4.0),
            'max_coordination': self._safe_float(row.get('max_coordination_number'), 12.0),
            
            # Bond lengths (affects phonon modes)
            'avg_bond_length': self._safe_float(row.get('avg_bond_length'), 3.0),
            'min_bond_length': self._safe_float(row.get('min_bond_length'), 2.0),
            'max_bond_length': self._safe_float(row.get('max_bond_length'), 4.0),
            'bond_length_std': self._safe_float(row.get('bond_length_std'), 0.5),
            
            # Element composition (superconductivity indicators)
            'num_transition_metals': float(row.get('num_transition_metals', 0)),
            'num_rare_earths': float(row.get('num_rare_earths', 0)),
            'num_alkali': float(row.get('num_alkali', 0)),
            'num_alkaline_earth': float(row.get('num_alkaline_earth', 0)),
            'num_noble_metals': float(row.get('num_noble_metals', 0)),
            
            # Electronegativity (bonding character)
            'avg_electronegativity': self._safe_float(row.get('avg_electronegativity'), 2.0),
            'electronegativity_variance': self._safe_float(row.get('electronegativity_variance'), 0.1),
            'electronegativity_range': self._safe_float(row.get('electronegativity_range'), 0.0),
            
            # Atomic mass (phonon frequencies)
            'avg_atomic_mass': self._safe_float(row.get('avg_atomic_mass'), 100.0),
            'atomic_mass_variance': self._safe_float(row.get('atomic_mass_variance'), 100.0),
            
            # Thermal properties (CRITICAL when available!)
            'debye_temperature': self._safe_float(row.get('debye_temperature'), 200.0),
            'has_debye_data': float(pd.notna(row.get('debye_temperature'))),
            
            # Stability
            'energy_above_hull': self._safe_float(row.get('energy_above_hull'), 0.0),
            'is_stable': float(row.get('is_stable', True) == True),
        }
        
        return enhanced_features
    
    def _safe_float(self, value, default=0.0):
        """Safely convert to float with default"""
        if pd.isna(value) or value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

class ImprovedTcPredictor(SuperconductorTcPredictor):
    """
    Enhanced Tc predictor that integrates Materials Project data
    """
    
    def __init__(self, device: str = None):
        super().__init__(device)
        self.data_integrator = EnhancedDataIntegrator()
        logger.info("🚀 Initialized ImprovedTcPredictor with enhanced features")
    
    def process_structures_enhanced(
        self, 
        csv_file: str, 
        structures_dir: str, 
        max_structures: int = 1000
    ) -> List[Data]:
        """
        Process structures with enhanced Materials Project features
        """
        logger.info(f"🔧 Processing structures with ENHANCED features...")
        
        try:
            # Load CSV data
            csv_data = pd.read_csv(csv_file)
            logger.info(f"Loaded CSV with {len(csv_data)} entries")
            
            # Get available structure files
            structure_files = list(Path(structures_dir).glob("*.cif"))
            logger.info(f"Found {len(structure_files)} structure files")
            
            if max_structures:
                structure_files = structure_files[:max_structures]
                logger.info(f"Limited to {max_structures} structures")
            
            dataset = []
            processed_count = 0
            error_count = 0
            enhanced_count = 0
            
            for structure_file in structure_files:
                if processed_count >= max_structures:
                    break
                
                try:
                    material_id = structure_file.stem
                    
                    # Find CSV entry
                    csv_match = csv_data[csv_data['material_id'] == material_id]
                    if csv_match.empty:
                        is_metal = True
                        formation_energy = -1.0
                        band_gap = 0.0
                    else:
                        csv_row = csv_match.iloc[0]
                        is_metal = csv_row.get('is_metal', True)
                        formation_energy = csv_row.get('formation_energy_per_atom', -1.0)
                        band_gap = csv_row.get('band_gap', 0.0)
                    
                    # Load structure
                    structure = Structure.from_file(str(structure_file))
                    
                    # Basic material properties
                    material_props = self._calculate_advanced_features(structure)
                    material_props['formation_energy_per_atom'] = formation_energy
                    material_props['band_gap'] = band_gap
                    material_props['density'] = structure.density
                    material_props['is_metal'] = is_metal
                    
                    # ⭐ GET ENHANCED FEATURES FROM MATERIALS PROJECT ⭐
                    enhanced_features = self.data_integrator.get_enhanced_features(material_id)
                    
                    if enhanced_features:
                        material_props.update(enhanced_features)
                        enhanced_count += 1
                    
                    # Estimate Tc
                    target_tc = self._estimate_tc(structure, material_props)
                    
                    # Create graph with enhanced features
                    graph_data = self.structure_to_graph(structure, material_props)
                    graph_data.y = torch.tensor([target_tc], dtype=torch.float32)
                    dataset.append(graph_data)
                    
                    processed_count += 1
                    
                    if processed_count % 200 == 0:
                        logger.info(f"Processed {processed_count} structures ({enhanced_count} with enhanced features)...")
                
                except Exception as e:
                    error_count += 1
                    if error_count < 5:
                        logger.warning(f"Error processing {structure_file}: {e}")
                    continue
            
            logger.info(f"✅ Successfully processed {processed_count} structures")
            logger.info(f"   Enhanced features: {enhanced_count}/{processed_count} ({enhanced_count/processed_count*100:.1f}%)")
            logger.info(f"   Errors: {error_count}")
            
            # Analyze Tc distribution
            if len(dataset) > 0:
                tc_values = [float(data.y.item()) for data in dataset]
                logger.info("\n📊 Tc Distribution:")
                logger.info(f"   Mean: {np.mean(tc_values):.2f}K")
                logger.info(f"   Std: {np.std(tc_values):.2f}K")
                logger.info(f"   Range: {np.min(tc_values):.2f}K - {np.max(tc_values):.2f}K")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error in process_structures_enhanced: {e}")
            return []

def train_improved_model(max_structures=500, num_epochs=50):
    """
    Train model with enhanced features
    """
    logger.info("="*70)
    logger.info("IMPROVED TRAINING WITH ENHANCED MATERIALS PROJECT DATA")
    logger.info("="*70)
    
    # Initialize improved predictor
    predictor = ImprovedTcPredictor()
    
    # Process structures with enhanced features
    logger.info("\n📊 Loading dataset with enhanced features...")
    dataset = predictor.process_structures_enhanced(
        csv_file='data/superconductors.csv',
        structures_dir='structures/superconductors',
        max_structures=max_structures
    )
    
    if len(dataset) < 50:
        logger.error("❌ Not enough data. Exiting.")
        return
    
    logger.info(f"✅ Dataset ready: {len(dataset)} samples")
    
    # Check feature dimensions
    sample = dataset[0]
    num_node_features = sample.x.size(1)
    num_material_features = sample.material_props.size(0)
    
    logger.info(f"\n🔧 Model configuration:")
    logger.info(f"   Node features: {num_node_features}")
    logger.info(f"   Material features: {num_material_features}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, temp = torch.utils.data.random_split(
        dataset, [train_size, val_size + test_size]
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp, [val_size, test_size]
    )
    
    logger.info(f"\n📈 Dataset split:")
    logger.info(f"   Train: {len(train_dataset)}")
    logger.info(f"   Val: {len(val_dataset)}")
    logger.info(f"   Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Create model
    model = EnhancedCrystalTcGNN(
        num_node_features,
        num_material_features,
        hidden_dim=128
    ).to(predictor.device)
    
    logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )
    
    criterion = PhysicsAwareTcLoss(alpha=1.0, beta=0.5, gamma=0.2)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training loop
    logger.info(f"\n🏋️ Training for {num_epochs} epochs...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 15
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in train_loader:
            batch = batch.to(predictor.device)
            optimizer.zero_grad()
            
            output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
            loss, _, _, _ = criterion(output.squeeze(), batch.y.squeeze(), batch.material_props)
            
            # Check for NaN
            if torch.isnan(loss):
                logger.warning(f"⚠️ NaN loss detected in epoch {epoch}, skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        if train_batches == 0:
            logger.error("No valid training batches")
            break
        
        avg_train_loss = train_loss / train_batches
        
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
                loss, _, _, _ = criterion(output.squeeze(), batch.y.squeeze(), batch.material_props)
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_batches += 1
                    predictions.extend(output.squeeze().cpu().numpy())
                    targets.extend(batch.y.squeeze().cpu().numpy())
        
        if val_batches == 0:
            logger.error("No valid validation batches")
            break
        
        avg_val_loss = val_loss / val_batches
        
        # Metrics
        if len(predictions) > 1:
            r2 = r2_score(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
        else:
            r2, mae = 0.0, 0.0
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/improved_tc_model.pt')
            logger.info(f"💾 Saved best model at epoch {epoch}")
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch:3d}/{num_epochs}: "
                       f"Train Loss={avg_train_loss:.3f}, "
                       f"Val Loss={avg_val_loss:.3f}, "
                       f"R²={r2:.3f}, MAE={mae:.2f}K")
        
        if patience_counter >= patience_limit:
            logger.info(f"⏹️ Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*70)
    
    model.load_state_dict(torch.load('models/improved_tc_model.pt'))
    model.eval()
    
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(predictor.device)
            output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
            test_predictions.extend(output.squeeze().cpu().numpy())
            test_targets.extend(batch.y.squeeze().cpu().numpy())
    
    # Calculate metrics
    test_r2 = r2_score(test_targets, test_predictions)
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    
    # Metrics by Tc range
    test_targets = np.array(test_targets)
    test_predictions = np.array(test_predictions)
    
    low_mask = test_targets < 10
    med_mask = (test_targets >= 10) & (test_targets < 50)
    high_mask = test_targets >= 50
    
    logger.info(f"\n📊 OVERALL PERFORMANCE:")
    logger.info(f"   R² Score:  {test_r2:.4f}")
    logger.info(f"   MAE:       {test_mae:.2f} K")
    logger.info(f"   RMSE:      {test_rmse:.2f} K")
    
    logger.info(f"\n📊 PERFORMANCE BY Tc RANGE:")
    
    if np.sum(low_mask) > 1:
        low_r2 = r2_score(test_targets[low_mask], test_predictions[low_mask])
        low_mae = mean_absolute_error(test_targets[low_mask], test_predictions[low_mask])
        logger.info(f"   Low Tc (<10K):     R²={low_r2:.4f}, MAE={low_mae:.2f}K ({np.sum(low_mask)} samples)")
    
    if np.sum(med_mask) > 1:
        med_r2 = r2_score(test_targets[med_mask], test_predictions[med_mask])
        med_mae = mean_absolute_error(test_targets[med_mask], test_predictions[med_mask])
        logger.info(f"   Medium Tc (10-50K): R²={med_r2:.4f}, MAE={med_mae:.2f}K ({np.sum(med_mask)} samples)")
    
    if np.sum(high_mask) > 1:
        high_r2 = r2_score(test_targets[high_mask], test_predictions[high_mask])
        high_mae = mean_absolute_error(test_targets[high_mask], test_predictions[high_mask])
        logger.info(f"   High Tc (>50K):     R²={high_r2:.4f}, MAE={high_mae:.2f}K ({np.sum(high_mask)} samples)")
    
    # Sample predictions
    logger.info(f"\n🔍 SAMPLE PREDICTIONS:")
    indices = np.random.choice(len(test_targets), min(10, len(test_targets)), replace=False)
    for idx in indices:
        error_pct = abs(test_predictions[idx] - test_targets[idx]) / test_targets[idx] * 100
        logger.info(f"   Target: {test_targets[idx]:6.2f}K → Predicted: {test_predictions[idx]:6.2f}K "
                   f"(Error: {error_pct:5.1f}%)")
    
    logger.info("\n" + "="*70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nModel saved to: models/improved_tc_model.pt")
    logger.info(f"\n🎯 Key improvements from enhanced data:")
    logger.info(f"   - Real Fermi energies (not estimated!)")
    logger.info(f"   - Actual coordination numbers")
    logger.info(f"   - Debye temperatures (when available)")
    logger.info(f"   - Element composition details")
    logger.info(f"   - Mechanical properties")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with enhanced Materials Project data')
    parser.add_argument('--max-structures', type=int, default=500,
                       help='Maximum structures to use (default: 500)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    
    try:
        train_improved_model(
            max_structures=args.max_structures,
            num_epochs=args.epochs
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

