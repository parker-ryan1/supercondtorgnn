"""
Iterative Model Improvement Pipeline
Automatically fetches data, trains, evaluates, and improves until performance targets are met
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Performance targets
TARGETS = {
    'overall_r2': 0.90,      # Target overall R²
    'overall_mae': 7.0,      # Target MAE in Kelvin
    'low_tc_r2': 0.60,       # Target R² for low Tc (<10K)
    'medium_tc_r2': 0.70,    # Target R² for medium Tc (10-50K)
    'high_tc_r2': 0.75,      # Target R² for high Tc (>50K)
    'enhanced_coverage': 80.0 # Target enhanced data coverage %
}

MAX_ITERATIONS = 5
ITERATION_LOG_FILE = 'results/iteration_log.json'

class IterativeImprover:
    def __init__(self):
        self.iteration = 0
        self.history = []
        self.current_metrics = {}
        os.makedirs('results', exist_ok=True)
        
    def load_iteration_history(self):
        """Load previous iteration history if exists"""
        if Path(ITERATION_LOG_FILE).exists():
            with open(ITERATION_LOG_FILE, 'r') as f:
                data = json.load(f)
                self.history = data.get('history', [])
                self.iteration = data.get('last_iteration', 0)
                logger.info(f"📜 Loaded history: {len(self.history)} previous iterations")
    
    def save_iteration_history(self):
        """Save iteration history"""
        data = {
            'last_iteration': self.iteration,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        with open(ITERATION_LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 Saved iteration history")
    
    def check_targets_met(self, metrics):
        """Check if all performance targets are met"""
        met = []
        not_met = []
        
        for key, target in TARGETS.items():
            actual = metrics.get(key, 0)
            is_met = actual >= target
            
            if is_met:
                met.append(f"✅ {key}: {actual:.2f} >= {target:.2f}")
            else:
                not_met.append(f"❌ {key}: {actual:.2f} < {target:.2f}")
        
        logger.info("\n🎯 TARGET EVALUATION:")
        for item in met:
            logger.info(f"  {item}")
        for item in not_met:
            logger.info(f"  {item}")
        
        return len(not_met) == 0
    
    def get_material_ids_with_structures(self):
        """Get all material IDs that have structure files"""
        structures_dir = Path('structures/superconductors')
        if not structures_dir.exists():
            logger.error(f"❌ Structures directory not found: {structures_dir}")
            return []
        
        structure_files = list(structures_dir.glob('*.cif'))
        material_ids = [f.stem for f in structure_files]
        logger.info(f"📦 Found {len(material_ids)} materials with structures")
        return material_ids
    
    def check_enhanced_data_coverage(self, material_ids):
        """Check how many materials have enhanced data"""
        enhanced_file = 'data/enhanced_superconductors_full.csv'
        
        if not Path(enhanced_file).exists():
            logger.warning(f"⚠️ Enhanced data file not found: {enhanced_file}")
            return 0, []
        
        enhanced_df = pd.read_csv(enhanced_file)
        enhanced_ids = set(enhanced_df['material_id'].tolist())
        
        coverage = len(enhanced_ids & set(material_ids)) / len(material_ids) * 100
        missing_ids = list(set(material_ids) - enhanced_ids)
        
        logger.info(f"📊 Enhanced data coverage: {coverage:.1f}% ({len(enhanced_ids & set(material_ids))}/{len(material_ids)})")
        logger.info(f"📊 Missing enhanced data for: {len(missing_ids)} materials")
        
        return coverage, missing_ids
    
    def fetch_enhanced_data_batch(self, material_ids, batch_size=100):
        """Fetch enhanced data for a batch of materials"""
        logger.info(f"🔍 Fetching enhanced data for {len(material_ids)} materials...")
        
        # Import the fetch function
        try:
            from fetch_detailed_data import fetch_enhanced_properties
        except ImportError:
            logger.error("❌ Could not import fetch_enhanced_properties")
            return False
        
        try:
            # Fetch in batches
            all_enhanced_data = []
            for i in range(0, len(material_ids), batch_size):
                batch = material_ids[i:i+batch_size]
                logger.info(f"  Fetching batch {i//batch_size + 1}/{(len(material_ids)-1)//batch_size + 1} ({len(batch)} materials)...")
                
                batch_data = fetch_enhanced_properties(batch, max_materials=len(batch))
                all_enhanced_data.extend(batch_data)
                
                logger.info(f"  ✅ Fetched {len(batch_data)} materials")
            
            # Load existing data
            enhanced_file = 'data/enhanced_superconductors_full.csv'
            if Path(enhanced_file).exists():
                existing_df = pd.read_csv(enhanced_file)
                logger.info(f"  📂 Loaded {len(existing_df)} existing enhanced materials")
            else:
                existing_df = pd.DataFrame()
            
            # Combine with new data
            new_df = pd.DataFrame(all_enhanced_data)
            
            if not existing_df.empty:
                # Remove duplicates (keep new data)
                existing_df = existing_df[~existing_df['material_id'].isin(new_df['material_id'])]
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Save combined data
            combined_df.to_csv(enhanced_file, index=False)
            logger.info(f"  💾 Saved {len(combined_df)} total enhanced materials")
            
            # Also save JSON
            json_file = 'data/enhanced_superconductors_full.json'
            with open(json_file, 'w') as f:
                json.dump(all_enhanced_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error fetching enhanced data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_model(self, max_structures=1000):
        """Train model with current data"""
        logger.info(f"🚀 Training model (iteration {self.iteration})...")
        
        try:
            # Import training script
            import train_large_scale
            
            # Run training
            logger.info(f"  Training on up to {max_structures} structures...")
            
            # We'll call the training directly
            from scripts.gnn_model import SuperconductorTcPredictor, EnhancedCrystalTcGNN, PhysicsAwareTcLoss
            from torch_geometric.loader import DataLoader
            from sklearn.model_selection import train_test_split
            
            predictor = SuperconductorTcPredictor()
            
            # Load enhanced data
            enhanced_data_path = 'data/enhanced_superconductors_full.csv'
            if Path(enhanced_data_path).exists():
                enhanced_df = pd.read_csv(enhanced_data_path)
                logger.info(f"  ✅ Loaded enhanced data: {len(enhanced_df)} materials")
            else:
                enhanced_df = pd.DataFrame()
                logger.warning(f"  ⚠️ No enhanced data found")
            
            # Process structures
            dataset = predictor.process_structures_for_tc(
                csv_file='data/superconductors.csv',
                structures_dir='structures/superconductors',
                max_structures=max_structures
            )
            
            if len(dataset) < 50:
                logger.error("❌ Not enough structures for training")
                return None
            
            logger.info(f"  ✅ Processed {len(dataset)} structures")
            
            # Split dataset
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=(val_size + test_size), random_state=42)
            val_idx, test_idx = train_test_split(temp_idx, test_size=(test_size / (val_size + test_size)), random_state=42)
            
            train_dataset = [dataset[i] for i in train_idx]
            val_dataset = [dataset[i] for i in val_idx]
            test_dataset = [dataset[i] for i in test_idx]
            
            logger.info(f"  📊 Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            
            # Create model
            num_node_features = dataset[0].x.size(1)
            num_material_features = dataset[0].material_props.size(0)
            
            model = EnhancedCrystalTcGNN(
                num_node_features, 
                num_material_features, 
                hidden_dim=256
            ).to(predictor.device)
            
            logger.info(f"  🤖 Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            criterion = PhysicsAwareTcLoss(alpha=1.0, beta=0.5, gamma=0.3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 20
            num_epochs = 50
            
            for epoch in range(num_epochs):
                # Training
                model.train()
                total_loss = 0
                num_batches = 0
                
                for batch in train_loader:
                    try:
                        batch = batch.to(predictor.device)
                        optimizer.zero_grad()
                        
                        output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
                        loss, _, _, _ = criterion(output.squeeze(), batch.y.squeeze(), batch.material_props)
                        
                        if torch.isnan(loss):
                            logger.warning(f"  ⚠️ NaN loss detected, skipping batch")
                            continue
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        total_loss += loss.item()
                        num_batches += 1
                    except Exception as e:
                        logger.warning(f"  ⚠️ Batch error: {e}")
                        continue
                
                if num_batches == 0:
                    logger.error("  ❌ No valid batches in epoch")
                    break
                
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
                            loss, _, _, _ = criterion(output.squeeze(), batch.y.squeeze(), batch.material_props)
                            
                            if not torch.isnan(loss):
                                val_loss += loss.item()
                                val_batches += 1
                                predictions.extend(output.squeeze().cpu().numpy().flatten())
                                targets.extend(batch.y.squeeze().cpu().numpy().flatten())
                        except:
                            continue
                
                if val_batches > 0:
                    avg_val_loss = val_loss / val_batches
                    scheduler.step(avg_val_loss)
                    
                    if len(predictions) > 1:
                        val_r2 = r2_score(targets, predictions)
                        val_mae = mean_absolute_error(targets, predictions)
                    else:
                        val_r2 = 0
                        val_mae = 0
                    
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"  Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.2f}, Val Loss={avg_val_loss:.2f}, R²={val_r2:.3f}, MAE={val_mae:.2f}K")
                    
                    # Early stopping
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), f'models/iteration_{self.iteration}_best.pt')
                    else:
                        patience_counter += 1
                        if patience_counter >= max_patience:
                            logger.info(f"  ⏸️ Early stopping at epoch {epoch+1}")
                            break
            
            # Final evaluation
            logger.info(f"  📊 Evaluating final model...")
            model.load_state_dict(torch.load(f'models/iteration_{self.iteration}_best.pt'))
            model.eval()
            
            test_predictions = []
            test_targets = []
            
            with torch.no_grad():
                for batch in test_loader:
                    try:
                        batch = batch.to(predictor.device)
                        output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
                        test_predictions.extend(output.squeeze().cpu().numpy().flatten())
                        test_targets.extend(batch.y.squeeze().cpu().numpy().flatten())
                    except:
                        continue
            
            if len(test_predictions) < 2:
                logger.error("  ❌ Not enough predictions for evaluation")
                return None
            
            # Calculate comprehensive metrics
            test_predictions = np.array(test_predictions)
            test_targets = np.array(test_targets)
            
            overall_r2 = r2_score(test_targets, test_predictions)
            overall_mae = mean_absolute_error(test_targets, test_predictions)
            overall_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
            
            # By Tc range
            low_mask = test_targets < 10
            med_mask = (test_targets >= 10) & (test_targets < 50)
            high_mask = test_targets >= 50
            
            low_r2 = r2_score(test_targets[low_mask], test_predictions[low_mask]) if np.sum(low_mask) > 1 else 0
            med_r2 = r2_score(test_targets[med_mask], test_predictions[med_mask]) if np.sum(med_mask) > 1 else 0
            high_r2 = r2_score(test_targets[high_mask], test_predictions[high_mask]) if np.sum(high_mask) > 1 else 0
            
            low_mae = mean_absolute_error(test_targets[low_mask], test_predictions[low_mask]) if np.sum(low_mask) > 0 else 0
            med_mae = mean_absolute_error(test_targets[med_mask], test_predictions[med_mask]) if np.sum(med_mask) > 0 else 0
            high_mae = mean_absolute_error(test_targets[high_mask], test_predictions[high_mask]) if np.sum(high_mask) > 0 else 0
            
            # Enhanced data coverage
            material_ids = self.get_material_ids_with_structures()
            enhanced_coverage, _ = self.check_enhanced_data_coverage(material_ids)
            
            metrics = {
                'overall_r2': overall_r2,
                'overall_mae': overall_mae,
                'overall_rmse': overall_rmse,
                'low_tc_r2': low_r2,
                'medium_tc_r2': med_r2,
                'high_tc_r2': high_r2,
                'low_tc_mae': low_mae,
                'medium_tc_mae': med_mae,
                'high_tc_mae': high_mae,
                'enhanced_coverage': enhanced_coverage,
                'dataset_size': len(dataset),
                'test_size': len(test_predictions)
            }
            
            logger.info(f"\n  ✅ TRAINING COMPLETE!")
            logger.info(f"  📊 Overall: R²={overall_r2:.4f}, MAE={overall_mae:.2f}K")
            logger.info(f"  📊 Low Tc: R²={low_r2:.4f}, MAE={low_mae:.2f}K")
            logger.info(f"  📊 Med Tc: R²={med_r2:.4f}, MAE={med_mae:.2f}K")
            logger.info(f"  📊 High Tc: R²={high_r2:.4f}, MAE={high_mae:.2f}K")
            logger.info(f"  📊 Enhanced Coverage: {enhanced_coverage:.1f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_iteration(self):
        """Run one complete improvement iteration"""
        self.iteration += 1
        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 ITERATION {self.iteration}/{MAX_ITERATIONS}")
        logger.info(f"{'='*70}\n")
        
        iteration_start = datetime.now()
        
        # Step 1: Check current data coverage
        logger.info("📋 Step 1: Checking data coverage...")
        material_ids = self.get_material_ids_with_structures()
        if not material_ids:
            logger.error("❌ No materials found. Cannot proceed.")
            return False
        
        coverage, missing_ids = self.check_enhanced_data_coverage(material_ids)
        
        # Step 2: Fetch missing enhanced data
        if coverage < TARGETS['enhanced_coverage'] and missing_ids:
            logger.info(f"\n📋 Step 2: Fetching enhanced data for {len(missing_ids)} missing materials...")
            fetch_batch_size = min(200, len(missing_ids))  # Fetch up to 200 at a time
            materials_to_fetch = missing_ids[:fetch_batch_size]
            
            success = self.fetch_enhanced_data_batch(materials_to_fetch, batch_size=50)
            if not success:
                logger.warning("⚠️ Enhanced data fetch had issues, continuing with existing data...")
        else:
            logger.info(f"\n📋 Step 2: Enhanced data coverage sufficient ({coverage:.1f}%), skipping fetch")
        
        # Step 3: Train model
        logger.info(f"\n📋 Step 3: Training model...")
        metrics = self.train_model(max_structures=1000)
        
        if metrics is None:
            logger.error("❌ Training failed")
            return False
        
        # Save metrics
        self.current_metrics = metrics
        iteration_time = (datetime.now() - iteration_start).total_seconds()
        
        iteration_record = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': iteration_time,
            'metrics': metrics
        }
        self.history.append(iteration_record)
        self.save_iteration_history()
        
        # Step 4: Check if targets met
        logger.info(f"\n📋 Step 4: Evaluating targets...")
        targets_met = self.check_targets_met(metrics)
        
        if targets_met:
            logger.info(f"\n🎉 SUCCESS! All targets met in iteration {self.iteration}!")
            return True
        else:
            logger.info(f"\n🔄 Targets not met, will continue improving...")
            return False
    
    def run(self):
        """Run the iterative improvement loop"""
        logger.info("\n" + "="*70)
        logger.info("🚀 STARTING ITERATIVE IMPROVEMENT PIPELINE")
        logger.info("="*70 + "\n")
        
        logger.info("🎯 PERFORMANCE TARGETS:")
        for key, value in TARGETS.items():
            logger.info(f"  • {key}: {value}")
        logger.info("")
        
        self.load_iteration_history()
        
        while self.iteration < MAX_ITERATIONS:
            success = self.run_iteration()
            
            if success:
                logger.info(f"\n{'='*70}")
                logger.info(f"✅ IMPROVEMENT PIPELINE COMPLETE!")
                logger.info(f"   Reached targets in {self.iteration} iterations")
                logger.info(f"{'='*70}\n")
                break
        else:
            logger.info(f"\n{'='*70}")
            logger.info(f"⚠️ REACHED MAX ITERATIONS ({MAX_ITERATIONS})")
            logger.info(f"   Best metrics achieved:")
            if self.current_metrics:
                for key, value in self.current_metrics.items():
                    logger.info(f"     • {key}: {value:.2f}")
            logger.info(f"{'='*70}\n")
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate a summary report of all iterations"""
        report_file = 'results/improvement_summary.md'
        
        with open(report_file, 'w') as f:
            f.write("# Iterative Model Improvement Summary\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Iterations**: {len(self.history)}\n\n")
            
            f.write("## Performance Targets\n\n")
            f.write("| Metric | Target |\n")
            f.write("|--------|--------|\n")
            for key, value in TARGETS.items():
                f.write(f"| {key} | {value} |\n")
            f.write("\n")
            
            f.write("## Iteration History\n\n")
            for record in self.history:
                f.write(f"### Iteration {record['iteration']}\n\n")
                f.write(f"**Time**: {record['timestamp']}\n\n")
                f.write(f"**Duration**: {record['duration_seconds']:.1f}s\n\n")
                
                f.write("**Metrics**:\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in record['metrics'].items():
                    f.write(f"| {key} | {value:.4f} |\n")
                f.write("\n")
            
            f.write("## Final Recommendations\n\n")
            if self.current_metrics:
                coverage = self.current_metrics.get('enhanced_coverage', 0)
                overall_r2 = self.current_metrics.get('overall_r2', 0)
                
                if coverage < TARGETS['enhanced_coverage']:
                    f.write(f"- **Fetch more enhanced data**: Current coverage is {coverage:.1f}%, target is {TARGETS['enhanced_coverage']}%\n")
                
                if overall_r2 < TARGETS['overall_r2']:
                    f.write(f"- **Improve model architecture**: Current R² is {overall_r2:.4f}, target is {TARGETS['overall_r2']}\n")
                    f.write("  - Try larger hidden dimensions\n")
                    f.write("  - Add more layers\n")
                    f.write("  - Use ensemble methods\n")
                
                f.write("\n")
        
        logger.info(f"📝 Summary report saved to: {report_file}")

def main():
    improver = IterativeImprover()
    improver.run()

if __name__ == "__main__":
    main()


