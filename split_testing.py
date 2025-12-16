"""
Comprehensive Split Testing Framework
Tests different data splits, k-fold CV, and stratified sampling strategies
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SplitTester:
    def __init__(self):
        self.results = {
            'split_ratios': [],
            'kfold_cv': [],
            'stratified_cv': [],
            'tc_range_splits': [],
            'random_seed_tests': []
        }
        os.makedirs('results/split_testing', exist_ok=True)
    
    def load_data(self, max_structures=1000):
        """Load and prepare data"""
        logger.info("📦 Loading data...")
        
        from scripts.gnn_model import SuperconductorTcPredictor
        
        predictor = SuperconductorTcPredictor()
        
        # Load enhanced data if available
        enhanced_data_path = 'data/enhanced_superconductors_full.csv'
        if Path(enhanced_data_path).exists():
            enhanced_df = pd.read_csv(enhanced_data_path)
            logger.info(f"  ✅ Loaded {len(enhanced_df)} enhanced materials")
        else:
            enhanced_df = pd.DataFrame()
            logger.warning("  ⚠️ No enhanced data found")
        
        # Process structures
        dataset = predictor.process_structures_for_tc(
            csv_file='data/superconductors.csv',
            structures_dir='structures/superconductors',
            max_structures=max_structures
        )
        
        if len(dataset) < 50:
            logger.error("❌ Not enough data for split testing")
            return None, None
        
        # Extract Tc values for stratification
        tc_values = np.array([data.y.item() for data in dataset])
        
        logger.info(f"  ✅ Loaded {len(dataset)} structures")
        logger.info(f"  📊 Tc range: {tc_values.min():.2f}K - {tc_values.max():.2f}K")
        logger.info(f"  📊 Mean Tc: {tc_values.mean():.2f}K ± {tc_values.std():.2f}K")
        
        return dataset, tc_values
    
    def create_tc_bins(self, tc_values, n_bins=5):
        """Create bins for stratified sampling based on Tc"""
        # Use quantiles for balanced bins
        bins = np.percentile(tc_values, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(tc_values, bins[1:-1])
        return bin_indices
    
    def test_split_ratios(self, dataset, tc_values):
        """Test different train/val/test split ratios"""
        logger.info("\n" + "="*70)
        logger.info("🔬 TEST 1: Different Split Ratios")
        logger.info("="*70)
        
        split_configs = [
            {'train': 0.60, 'val': 0.20, 'test': 0.20, 'name': '60/20/20'},
            {'train': 0.70, 'val': 0.15, 'test': 0.15, 'name': '70/15/15'},
            {'train': 0.80, 'val': 0.10, 'test': 0.10, 'name': '80/10/10'},
            {'train': 0.85, 'val': 0.10, 'test': 0.05, 'name': '85/10/5'},
            {'train': 0.70, 'val': 0.20, 'test': 0.10, 'name': '70/20/10'},
        ]
        
        for config in split_configs:
            logger.info(f"\n📋 Testing split: {config['name']}")
            
            # Split data
            train_size = int(config['train'] * len(dataset))
            val_size = int(config['val'] * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            indices = np.random.permutation(len(dataset))
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
            
            train_dataset = [dataset[i] for i in train_idx]
            val_dataset = [dataset[i] for i in val_idx]
            test_dataset = [dataset[i] for i in test_idx]
            
            logger.info(f"  Sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            
            # Train and evaluate
            metrics = self._train_and_evaluate(
                train_dataset, val_dataset, test_dataset,
                name=f"split_{config['name'].replace('/', '_')}"
            )
            
            if metrics:
                result = {
                    'config': config,
                    'sizes': {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)},
                    'metrics': metrics
                }
                self.results['split_ratios'].append(result)
                
                logger.info(f"  ✅ R²={metrics['test_r2']:.4f}, MAE={metrics['test_mae']:.2f}K")
    
    def test_kfold_cv(self, dataset, tc_values, k=5):
        """Test k-fold cross-validation"""
        logger.info("\n" + "="*70)
        logger.info(f"🔬 TEST 2: {k}-Fold Cross-Validation")
        logger.info("="*70)
        
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
            logger.info(f"\n📋 Fold {fold + 1}/{k}")
            
            # Further split train into train/val
            train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)
            
            train_dataset = [dataset[i] for i in train_idx]
            val_dataset = [dataset[i] for i in val_idx]
            test_dataset = [dataset[i] for i in test_idx]
            
            logger.info(f"  Sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            
            metrics = self._train_and_evaluate(
                train_dataset, val_dataset, test_dataset,
                name=f"kfold_{fold+1}"
            )
            
            if metrics:
                fold_results.append(metrics)
                logger.info(f"  ✅ R²={metrics['test_r2']:.4f}, MAE={metrics['test_mae']:.2f}K")
        
        # Aggregate results
        if fold_results:
            avg_metrics = {
                'test_r2': np.mean([m['test_r2'] for m in fold_results]),
                'test_mae': np.mean([m['test_mae'] for m in fold_results]),
                'test_rmse': np.mean([m['test_rmse'] for m in fold_results]),
                'std_r2': np.std([m['test_r2'] for m in fold_results]),
                'std_mae': np.std([m['test_mae'] for m in fold_results]),
            }
            
            self.results['kfold_cv'].append({
                'k': k,
                'fold_results': fold_results,
                'average': avg_metrics
            })
            
            logger.info(f"\n📊 {k}-Fold CV Summary:")
            logger.info(f"  Avg R²: {avg_metrics['test_r2']:.4f} ± {avg_metrics['std_r2']:.4f}")
            logger.info(f"  Avg MAE: {avg_metrics['test_mae']:.2f} ± {avg_metrics['std_mae']:.2f}K")
    
    def test_stratified_cv(self, dataset, tc_values, k=5):
        """Test stratified k-fold CV based on Tc ranges"""
        logger.info("\n" + "="*70)
        logger.info(f"🔬 TEST 3: Stratified {k}-Fold Cross-Validation")
        logger.info("="*70)
        
        # Create bins for stratification
        bin_indices = self.create_tc_bins(tc_values, n_bins=5)
        
        logger.info(f"  Created {len(np.unique(bin_indices))} Tc bins for stratification")
        for i, bin_idx in enumerate(np.unique(bin_indices)):
            bin_mask = bin_indices == bin_idx
            bin_tc = tc_values[bin_mask]
            logger.info(f"    Bin {i+1}: {np.sum(bin_mask)} samples, Tc range: {bin_tc.min():.1f}-{bin_tc.max():.1f}K")
        
        skfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(skfold.split(dataset, bin_indices)):
            logger.info(f"\n📋 Stratified Fold {fold + 1}/{k}")
            
            # Further split train into train/val (stratified)
            train_bins = bin_indices[train_idx]
            train_idx, val_idx = train_test_split(
                train_idx, test_size=0.15, random_state=42, stratify=train_bins
            )
            
            train_dataset = [dataset[i] for i in train_idx]
            val_dataset = [dataset[i] for i in val_idx]
            test_dataset = [dataset[i] for i in test_idx]
            
            logger.info(f"  Sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            
            metrics = self._train_and_evaluate(
                train_dataset, val_dataset, test_dataset,
                name=f"stratified_fold_{fold+1}"
            )
            
            if metrics:
                fold_results.append(metrics)
                logger.info(f"  ✅ R²={metrics['test_r2']:.4f}, MAE={metrics['test_mae']:.2f}K")
        
        # Aggregate results
        if fold_results:
            avg_metrics = {
                'test_r2': np.mean([m['test_r2'] for m in fold_results]),
                'test_mae': np.mean([m['test_mae'] for m in fold_results]),
                'test_rmse': np.mean([m['test_rmse'] for m in fold_results]),
                'std_r2': np.std([m['test_r2'] for m in fold_results]),
                'std_mae': np.std([m['test_mae'] for m in fold_results]),
            }
            
            self.results['stratified_cv'].append({
                'k': k,
                'fold_results': fold_results,
                'average': avg_metrics
            })
            
            logger.info(f"\n📊 Stratified {k}-Fold CV Summary:")
            logger.info(f"  Avg R²: {avg_metrics['test_r2']:.4f} ± {avg_metrics['std_r2']:.4f}")
            logger.info(f"  Avg MAE: {avg_metrics['test_mae']:.2f} ± {avg_metrics['std_mae']:.2f}K")
    
    def test_random_seeds(self, dataset, tc_values, n_seeds=5):
        """Test model stability across different random seeds"""
        logger.info("\n" + "="*70)
        logger.info(f"🔬 TEST 4: Random Seed Stability ({n_seeds} seeds)")
        logger.info("="*70)
        
        seed_results = []
        for seed in range(42, 42 + n_seeds):
            logger.info(f"\n📋 Testing seed: {seed}")
            
            # Set seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Split data
            indices = np.random.permutation(len(dataset))
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
            
            train_dataset = [dataset[i] for i in train_idx]
            val_dataset = [dataset[i] for i in val_idx]
            test_dataset = [dataset[i] for i in test_idx]
            
            metrics = self._train_and_evaluate(
                train_dataset, val_dataset, test_dataset,
                name=f"seed_{seed}"
            )
            
            if metrics:
                seed_results.append({'seed': seed, 'metrics': metrics})
                logger.info(f"  ✅ R²={metrics['test_r2']:.4f}, MAE={metrics['test_mae']:.2f}K")
        
        # Aggregate
        if seed_results:
            avg_metrics = {
                'test_r2': np.mean([r['metrics']['test_r2'] for r in seed_results]),
                'test_mae': np.mean([r['metrics']['test_mae'] for r in seed_results]),
                'std_r2': np.std([r['metrics']['test_r2'] for r in seed_results]),
                'std_mae': np.std([r['metrics']['test_mae'] for r in seed_results]),
            }
            
            self.results['random_seed_tests'].append({
                'n_seeds': n_seeds,
                'seed_results': seed_results,
                'average': avg_metrics
            })
            
            logger.info(f"\n📊 Random Seed Stability Summary:")
            logger.info(f"  Avg R²: {avg_metrics['test_r2']:.4f} ± {avg_metrics['std_r2']:.4f}")
            logger.info(f"  Avg MAE: {avg_metrics['test_mae']:.2f} ± {avg_metrics['std_mae']:.2f}K")
            logger.info(f"  Coefficient of Variation (R²): {avg_metrics['std_r2']/avg_metrics['test_r2']*100:.1f}%")
    
    def _train_and_evaluate(self, train_dataset, val_dataset, test_dataset, name="model", epochs=30):
        """Train a model and evaluate it"""
        try:
            from scripts.gnn_model import SuperconductorTcPredictor, EnhancedCrystalTcGNN
            from torch_geometric.loader import DataLoader
            
            predictor = SuperconductorTcPredictor()
            
            # Create model
            num_node_features = train_dataset[0].x.size(1)
            num_material_features = train_dataset[0].material_props.size(0)
            
            model = EnhancedCrystalTcGNN(
                num_node_features, 
                num_material_features, 
                hidden_dim=128
            ).to(predictor.device)
            
            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            criterion = torch.nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 10
            
            for epoch in range(epochs):
                # Training
                model.train()
                total_loss = 0
                num_batches = 0
                
                for batch in train_loader:
                    try:
                        batch = batch.to(predictor.device)
                        optimizer.zero_grad()
                        
                        output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
                        loss = criterion(output.squeeze(), batch.y.squeeze())
                        
                        if torch.isnan(loss):
                            continue
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        total_loss += loss.item()
                        num_batches += 1
                    except:
                        continue
                
                if num_batches == 0:
                    continue
                
                # Validation
                model.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        try:
                            batch = batch.to(predictor.device)
                            output = model(batch.x, batch.edge_index, batch.batch, batch.material_props)
                            loss = criterion(output.squeeze(), batch.y.squeeze())
                            
                            if not torch.isnan(loss):
                                val_loss += loss.item()
                                val_batches += 1
                        except:
                            continue
                
                if val_batches > 0:
                    avg_val_loss = val_loss / val_batches
                    scheduler.step(avg_val_loss)
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), f'results/split_testing/{name}_best.pt')
                    else:
                        patience_counter += 1
                        if patience_counter >= max_patience:
                            break
            
            # Load best model and evaluate on test set
            if Path(f'results/split_testing/{name}_best.pt').exists():
                model.load_state_dict(torch.load(f'results/split_testing/{name}_best.pt'))
            
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
                return None
            
            # Calculate metrics
            test_predictions = np.array(test_predictions)
            test_targets = np.array(test_targets)
            
            r2 = r2_score(test_targets, test_predictions)
            mae = mean_absolute_error(test_targets, test_predictions)
            rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
            
            # By Tc range
            low_mask = test_targets < 10
            med_mask = (test_targets >= 10) & (test_targets < 50)
            high_mask = test_targets >= 50
            
            metrics = {
                'test_r2': r2,
                'test_mae': mae,
                'test_rmse': rmse,
                'n_test': len(test_predictions)
            }
            
            if np.sum(low_mask) > 1:
                metrics['low_r2'] = r2_score(test_targets[low_mask], test_predictions[low_mask])
                metrics['low_mae'] = mean_absolute_error(test_targets[low_mask], test_predictions[low_mask])
            
            if np.sum(med_mask) > 1:
                metrics['med_r2'] = r2_score(test_targets[med_mask], test_predictions[med_mask])
                metrics['med_mae'] = mean_absolute_error(test_targets[med_mask], test_predictions[med_mask])
            
            if np.sum(high_mask) > 1:
                metrics['high_r2'] = r2_score(test_targets[high_mask], test_predictions[high_mask])
                metrics['high_mae'] = mean_absolute_error(test_targets[high_mask], test_predictions[high_mask])
            
            return metrics
            
        except Exception as e:
            logger.error(f"  ❌ Training failed: {e}")
            return None
    
    def generate_visualizations(self):
        """Generate visualization plots"""
        logger.info("\n📊 Generating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Split Ratios Comparison
        if self.results['split_ratios']:
            ax = axes[0, 0]
            configs = [r['config']['name'] for r in self.results['split_ratios']]
            r2_scores = [r['metrics']['test_r2'] for r in self.results['split_ratios']]
            mae_scores = [r['metrics']['test_mae'] for r in self.results['split_ratios']]
            
            x = np.arange(len(configs))
            width = 0.35
            
            ax2 = ax.twinx()
            bars1 = ax.bar(x - width/2, r2_scores, width, label='R² Score', color='skyblue')
            bars2 = ax2.bar(x + width/2, mae_scores, width, label='MAE (K)', color='lightcoral')
            
            ax.set_xlabel('Split Configuration')
            ax.set_ylabel('R² Score', color='skyblue')
            ax2.set_ylabel('MAE (K)', color='lightcoral')
            ax.set_title('Performance vs Split Ratio')
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=45)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: K-Fold CV Results
        if self.results['kfold_cv']:
            ax = axes[0, 1]
            cv_result = self.results['kfold_cv'][0]
            folds = list(range(1, len(cv_result['fold_results']) + 1))
            r2_scores = [f['test_r2'] for f in cv_result['fold_results']]
            
            ax.plot(folds, r2_scores, 'o-', linewidth=2, markersize=8, color='green')
            ax.axhline(cv_result['average']['test_r2'], color='red', linestyle='--', 
                      label=f"Mean: {cv_result['average']['test_r2']:.4f}")
            ax.fill_between(folds, 
                           cv_result['average']['test_r2'] - cv_result['average']['std_r2'],
                           cv_result['average']['test_r2'] + cv_result['average']['std_r2'],
                           alpha=0.2, color='green')
            ax.set_xlabel('Fold Number')
            ax.set_ylabel('R² Score')
            ax.set_title(f'{cv_result["k"]}-Fold Cross-Validation')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Stratified vs Regular CV
        if self.results['stratified_cv'] and self.results['kfold_cv']:
            ax = axes[1, 0]
            
            kfold_avg = self.results['kfold_cv'][0]['average']
            strat_avg = self.results['stratified_cv'][0]['average']
            
            categories = ['Regular K-Fold', 'Stratified K-Fold']
            r2_means = [kfold_avg['test_r2'], strat_avg['test_r2']]
            r2_stds = [kfold_avg['std_r2'], strat_avg['std_r2']]
            
            x = np.arange(len(categories))
            bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=10, color=['blue', 'orange'], alpha=0.7)
            ax.set_ylabel('R² Score')
            ax.set_title('Regular vs Stratified Cross-Validation')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, val, std) in enumerate(zip(bars, r2_means, r2_stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}\n±{std:.4f}',
                       ha='center', va='bottom')
        
        # Plot 4: Random Seed Stability
        if self.results['random_seed_tests']:
            ax = axes[1, 1]
            seed_result = self.results['random_seed_tests'][0]
            seeds = [r['seed'] for r in seed_result['seed_results']]
            r2_scores = [r['metrics']['test_r2'] for r in seed_result['seed_results']]
            
            ax.plot(seeds, r2_scores, 'o-', linewidth=2, markersize=8, color='purple')
            ax.axhline(seed_result['average']['test_r2'], color='red', linestyle='--',
                      label=f"Mean: {seed_result['average']['test_r2']:.4f}")
            ax.fill_between(seeds,
                           seed_result['average']['test_r2'] - seed_result['average']['std_r2'],
                           seed_result['average']['test_r2'] + seed_result['average']['std_r2'],
                           alpha=0.2, color='purple')
            ax.set_xlabel('Random Seed')
            ax.set_ylabel('R² Score')
            ax.set_title('Model Stability Across Random Seeds')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/split_testing/comparison_plots.png', dpi=300, bbox_inches='tight')
        logger.info("  ✅ Saved: results/split_testing/comparison_plots.png")
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive report"""
        logger.info("\n📝 Generating report...")
        
        report_file = 'results/split_testing/split_test_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Split Testing Comprehensive Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test 1: Split Ratios
            if self.results['split_ratios']:
                f.write("## 1. Split Ratio Testing\n\n")
                f.write("| Split Config | Train | Val | Test | R² Score | MAE (K) | RMSE (K) |\n")
                f.write("|--------------|-------|-----|------|----------|---------|----------|\n")
                
                for result in self.results['split_ratios']:
                    config = result['config']
                    metrics = result['metrics']
                    f.write(f"| {config['name']} | {result['sizes']['train']} | {result['sizes']['val']} | {result['sizes']['test']} | ")
                    f.write(f"{metrics['test_r2']:.4f} | {metrics['test_mae']:.2f} | {metrics['test_rmse']:.2f} |\n")
                
                # Best split
                best = max(self.results['split_ratios'], key=lambda x: x['metrics']['test_r2'])
                f.write(f"\n**Best Split**: {best['config']['name']} (R²={best['metrics']['test_r2']:.4f})\n\n")
            
            # Test 2: K-Fold CV
            if self.results['kfold_cv']:
                f.write("## 2. K-Fold Cross-Validation\n\n")
                cv_result = self.results['kfold_cv'][0]
                f.write(f"**K**: {cv_result['k']}\n\n")
                f.write(f"**Average R²**: {cv_result['average']['test_r2']:.4f} ± {cv_result['average']['std_r2']:.4f}\n\n")
                f.write(f"**Average MAE**: {cv_result['average']['test_mae']:.2f} ± {cv_result['average']['std_mae']:.2f}K\n\n")
                
                f.write("| Fold | R² Score | MAE (K) | RMSE (K) | N Test |\n")
                f.write("|------|----------|---------|----------|--------|\n")
                for i, fold in enumerate(cv_result['fold_results']):
                    f.write(f"| {i+1} | {fold['test_r2']:.4f} | {fold['test_mae']:.2f} | {fold['test_rmse']:.2f} | {fold['n_test']} |\n")
                f.write("\n")
            
            # Test 3: Stratified CV
            if self.results['stratified_cv']:
                f.write("## 3. Stratified K-Fold Cross-Validation\n\n")
                strat_result = self.results['stratified_cv'][0]
                f.write(f"**K**: {strat_result['k']}\n\n")
                f.write(f"**Average R²**: {strat_result['average']['test_r2']:.4f} ± {strat_result['average']['std_r2']:.4f}\n\n")
                f.write(f"**Average MAE**: {strat_result['average']['test_mae']:.2f} ± {strat_result['average']['std_mae']:.2f}K\n\n")
            
            # Test 4: Random Seeds
            if self.results['random_seed_tests']:
                f.write("## 4. Random Seed Stability Analysis\n\n")
                seed_result = self.results['random_seed_tests'][0]
                f.write(f"**Number of Seeds Tested**: {seed_result['n_seeds']}\n\n")
                f.write(f"**Average R²**: {seed_result['average']['test_r2']:.4f} ± {seed_result['average']['std_r2']:.4f}\n\n")
                f.write(f"**Average MAE**: {seed_result['average']['test_mae']:.2f} ± {seed_result['average']['std_mae']:.2f}K\n\n")
                f.write(f"**Coefficient of Variation (R²)**: {seed_result['average']['std_r2']/seed_result['average']['test_r2']*100:.2f}%\n\n")
                
                if seed_result['average']['std_r2']/seed_result['average']['test_r2'] < 0.05:
                    f.write("✅ **Model is highly stable across different random seeds!**\n\n")
                elif seed_result['average']['std_r2']/seed_result['average']['test_r2'] < 0.1:
                    f.write("🟡 **Model shows good stability with minor variations.**\n\n")
                else:
                    f.write("⚠️ **Model shows significant variability - consider ensemble methods.**\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if self.results['split_ratios']:
                best_split = max(self.results['split_ratios'], key=lambda x: x['metrics']['test_r2'])
                f.write(f"1. **Optimal Split Ratio**: Use {best_split['config']['name']} split for best performance\n")
            
            if self.results['stratified_cv'] and self.results['kfold_cv']:
                strat_r2 = self.results['stratified_cv'][0]['average']['test_r2']
                kfold_r2 = self.results['kfold_cv'][0]['average']['test_r2']
                if strat_r2 > kfold_r2:
                    f.write("2. **Use Stratified Sampling**: Stratified CV shows better performance\n")
                else:
                    f.write("2. **Regular Sampling OK**: No significant benefit from stratification\n")
            
            f.write("3. **Cross-Validation**: Use k-fold CV for robust model evaluation\n")
            f.write("4. **Ensemble Methods**: Consider training multiple models with different splits for ensemble predictions\n")
        
        logger.info(f"  ✅ Saved: {report_file}")
    
    def run_all_tests(self, max_structures=1000):
        """Run all split tests"""
        logger.info("\n" + "="*70)
        logger.info("🚀 COMPREHENSIVE SPLIT TESTING")
        logger.info("="*70 + "\n")
        
        # Load data
        dataset, tc_values = self.load_data(max_structures)
        if dataset is None:
            return
        
        # Run all tests
        self.test_split_ratios(dataset, tc_values)
        self.test_kfold_cv(dataset, tc_values, k=5)
        self.test_stratified_cv(dataset, tc_values, k=5)
        self.test_random_seeds(dataset, tc_values, n_seeds=3)  # Reduced to 3 for speed
        
        # Generate outputs
        self.generate_visualizations()
        self.generate_report()
        
        # Save results
        with open('results/split_testing/all_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info("\n" + "="*70)
        logger.info("✅ SPLIT TESTING COMPLETE!")
        logger.info("="*70)
        logger.info("\nResults saved to: results/split_testing/")
        logger.info("  - split_test_report.md (detailed report)")
        logger.info("  - comparison_plots.png (visualizations)")
        logger.info("  - all_results.json (raw data)")

def main():
    tester = SplitTester()
    tester.run_all_tests(max_structures=500)  # Start with 500 for faster testing

if __name__ == "__main__":
    main()


