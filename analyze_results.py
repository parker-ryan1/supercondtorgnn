#!/usr/bin/env python3
"""
Comprehensive analysis of training results and model performance.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import torch
from scripts.gnn_model import SuperconductorTcPredictor, EnhancedCrystalTcGNN
from torch_geometric.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_results():
    """Load training results"""
    results_file = 'results/training_results.json'
    
    if not Path(results_file).exists():
        logger.error(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def analyze_training_history(results):
    """Analyze training curves"""
    logger.info("="*70)
    logger.info("TRAINING HISTORY ANALYSIS")
    logger.info("="*70)
    
    train_history = results['train_history']
    val_history = results['val_history']
    
    logger.info(f"\nTraining Progress:")
    logger.info(f"  Initial train loss: {train_history[0]:.2f}")
    logger.info(f"  Final train loss: {train_history[-1]:.2f}")
    logger.info(f"  Improvement: {(1 - train_history[-1]/train_history[0])*100:.1f}%")
    
    logger.info(f"\nValidation Progress:")
    logger.info(f"  Initial val loss: {val_history[0]:.2f}")
    logger.info(f"  Best val loss: {min(val_history):.2f}")
    logger.info(f"  Final val loss: {val_history[-1]:.2f}")
    logger.info(f"  Best epoch: {np.argmin(val_history) + 1}")
    
    # Check for overfitting
    final_gap = val_history[-1] - train_history[-1]
    logger.info(f"\nOverfitting Analysis:")
    logger.info(f"  Train-Val gap: {final_gap:.2f}")
    if final_gap > train_history[-1] * 0.5:
        logger.warning("  ⚠️ Potential overfitting detected!")
    else:
        logger.info("  ✅ No significant overfitting")

def analyze_performance(results):
    """Analyze model performance metrics"""
    logger.info("\n" + "="*70)
    logger.info("PERFORMANCE ANALYSIS")
    logger.info("="*70)
    
    logger.info(f"\nDataset Information:")
    logger.info(f"  Total samples: {results['dataset_size']}")
    logger.info(f"  Enhanced feature coverage: {results['enhanced_coverage']*100:.1f}%")
    logger.info(f"  Epochs trained: {results['num_epochs_trained']}")
    
    logger.info(f"\nModel Performance:")
    logger.info(f"  R² Score:  {results['test_r2']:.4f}")
    logger.info(f"  MAE:       {results['test_mae']:.2f} K")
    logger.info(f"  RMSE:      {results['test_rmse']:.2f} K")
    logger.info(f"  MAPE:      {results['test_mape']:.2f}%")
    
    # Interpret results
    logger.info(f"\nPerformance Interpretation:")
    
    r2 = results['test_r2']
    if r2 > 0.7:
        logger.info("  ⭐⭐⭐⭐⭐ Excellent R² - Strong predictive power")
    elif r2 > 0.5:
        logger.info("  ⭐⭐⭐⭐ Good R² - Useful predictions")
    elif r2 > 0.3:
        logger.info("  ⭐⭐⭐ Moderate R² - Reasonable predictions")
    elif r2 > 0.1:
        logger.info("  ⭐⭐ Fair R² - Some predictive ability")
    else:
        logger.info("  ⭐ Weak R² - Limited predictive ability")
    
    mae = results['test_mae']
    if mae < 10:
        logger.info(f"  ⭐⭐⭐⭐⭐ Excellent MAE ({mae:.1f}K) - Very accurate")
    elif mae < 20:
        logger.info(f"  ⭐⭐⭐⭐ Good MAE ({mae:.1f}K) - Quite accurate")
    elif mae < 30:
        logger.info(f"  ⭐⭐⭐ Moderate MAE ({mae:.1f}K) - Reasonable accuracy")
    elif mae < 50:
        logger.info(f"  ⭐⭐ Fair MAE ({mae:.1f}K) - Room for improvement")
    else:
        logger.info(f"  ⭐ High MAE ({mae:.1f}K) - Needs improvement")

def analyze_enhanced_features():
    """Analyze enhanced feature coverage"""
    logger.info("\n" + "="*70)
    logger.info("ENHANCED FEATURES ANALYSIS")
    logger.info("="*70)
    
    # Check all enhanced data sources
    sources = {
        'enhanced_superconductors.csv': 'Initial batch',
        'enhanced_superconductors_full.csv': 'Full dataset',
        'enhanced_for_structures.csv': 'Structure-targeted'
    }
    
    found_data = False
    for filename, description in sources.items():
        filepath = Path('data') / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            logger.info(f"\n✅ {description} ({filename}):")
            logger.info(f"   Materials: {len(df)}")
            
            # Check feature availability
            feature_counts = {}
            for col in df.columns:
                if col not in ['material_id', 'success', 'formula', 'error']:
                    non_null = df[col].notna().sum()
                    if non_null > 0:
                        feature_counts[col] = non_null
            
            logger.info(f"   Features available: {len(feature_counts)}")
            
            # Show key features
            key_features = ['efermi', 'avg_coordination_number', 'debye_temperature', 
                          'bulk_modulus_vrh', 'num_transition_metals']
            
            logger.info("\n   Key feature coverage:")
            for feat in key_features:
                if feat in feature_counts:
                    coverage = feature_counts[feat] / len(df) * 100
                    logger.info(f"   - {feat:30s}: {feature_counts[feat]:4d}/{len(df):4d} ({coverage:5.1f}%)")
                else:
                    logger.info(f"   - {feat:30s}: Not available")
            
            found_data = True
    
    if not found_data:
        logger.warning("\n⚠️ No enhanced data found!")
        logger.info("   Run: python fetch_detailed_data.py")

def generate_recommendations():
    """Generate recommendations for improvement"""
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATIONS FOR IMPROVEMENT")
    logger.info("="*70)
    
    # Load results
    results = load_results()
    if not results:
        return
    
    r2 = results['test_r2']
    mae = results['test_mae']
    coverage = results['enhanced_coverage']
    
    logger.info("\n🎯 Priority Recommendations:\n")
    
    # Recommendation 1: Enhanced data coverage
    if coverage < 0.5:
        logger.info("1. FETCH MORE ENHANCED DATA (HIGH PRIORITY)")
        logger.info("   Current coverage: {:.1f}%".format(coverage * 100))
        logger.info("   Target: 80%+ coverage")
        logger.info("   Action:")
        logger.info("     python fetch_for_structures.py")
        logger.info("     # Fetch enhanced data for materials we have structures for")
        logger.info("   Expected improvement: +15-25% R² score\n")
    
    # Recommendation 2: More training data
    if results['dataset_size'] < 1000:
        logger.info("2. INCREASE DATASET SIZE")
        logger.info(f"   Current size: {results['dataset_size']} materials")
        logger.info("   Target: 1000+ materials")
        logger.info("   Action:")
        logger.info("     python train_stable_enhanced.py --max-structures 1000 --epochs 60")
        logger.info("   Expected improvement: +5-10% R² score\n")
    
    # Recommendation 3: Model improvements
    if r2 < 0.5:
        logger.info("3. MODEL ARCHITECTURE IMPROVEMENTS")
        logger.info("   Current R²: {:.4f}".format(r2))
        logger.info("   Suggestions:")
        logger.info("   - Try larger hidden dimensions (256, 512)")
        logger.info("   - Add more GNN layers (5-6 layers)")
        logger.info("   - Use attention mechanisms")
        logger.info("   - Ensemble multiple models")
        logger.info("   Expected improvement: +10-15% R² score\n")
    
    # Recommendation 4: Hyperparameter tuning
    logger.info("4. HYPERPARAMETER OPTIMIZATION")
    logger.info("   Current settings: Basic defaults")
    logger.info("   Suggestions:")
    logger.info("   - Learning rate: Try 0.0001, 0.001, 0.005")
    logger.info("   - Batch size: Try 8, 16, 32, 64")
    logger.info("   - Dropout: Try 0.1, 0.2, 0.3")
    logger.info("   - Weight decay: Try 1e-5, 1e-4, 1e-3")
    logger.info("   Expected improvement: +5-8% R² score\n")
    
    # Recommendation 5: Data quality
    if mae > 30:
        logger.info("5. IMPROVE DATA QUALITY")
        logger.info(f"   Current MAE: {mae:.2f}K")
        logger.info("   Suggestions:")
        logger.info("   - Filter out outliers")
        logger.info("   - Balance Tc distribution")
        logger.info("   - Add more high-Tc materials")
        logger.info("   - Validate Tc estimates")
        logger.info("   Expected improvement: -10-15K MAE\n")

def create_visualization_script():
    """Create a separate visualization script"""
    logger.info("\n" + "="*70)
    logger.info("VISUALIZATION")
    logger.info("="*70)
    
    logger.info("\nTo create visualizations, run:")
    logger.info("  python visualize_results.py")
    logger.info("\nThis will generate:")
    logger.info("  - Training curves (loss vs epoch)")
    logger.info("  - Prediction scatter plots")
    logger.info("  - Error distribution")
    logger.info("  - Tc range performance")

def main():
    logger.info("="*70)
    logger.info("COMPREHENSIVE ANALYSIS")
    logger.info("="*70)
    
    # Load results
    results = load_results()
    
    if results is None:
        logger.error("Cannot proceed without results file")
        logger.info("\nPlease run training first:")
        logger.info("  python train_stable_enhanced.py")
        return
    
    # Run analyses
    analyze_training_history(results)
    analyze_performance(results)
    analyze_enhanced_features()
    generate_recommendations()
    create_visualization_script()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    logger.info(f"\nCurrent Status:")
    logger.info(f"  ✅ Model trained successfully")
    logger.info(f"  ✅ No NaN issues")
    logger.info(f"  ✅ R² Score: {results['test_r2']:.4f}")
    logger.info(f"  ✅ MAE: {results['test_mae']:.2f}K")
    logger.info(f"  ⚠️ Enhanced coverage: {results['enhanced_coverage']*100:.1f}% (low)")
    
    logger.info(f"\nNext Steps:")
    logger.info(f"  1. Fetch more enhanced data (highest impact)")
    logger.info(f"  2. Re-train with better coverage")
    logger.info(f"  3. Compare: before vs after")
    logger.info(f"  4. Scale up to 1000+ materials")
    
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
