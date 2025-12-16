#!/usr/bin/env python3
"""
Create visualizations of training results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_results():
    """Load training results"""
    with open('results/training_results.json', 'r') as f:
        return json.load(f)

def plot_training_curves(results):
    """Plot training and validation loss curves"""
    train_history = results['train_history']
    val_history = results['val_history']
    epochs = range(1, len(train_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_history, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_history, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300)
    logger.info("✅ Saved: results/training_curves.png")
    plt.close()

def plot_loss_improvement(results):
    """Plot loss improvement percentage"""
    train_history = results['train_history']
    val_history = results['val_history']
    epochs = range(1, len(train_history) + 1)
    
    train_improve = [(1 - loss/train_history[0]) * 100 for loss in train_history]
    val_improve = [(1 - loss/val_history[0]) * 100 for loss in val_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_improve, 'b-', label='Training', linewidth=2)
    plt.plot(epochs, val_improve, 'r-', label='Validation', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.title('Loss Improvement Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/loss_improvement.png', dpi=300)
    logger.info("✅ Saved: results/loss_improvement.png")
    plt.close()

def plot_metrics_summary(results):
    """Plot summary of key metrics"""
    metrics = {
        'R² Score': results['test_r2'],
        'MAE (K)': results['test_mae'] / 100,  # Normalize for display
        'RMSE (K)': results['test_rmse'] / 100,  # Normalize for display
        'Coverage (%)': results['enhanced_coverage'] * 100
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(metrics.keys(), metrics.values(), 
                   color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/metrics_summary.png', dpi=300)
    logger.info("✅ Saved: results/metrics_summary.png")
    plt.close()

def create_summary_table(results):
    """Create a summary table visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = [
        ['Metric', 'Value', 'Rating'],
        ['', '', ''],
        ['Dataset Size', f"{results['dataset_size']}", '✓'],
        ['Enhanced Coverage', f"{results['enhanced_coverage']*100:.1f}%", '⚠' if results['enhanced_coverage'] < 0.5 else '✓'],
        ['Epochs Trained', f"{results['num_epochs_trained']}", '✓'],
        ['', '', ''],
        ['R² Score', f"{results['test_r2']:.4f}", '★★' if results['test_r2'] > 0.2 else '★'],
        ['MAE', f"{results['test_mae']:.2f} K", '★★' if results['test_mae'] < 40 else '★'],
        ['RMSE', f"{results['test_rmse']:.2f} K", '★★' if results['test_rmse'] < 60 else '★'],
        ['MAPE', f"{results['test_mape']:.2f}%", '★★★'],
        ['', '', ''],
        ['Training Improvement', f"{(1 - results['train_history'][-1]/results['train_history'][0])*100:.1f}%", '✓'],
        ['Validation Best', f"{min(results['val_history']):.2f}", '✓'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style separators
    for col in range(3):
        table[(1, col)].set_facecolor('#ecf0f1')
        table[(5, col)].set_facecolor('#ecf0f1')
        table[(10, col)].set_facecolor('#ecf0f1')
    
    plt.title('Training Summary Report', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/summary_table.png', dpi=300, bbox_inches='tight')
    logger.info("✅ Saved: results/summary_table.png")
    plt.close()

def main():
    logger.info("="*70)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*70)
    
    # Check if results exist
    if not Path('results/training_results.json').exists():
        logger.error("❌ No training results found!")
        logger.info("Please run: python train_stable_enhanced.py")
        return
    
    # Load results
    results = load_results()
    
    # Create visualizations
    logger.info("\nGenerating plots...")
    plot_training_curves(results)
    plot_loss_improvement(results)
    plot_metrics_summary(results)
    create_summary_table(results)
    
    logger.info("\n" + "="*70)
    logger.info("VISUALIZATIONS COMPLETE!")
    logger.info("="*70)
    logger.info("\nGenerated files:")
    logger.info("  📊 results/training_curves.png")
    logger.info("  📊 results/loss_improvement.png")
    logger.info("  📊 results/metrics_summary.png")
    logger.info("  📊 results/summary_table.png")
    logger.info("\nOpen these files to see your results!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()


