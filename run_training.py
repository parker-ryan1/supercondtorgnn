#!/usr/bin/env python3
"""
Training script for superconductor Tc prediction model.

This script trains a Graph Neural Network model to predict superconducting
transition temperatures (Tc) of materials from the Materials Project database.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import project modules
from train_superconductor_gnn import SuperconductorTrainer
from crystal_graph_features import CrystalGraphFeatureExtractor
from enhanced_gnn_model import SuperconductorGNN, ResidualGNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("superconductor_training.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train a GNN model for superconductor Tc prediction"
    )
    
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--models-dir", default="models", help="Path to models directory")
    parser.add_argument("--results-dir", default="results", help="Path to results directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--model-type", choices=["standard", "residual"], default="residual", 
                        help="GNN model type")
    parser.add_argument("--min-tc", type=float, default=50.0, help="Minimum Tc for candidate materials")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create trainer
    trainer = SuperconductorTrainer(config_file=args.config)
    
    # Override config with command-line arguments
    trainer.data_dir = args.data_dir
    trainer.models_dir = args.models_dir
    trainer.results_dir = args.results_dir
    trainer.num_epochs = args.epochs
    trainer.batch_size = args.batch_size
    trainer.learning_rate = args.learning_rate
    trainer.hidden_dim = args.hidden_dim
    
    # Print training configuration
    logger.info("Training configuration:")
    logger.info(f"  Data directory: {trainer.data_dir}")
    logger.info(f"  Models directory: {trainer.models_dir}")
    logger.info(f"  Results directory: {trainer.results_dir}")
    logger.info(f"  Epochs: {trainer.num_epochs}")
    logger.info(f"  Batch size: {trainer.batch_size}")
    logger.info(f"  Learning rate: {trainer.learning_rate}")
    logger.info(f"  Hidden dimension: {trainer.hidden_dim}")
    logger.info(f"  Model type: {args.model_type}")
    
    # Run the full training pipeline
    trainer.run_full_pipeline()
    
    # Identify candidates with specified minimum Tc
    candidates = trainer.identify_candidates(min_tc=args.min_tc)
    
    # Print top candidates
    logger.info(f"\nTop candidates with predicted Tc > {args.min_tc}K:")
    for i, (_, row) in enumerate(candidates.head(6).iterrows()):
        logger.info(f"{i+1}. {row['material_id']} - {row['formula']} - "
                   f"Predicted Tc: {row['predicted_tc']:.2f}K")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
