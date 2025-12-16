#!/usr/bin/env python3
"""
Quick start script for the Superconductor Analysis and Prediction project.

This script provides a simple way to run the main components of the project.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import project modules
from config import load_config
from data_collector import SuperconductorDataCollector
from structure_visualizer import process_structures
from data_preprocessing import prepare_superconductor_data
from gnn_model import SuperconductorTcPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("superconductor.log")
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ["data", "models", "structures", "visualization"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        logger.info(f"Directory {dir_name} is ready")

def collect_data(api_key=None, limit=100):
    """Collect data from various sources."""
    logger.info("Starting data collection")
    
    # Create data collector
    collector = SuperconductorDataCollector(api_key=api_key)
    
    # Collect all data
    results = collector.collect_and_process_all(limit=limit)
    
    logger.info("Data collection completed")
    return results

def visualize_structures(limit=None):
    """Generate visualizations for collected structures."""
    logger.info("Starting structure visualization")
    
    # Process Ti compounds
    ti_file = Path("data/ti_compounds_structures.json")
    if ti_file.exists():
        import json
        with open(ti_file, 'r') as f:
            ti_structures = json.load(f)
        process_structures(
            ti_structures,
            output_dir="visualization/ti_compounds",
            desc="Ti compounds",
            limit=limit
        )
    
    # Process superconductors
    sc_file = Path("data/superconductors_structures.json")
    if sc_file.exists():
        import json
        with open(sc_file, 'r') as f:
            sc_structures = json.load(f)
        process_structures(
            sc_structures,
            output_dir="visualization/superconductors",
            desc="superconductors",
            limit=limit
        )
    
    logger.info("Structure visualization completed")

def prepare_data():
    """Prepare data for model training."""
    logger.info("Preparing data for model training")
    
    dataset, norm_stats, (train_indices, val_indices, test_indices) = prepare_superconductor_data()
    
    logger.info(f"Dataset prepared with {len(dataset)} materials")
    logger.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return dataset, norm_stats, (train_indices, val_indices, test_indices)

def train_model():
    """Train the GNN model."""
    logger.info("Starting model training")
    
    # Prepare data
    dataset, norm_stats, (train_indices, val_indices, test_indices) = prepare_data()
    
    # Create data loaders
    from torch_geometric.loader import DataLoader
    
    train_loader = DataLoader([dataset[i] for i in train_indices], batch_size=32, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val_indices], batch_size=32)
    test_loader = DataLoader([dataset[i] for i in test_indices], batch_size=32)
    
    # Create predictor
    predictor = SuperconductorTcPredictor()
    
    # Create model
    sample_data = dataset[0]
    num_node_features = sample_data.x.shape[1]
    num_global_features = sample_data.global_features.shape[0] if hasattr(sample_data, 'global_features') else 0
    
    model = predictor.create_model(
        num_node_features=num_node_features,
        num_global_features=num_global_features,
        use_attention=True
    )
    
    # Train model
    history = predictor.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        early_stopping_patience=10,
        log_dir="models/logs"
    )
    
    # Evaluate on test set
    test_loss = predictor.evaluate(test_loader)
    logger.info(f"Test loss: {test_loss:.4f}")
    
    # Save final model
    predictor.save_model("models/final_model.pt")
    
    logger.info("Model training completed")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Superconductor Analysis and Prediction"
    )
    
    parser.add_argument("--setup", action="store_true", help="Setup directories")
    parser.add_argument("--collect", action="store_true", help="Collect data")
    parser.add_argument("--visualize", action="store_true", help="Visualize structures")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--api-key", help="Materials Project API key")
    parser.add_argument("--limit", type=int, default=100, help="Limit the number of structures")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        load_config(args.config)
    
    # Get API key
    api_key = args.api_key or os.getenv("MP_API_KEY")
    
    # Run requested steps
    if args.setup or args.all:
        setup_directories()
    
    if args.collect or args.all:
        if not api_key:
            logger.error("API key is required for data collection")
            return 1
        collect_data(api_key=api_key, limit=args.limit)
    
    if args.visualize or args.all:
        visualize_structures(limit=args.limit)
    
    if args.train or args.all:
        train_model()
    
    # If no specific action was requested, show help
    if not (args.setup or args.collect or args.visualize or args.train or args.all):
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
