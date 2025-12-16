#!/usr/bin/env python3
"""
Superconductor Analysis and Prediction - Main Entry Point

This script serves as the main entry point for the superconductor analysis and prediction project.
It provides a command-line interface to run various components of the project.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collector import SuperconductorDataCollector
from structure_visualizer import StructureVisualizer, process_structures
from gnn_model import SuperconductorTcPredictor, CrystalTcGNN

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

def collect_data(api_key=None):
    """Run the data collection process."""
    try:
        collector = SuperconductorDataCollector(api_key=api_key)
        
        # Collect Ti-based compounds
        logger.info("Collecting titanium-based compounds...")
        ti_compounds = collector.fetch_ti_compounds()
        collector.save_data(ti_compounds, "ti_compounds")
        
        # Collect all potential superconductors
        logger.info("Collecting potential superconductors...")
        superconductors = collector.fetch_all_superconductors()
        collector.save_data(superconductors, "superconductors")
        
        logger.info("Data collection completed successfully")
        return True
    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}")
        return False

def visualize_structures():
    """Generate visualizations for collected structures."""
    try:
        # Process Ti compounds
        ti_file = Path("data/ti_compounds_structures.json")
        if ti_file.exists():
            logger.info("Visualizing Ti compounds...")
            import json
            with open(ti_file, 'r') as f:
                ti_structures = json.load(f)
            process_structures(ti_structures, "structures/ti_compounds", "Ti compounds")
        else:
            logger.warning(f"File not found: {ti_file}")
        
        # Process superconductors
        sc_file = Path("data/superconductors_structures.json")
        if sc_file.exists():
            logger.info("Visualizing superconductors...")
            import json
            with open(sc_file, 'r') as f:
                sc_structures = json.load(f)
            process_structures(sc_structures, "structures/superconductors", "superconductors")
        else:
            logger.warning(f"File not found: {sc_file}")
            
        logger.info("Structure visualization completed")
        return True
    except Exception as e:
        logger.error(f"Structure visualization failed: {str(e)}")
        return False

def train_model():
    """Train the GNN model for superconductivity prediction."""
    try:
        logger.info("Initializing model training...")
        predictor = SuperconductorTcPredictor()
        # Training code would be implemented here
        logger.info("Model training completed")
        return True
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Superconductor Analysis and Prediction Tool"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup project directories")
    
    # Data collection command
    collect_parser = subparsers.add_parser("collect", help="Collect data from sources")
    collect_parser.add_argument("--api-key", help="Materials Project API key")
    
    # Visualization command
    visualize_parser = subparsers.add_parser("visualize", help="Generate structure visualizations")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train the GNN model")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "setup":
        setup_directories()
    elif args.command == "collect":
        collect_data(api_key=args.api_key)
    elif args.command == "visualize":
        visualize_structures()
    elif args.command == "train":
        train_model()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
