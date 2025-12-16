#!/usr/bin/env python3
"""
Complete pipeline for superconductor prediction and validation.

This script runs the full pipeline:
1. Data collection from Materials Project
2. GNN model training with GPU acceleration
3. Candidate identification
4. DFT validation of candidates
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import project modules
from config import load_config, Config
from data_collector import SuperconductorDataCollector
from train_superconductor_gnn import SuperconductorTrainer
from dft_validation import DFTValidator
from gpu_accelerated_training import create_accelerator_from_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("superconductor_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check GPU availability and display information."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s):")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            logger.info(f"  GPU {i}: {gpu_name} with {gpu_memory:.2f} GB memory")
        
        return True
    else:
        logger.warning("No GPU available. Will use CPU for computation.")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete superconductor prediction pipeline"
    )
    
    parser.add_argument("--api-key", default="DjKx0q7YivC5u73uKFIVPif813v7InYq", 
                        help="Materials Project API key")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--models-dir", default="models", help="Path to models directory")
    parser.add_argument("--results-dir", default="results", help="Path to results directory")
    parser.add_argument("--dft-dir", default="dft_validation", help="Path to DFT validation directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--min-tc", type=float, default=50.0, help="Minimum Tc for candidate materials")
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-validation", action="store_true", help="Skip DFT validation step")
    parser.add_argument("--max-results", type=int, default=100, help="Maximum number of materials to fetch")
    
    # GPU configuration arguments
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU device index to use")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision training")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.dft_dir, exist_ok=True)
    
    # Check GPU availability
    has_gpu = check_gpu_availability()
    
    # Load configuration
    if args.config:
        config_obj = load_config(args.config)
    else:
        config_obj = load_config()
    
    # Override GPU settings from command line arguments
    if args.use_gpu:
        config_obj.set("use_gpu", True)
    if args.no_gpu:
        config_obj.set("use_gpu", False)
    if args.gpu_device is not None:
        config_obj.set("gpu_device", args.gpu_device)
    if args.mixed_precision:
        config_obj.set("mixed_precision", True)
    if args.no_mixed_precision:
        config_obj.set("mixed_precision", False)
    
    # Log GPU configuration
    logger.info(f"GPU acceleration: {'Enabled' if config_obj.get('use_gpu') else 'Disabled'}")
    if config_obj.get("use_gpu"):
        logger.info(f"Using GPU device: {config_obj.get('gpu_device')}")
        logger.info(f"Mixed precision: {'Enabled' if config_obj.get('mixed_precision') else 'Disabled'}")
    
    # Step 1: Data Collection
    if not args.skip_collection:
        logger.info("Step 1: Data Collection")
        collector = SuperconductorDataCollector(api_key=args.api_key, data_dir=args.data_dir)
        collector.collect_and_process_all(max_results=args.max_results)
    else:
        logger.info("Skipping data collection step")
    
    # Step 2: Model Training and Candidate Identification
    if not args.skip_training:
        logger.info("Step 2: Model Training and Candidate Identification")
        trainer = SuperconductorTrainer(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            results_dir=args.results_dir
        )
        
        # Run training pipeline
        trainer.run_full_pipeline()
        
        # Get candidates file path
        candidates_file = os.path.join(args.results_dir, "top_10_candidates.csv")
    else:
        logger.info("Skipping model training step")
        # Try to find candidates file
        candidates_file = os.path.join(args.results_dir, "top_10_candidates.csv")
        if not os.path.exists(candidates_file):
            logger.warning(f"Candidates file not found: {candidates_file}")
            if not args.skip_validation:
                logger.error("Cannot proceed with validation without candidates file")
                return 1
    
    # Step 3: DFT Validation
    if not args.skip_validation:
        logger.info("Step 3: DFT Validation")
        
        if not os.path.exists(candidates_file):
            logger.error(f"Candidates file not found: {candidates_file}")
            return 1
        
        validator = DFTValidator(api_key=args.api_key, output_dir=args.dft_dir)
        results = validator.validate_candidates(candidates_file)
        
        # Print final results
        logger.info("\nFinal Results:")
        if "sc_likelihood_score" in results.columns:
            top = results.sort_values("sc_likelihood_score", ascending=False).head(6)
            for i, (_, row) in enumerate(top.iterrows()):
                logger.info(f"{i+1}. {row['material_id']} - {row['formula']} - "
                           f"Predicted Tc: {row['predicted_tc']:.2f}K, "
                           f"SC Likelihood: {row.get('sc_likelihood_score', 'N/A'):.2f}")
    else:
        logger.info("Skipping DFT validation step")
    
    logger.info("Pipeline completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)