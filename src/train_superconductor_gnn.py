"""
Training module for superconductor GNN models.

This module handles the training pipeline for superconductor
critical temperature prediction using Graph Neural Networks.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import time

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from config import config
from data_preprocessing import SuperconductorDataset, normalize_features, split_dataset
from gnn_model import SuperconductorTcPredictor
from data_collector import SuperconductorDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperconductorTrainer:
    """
    Trainer for superconductor critical temperature prediction.
    
    This class handles the full training pipeline, including
    data loading, preprocessing, model training, and evaluation.
    """
    
    def __init__(self, 
                 data_dir: Optional[str] = None,
                 models_dir: Optional[str] = None,
                 results_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Directory with data files
            models_dir: Directory to save/load models
            results_dir: Directory to save results
            device: Device to use (cuda or cpu)
        """
        # Set directories
        self.data_dir = data_dir or config.get("data_dir", "data")
        self.models_dir = models_dir or config.get("models_dir", "models")
        self.results_dir = results_dir or config.get("results_dir", "results")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu")
        
        # Initialize components
        self.predictor = SuperconductorTcPredictor(model_dir=self.models_dir, device=self.device)
        
        # Initialize data attributes
        self.dataset = None
        self.norm_stats = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        logger.info("SuperconductorTrainer initialized")
    
    def load_data(self):
        """Load and preprocess the dataset."""
        logger.info("Loading dataset...")
        
        # Set file paths
        structures_file = os.path.join(self.data_dir, config.get("superconductors_structures_file"))
        properties_file = os.path.join(self.data_dir, config.get("superconductors_file"))
        
        # Create dataset
        self.dataset = SuperconductorDataset(
            structures_file=structures_file,
            properties_file=properties_file,
            target_property="critical_temp"
        )
        
        logger.info(f"Dataset loaded with {len(self.dataset)} materials")
        
        # Normalize features
        logger.info("Normalizing features...")
        self.norm_stats, self.dataset = normalize_features(self.dataset)
        
        # Split dataset
        logger.info("Splitting dataset...")
        self.train_indices, self.val_indices, self.test_indices = split_dataset(
            self.dataset,
            test_ratio=config.get("test_split", 0.1),
            val_ratio=config.get("val_split", 0.1),
            random_state=42
        )
        
        logger.info(f"Train: {len(self.train_indices)}, Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")
        
        # Create data loaders
        batch_size = config.get("batch_size", 32)
        
        # For small datasets, use a smaller batch size
        if len(self.train_indices) < batch_size:
            batch_size = max(1, len(self.train_indices) // 2)
        
        self.train_loader = DataLoader(
            self.dataset, 
            batch_size=batch_size,
            sampler=self.train_indices
        )
        
        self.val_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.val_indices
        )
        
        self.test_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.test_indices
        )
        
        # Fix the global features for each batch
        self._fix_global_features()
        
        logger.info("Data loading and preparation completed")
    
    def _fix_global_features(self):
        """Fix global features to ensure correct batch dimension."""
        for batch in self.train_loader:
            if hasattr(batch, 'global_features'):
                batch_size = batch.num_graphs
                if batch.global_features.dim() == 2 and batch.global_features.size(0) != batch_size:
                    # Replicate global features for each graph in the batch
                    global_features = batch.global_features.repeat(batch_size, 1)
                    batch.global_features = global_features
        
        for batch in self.val_loader:
            if hasattr(batch, 'global_features'):
                batch_size = batch.num_graphs
                if batch.global_features.dim() == 2 and batch.global_features.size(0) != batch_size:
                    # Replicate global features for each graph in the batch
                    global_features = batch.global_features.repeat(batch_size, 1)
                    batch.global_features = global_features
        
        for batch in self.test_loader:
            if hasattr(batch, 'global_features'):
                batch_size = batch.num_graphs
                if batch.global_features.dim() == 2 and batch.global_features.size(0) != batch_size:
                    # Replicate global features for each graph in the batch
                    global_features = batch.global_features.repeat(batch_size, 1)
                    batch.global_features = global_features
    
    def train_model(self):
        """Train the GNN model."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        logger.info("Creating model...")
        
        # Get feature dimensions from the first data point
        data = self.dataset[0]
        num_node_features = data.x.size(1)
        num_global_features = data.global_features.size(0) if hasattr(data, 'global_features') else 0
        
        # Create model
        model = self.predictor.create_model(
            num_node_features=num_node_features,
            num_global_features=num_global_features,
            use_attention=True  # Use attention mechanism for better performance
        )
        
        # Set up training parameters
        num_epochs = config.get("num_epochs", 200)
        early_stopping_patience = config.get("early_stopping_patience", 20)
        
        # Create log directory for TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.results_dir, "logs", f"run_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Train the model
        logger.info(f"Starting training for {num_epochs} epochs...")
        history = self.predictor.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            log_dir=log_dir
        )
        
        # Save training history
        self._save_training_history(history)
        
        # Evaluate on test set
        test_loss = self.predictor.evaluate(self.test_loader)
        logger.info(f"Test Loss (MSE): {test_loss:.4f}")
        
        # Save the final model
        model_path = self.predictor.save_model(os.path.join(self.models_dir, "final_model.pt"))
        logger.info(f"Model saved to {model_path}")
        
        return history
    
    def _save_training_history(self, history: Dict[str, List[float]]):
        """
        Save training history to CSV and plot learning curves.
        
        Args:
            history: Dictionary with training history
        """
        # Save to CSV
        history_df = pd.DataFrame(history)
        history_path = os.path.join(self.results_dir, "training_history.csv")
        history_df.to_csv(history_path, index=False)
        
        # Plot learning curves
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train")
        if "val_loss" in history:
            plt.plot(history["val_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot learning rate
        if "learning_rate" in history:
            plt.subplot(1, 2, 2)
            plt.plot(history["learning_rate"])
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "learning_curves.png"), dpi=300)
    
    def predict_candidates(self, top_k: int = 10) -> pd.DataFrame:
        """
        Predict top-k candidate materials with highest critical temperature.
        
        Args:
            top_k: Number of top candidates to return
            
        Returns:
            DataFrame with top candidate materials
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        logger.info(f"Predicting top-{top_k} candidate materials...")
        
        # Make predictions on all materials
        all_loader = DataLoader(self.dataset, batch_size=32)
        predictions, true_values, material_ids = self.predictor.predict(all_loader)
        
        # Create DataFrame with results
        results = []
        for i in range(len(predictions)):
            # Handle both scalar and array predictions
            pred_value = float(predictions[i][0]) if predictions[i].size > 1 else float(predictions[i])
            true_value = float(true_values[i][0]) if true_values[i].size > 1 else float(true_values[i]) if true_values is not None else None
            
            results.append({
                "material_id": material_ids[i] if material_ids else f"material_{i}",
                "predicted_tc": pred_value,
                "actual_tc": true_value
            })
        
        results_df = pd.DataFrame(results)
        
        # Sort by predicted Tc (descending)
        results_df = results_df.sort_values("predicted_tc", ascending=False)
        
        # Save all predictions
        results_df.to_csv(os.path.join(self.results_dir, "all_predictions.csv"), index=False)
        
        # Get top-k candidates
        top_candidates = results_df.head(top_k)
        top_candidates.to_csv(os.path.join(self.results_dir, f"top_{top_k}_candidates.csv"), index=False)
        
        logger.info(f"Top-{top_k} candidates saved to {self.results_dir}/top_{top_k}_candidates.csv")
        
        return top_candidates
    
    def validate_candidates(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate candidate materials using DFT calculations.
        
        Args:
            candidates_df: DataFrame with candidate materials
            
        Returns:
            DataFrame with validation results
        """
        logger.info(f"Validating {len(candidates_df)} candidate materials...")
        
        # Initialize Materials Project API client
        api_key = config.get("mp_api_key")
        collector = SuperconductorDataCollector(api_key=api_key)
        
        # Validate each candidate
        validation_results = []
        
        for _, row in candidates_df.iterrows():
            material_id = row["material_id"]
            predicted_tc = row["predicted_tc"]
            
            try:
                # Get DFT data for the material
                dft_data = collector.get_dft_data(material_id)
                
                # Calculate stability metrics
                e_above_hull = dft_data.get("e_above_hull")
                is_stable = e_above_hull < 0.1 if e_above_hull is not None else False
                
                # Get electronic properties
                band_gap = dft_data.get("band_gap", 0)
                is_metal = band_gap < 0.1
                
                # Get density
                density = dft_data.get("density")
                
                # Get formation energy
                formation_energy = dft_data.get("formation_energy_per_atom")
                
                # Calculate validation score (higher is better)
                # Factors: predicted Tc, stability, metallic character
                validation_score = 0
                if predicted_tc is not None:
                    validation_score += min(100, predicted_tc) / 100  # Max 1 point for Tc
                if is_stable:
                    validation_score += 1  # 1 point for stability
                if is_metal:
                    validation_score += 1  # 1 point for metallic character
                
                # Store validation results
                validation_results.append({
                    "material_id": material_id,
                    "predicted_tc": predicted_tc,
                    "e_above_hull": e_above_hull,
                    "is_stable": is_stable,
                    "band_gap": band_gap,
                    "is_metal": is_metal,
                    "density": density,
                    "formation_energy": formation_energy,
                    "validation_score": validation_score
                })
                
                logger.info(f"Validated {material_id}: Score = {validation_score:.2f}")
                
            except Exception as e:
                logger.error(f"Error validating {material_id}: {str(e)}")
                
                # Store partial results
                validation_results.append({
                    "material_id": material_id,
                    "predicted_tc": predicted_tc,
                    "validation_error": str(e)
                })
        
        # Create DataFrame with validation results
        validation_df = pd.DataFrame(validation_results)
        
        # Sort by validation score (descending)
        if "validation_score" in validation_df.columns:
            validation_df = validation_df.sort_values("validation_score", ascending=False)
        
        # Save validation results
        validation_df.to_csv(os.path.join(self.results_dir, "validation_results.csv"), index=False)
        
        logger.info(f"Validation results saved to {self.results_dir}/validation_results.csv")
        
        return validation_df
    
    def run_full_pipeline(self):
        """Run the full training and prediction pipeline."""
        # Step 1: Load and preprocess data
        self.load_data()
        
        # Step 2: Train model
        self.train_model()
        
        # Step 3: Predict top candidates
        top_k = config.get("top_k_candidates", 10)
        candidates_df = self.predict_candidates(top_k=top_k)
        
        # Step 4: Validate candidates (if enabled)
        if config.get("validate_candidates", True):
            validation_df = self.validate_candidates(candidates_df)
            return validation_df
        else:
            return candidates_df

def main():
    """Main function for running the training pipeline."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create trainer
    trainer = SuperconductorTrainer()
    
    # Run full pipeline
    results = trainer.run_full_pipeline()
    
    # Print top candidates
    print("\nTop candidate materials:")
    print(results)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)