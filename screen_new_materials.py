#!/usr/bin/env python3
"""
Screen new superconductor candidates from Materials Project database.
Predicts Tc for materials NOT in the training set.
"""

import torch
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from gnn_model import SuperconductorTcPredictor, EnhancedCrystalTcGNN
from pymatgen.core import Structure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_trained_materials(csv_file: str, max_trained: int = 1000) -> set:
    """Load list of material IDs that were used in training"""
    try:
        df = pd.read_csv(csv_file)
        # Get first max_trained material IDs (these were used in training)
        trained_ids = set(df['material_id'].head(max_trained).values)
        logger.info(f"Loaded {len(trained_ids)} material IDs from training set")
        return trained_ids
    except Exception as e:
        logger.warning(f"Could not load trained materials: {e}")
        return set()

def screen_new_materials(
    predictor,
    model,
    structures_dir: str,
    csv_file: str,
    trained_materials: set,
    max_screen: int = 500,
    min_tc_threshold: float = 10.0
):
    """Screen new materials and predict their Tc"""
    
    logger.info(f"Screening new materials from {structures_dir}...")
    
    # Get all structure files
    structure_files = list(Path(structures_dir).glob("*.cif"))
    logger.info(f"Found {len(structure_files)} total structure files")
    
    # Load CSV for material properties
    try:
        csv_data = pd.read_csv(csv_file)
        logger.info(f"Loaded CSV with {len(csv_data)} entries")
    except Exception as e:
        logger.error(f"Could not load CSV: {e}")
        return []
    
    # Filter for NEW materials (not in training set)
    new_structure_files = []
    for structure_file in structure_files:
        material_id = structure_file.stem
        if material_id not in trained_materials:
            new_structure_files.append(structure_file)
    
    logger.info(f"Found {len(new_structure_files)} NEW materials (not in training)")
    
    # Limit screening
    if len(new_structure_files) > max_screen:
        logger.info(f"Limiting to {max_screen} materials for screening")
        new_structure_files = new_structure_files[:max_screen]
    
    # Screen materials
    results = []
    model.eval()
    
    logger.info(f"\nScreening {len(new_structure_files)} new materials...")
    
    with torch.no_grad():
        for i, structure_file in enumerate(tqdm(new_structure_files, desc="Screening")):
            try:
                material_id = structure_file.stem
                
                # Get material properties from CSV
                csv_match = csv_data[csv_data['material_id'] == material_id]
                if csv_match.empty:
                    continue
                
                csv_row = csv_match.iloc[0]
                
                # Load structure
                structure = Structure.from_file(str(structure_file))
                
                # Get material properties
                material_props = {
                    'formation_energy_per_atom': csv_row.get('formation_energy_per_atom', -1.0),
                    'band_gap': csv_row.get('band_gap', 0.0),
                    'density': structure.density,
                    'is_metal': csv_row.get('is_metal', True)
                }
                
                # Predict Tc
                predicted_tc = predictor.predict_tc(model, structure, material_props)
                
                # Store result
                result = {
                    'material_id': material_id,
                    'formula': csv_row.get('formula', 'Unknown'),
                    'predicted_tc': predicted_tc,
                    'formation_energy_per_atom': material_props['formation_energy_per_atom'],
                    'band_gap': material_props['band_gap'],
                    'density': structure.density,
                    'is_metal': material_props['is_metal'],
                    'spacegroup': csv_row.get('spacegroup', 'Unknown'),
                    'nelements': csv_row.get('nelements', 0),
                    'nsites': csv_row.get('nsites', 0)
                }
                
                results.append(result)
                
                # Log promising candidates in real-time
                if predicted_tc > min_tc_threshold:
                    logger.info(f"  🌟 Promising: {material_id} ({result['formula']}) "
                               f"- Predicted Tc: {predicted_tc:.2f}K")
                
            except Exception as e:
                logger.debug(f"Error processing {structure_file}: {e}")
                continue
    
    logger.info(f"\n✅ Screened {len(results)} new materials successfully")
    return results

def save_results(results: list, output_dir: str = "results"):
    """Save screening results"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    if not results:
        logger.warning("No results to save!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by predicted Tc (descending)
    df = df.sort_values('predicted_tc', ascending=False)
    
    # Save all predictions
    all_predictions_file = Path(output_dir) / "new_materials_predictions.csv"
    df.to_csv(all_predictions_file, index=False)
    logger.info(f"✅ Saved all predictions to {all_predictions_file}")
    
    # Save top candidates (Tc > 50K)
    high_tc_candidates = df[df['predicted_tc'] > 50.0]
    if len(high_tc_candidates) > 0:
        high_tc_file = Path(output_dir) / "high_tc_candidates.csv"
        high_tc_candidates.to_csv(high_tc_file, index=False)
        logger.info(f"✅ Saved {len(high_tc_candidates)} high-Tc candidates (>50K) to {high_tc_file}")
    
    # Save top 20 candidates
    top_20 = df.head(20)
    top_20_file = Path(output_dir) / "top_20_new_candidates.csv"
    top_20.to_csv(top_20_file, index=False)
    logger.info(f"✅ Saved top 20 candidates to {top_20_file}")
    
    # Generate statistics
    logger.info("\n" + "="*70)
    logger.info("SCREENING STATISTICS")
    logger.info("="*70)
    logger.info(f"Total materials screened: {len(df)}")
    logger.info(f"Mean predicted Tc: {df['predicted_tc'].mean():.2f}K")
    logger.info(f"Std predicted Tc: {df['predicted_tc'].std():.2f}K")
    logger.info(f"Max predicted Tc: {df['predicted_tc'].max():.2f}K")
    logger.info(f"Min predicted Tc: {df['predicted_tc'].min():.2f}K")
    logger.info(f"\nCandidates by Tc range:")
    logger.info(f"  Very High (>100K): {len(df[df['predicted_tc'] > 100])} materials")
    logger.info(f"  High (50-100K): {len(df[(df['predicted_tc'] > 50) & (df['predicted_tc'] <= 100)])} materials")
    logger.info(f"  Medium (10-50K): {len(df[(df['predicted_tc'] > 10) & (df['predicted_tc'] <= 50)])} materials")
    logger.info(f"  Low (<10K): {len(df[df['predicted_tc'] <= 10])} materials")
    logger.info("="*70)
    
    # Show top 10
    logger.info("\n🏆 TOP 10 NEW SUPERCONDUCTOR CANDIDATES:")
    logger.info("="*70)
    for i, row in top_20.head(10).iterrows():
        logger.info(f"{i+1:2d}. {row['material_id']:15s} | {row['formula']:20s} | "
                   f"Tc: {row['predicted_tc']:6.2f}K | "
                   f"Spacegroup: {row['spacegroup']}")
    logger.info("="*70)
    
    return df

def main():
    logger.info("="*70)
    logger.info("SUPERCONDUCTOR CANDIDATE SCREENING")
    logger.info("Finding NEW materials not in training set")
    logger.info("="*70)
    
    # Configuration
    structures_dir = 'structures/superconductors'
    csv_file = 'data/superconductors.csv'
    model_path = 'models/demo_best_model.pt'  # or 'models/production_tc_model.pt'
    max_trained = 50  # Number of materials used in training (adjust based on your training)
    max_screen = 500  # Number of NEW materials to screen
    
    # Step 1: Load trained model
    logger.info("\n1. Loading trained model...")
    predictor = SuperconductorTcPredictor()
    
    # Check which model to use
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        logger.warning(f"Model not found at {model_path}")
        logger.info("Available models:")
        for model_file in Path('models').glob('*.pt'):
            logger.info(f"  - {model_file}")
        
        # Try production model
        alt_model = Path('models/production_tc_model.pt')
        if alt_model.exists():
            model_path = str(alt_model)
            logger.info(f"Using {model_path} instead")
        else:
            logger.error("No trained model found! Please run demo_training.py first.")
            return
    
    # Create model
    model = EnhancedCrystalTcGNN(
        num_node_features=20,
        num_material_features=24,
        hidden_dim=64  # Match demo training
    ).to(predictor.device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=predictor.device))
        logger.info(f"✅ Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Please run demo_training.py to train a model first")
        return
    
    # Step 2: Get materials used in training
    logger.info("\n2. Identifying training materials...")
    trained_materials = load_trained_materials(csv_file, max_trained)
    
    # Step 3: Screen NEW materials
    logger.info("\n3. Screening new materials...")
    results = screen_new_materials(
        predictor=predictor,
        model=model,
        structures_dir=structures_dir,
        csv_file=csv_file,
        trained_materials=trained_materials,
        max_screen=max_screen,
        min_tc_threshold=10.0  # Log candidates with Tc > 10K
    )
    
    if not results:
        logger.error("No materials screened successfully!")
        return
    
    # Step 4: Save and analyze results
    logger.info("\n4. Saving results...")
    df = save_results(results, output_dir="results")
    
    # Step 5: Generate detailed analysis for top candidates
    logger.info("\n5. Analyzing top candidates...")
    top_candidates = df.head(10)
    
    logger.info("\n📊 DETAILED ANALYSIS OF TOP CANDIDATES:")
    logger.info("="*70)
    
    for idx, (i, row) in enumerate(top_candidates.iterrows(), 1):
        logger.info(f"\n{idx}. {row['material_id']} - {row['formula']}")
        logger.info(f"   Predicted Tc: {row['predicted_tc']:.2f}K")
        logger.info(f"   Formation Energy: {row['formation_energy_per_atom']:.3f} eV/atom")
        logger.info(f"   Band Gap: {row['band_gap']:.3f} eV")
        logger.info(f"   Density: {row['density']:.2f} g/cm³")
        logger.info(f"   Metallic: {'Yes' if row['is_metal'] else 'No'}")
        logger.info(f"   Space Group: {row['spacegroup']}")
        logger.info(f"   Elements: {row['nelements']}, Sites: {row['nsites']}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ SCREENING COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nResults saved to:")
    logger.info(f"  - results/new_materials_predictions.csv (all {len(df)} materials)")
    logger.info(f"  - results/top_20_new_candidates.csv (top 20)")
    if len(df[df['predicted_tc'] > 50]) > 0:
        logger.info(f"  - results/high_tc_candidates.csv (Tc > 50K)")
    
    logger.info(f"\n🎯 Next steps:")
    logger.info(f"  1. Review top candidates in results/top_20_new_candidates.csv")
    logger.info(f"  2. Validate with DFT calculations")
    logger.info(f"  3. Check experimental databases for known Tc values")
    logger.info(f"  4. Synthesize promising materials!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nScreening interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


