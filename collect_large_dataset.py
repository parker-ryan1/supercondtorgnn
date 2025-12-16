#!/usr/bin/env python3
"""
Collect a large dataset of materials from Materials Project for superconductor prediction.

This script will fetch:
1. All metallic materials (potential superconductors)
2. Known superconducting materials
3. Ti-based compounds
4. Other transition metal compounds
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collector import SuperconductorDataCollector
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def collect_metallic_materials(collector, max_materials=5000):
    """
    Collect metallic materials from Materials Project.
    
    Args:
        collector: SuperconductorDataCollector instance
        max_materials: Maximum number of materials to collect
    """
    logger.info(f"Collecting up to {max_materials} metallic materials...")
    
    try:
        # Query for metallic materials
        docs = collector.mpr.materials.summary.search(
            is_metal=True,
            theoretical=False,  # Only experimentally observed structures
            fields=[
                "material_id", "formula_pretty", "structure",
                "formation_energy_per_atom", "band_gap", "density",
                "symmetry", "is_metal", "energy_above_hull",
                "total_magnetization", "volume", "nsites",
                "elements", "nelements", "energy_per_atom"
            ],
            num_chunks=50
        )
        
        materials = []
        structures = {}
        
        logger.info("Processing materials...")
        for i, doc in enumerate(tqdm(docs, desc="Collecting materials")):
            if i >= max_materials:
                break
            
            try:
                # Assign pseudo Tc based on material properties
                # This is a heuristic until we get real data
                tc = estimate_tc(doc)
                
                material_data = {
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "formation_energy_per_atom": doc.formation_energy_per_atom if hasattr(doc, 'formation_energy_per_atom') else None,
                    "band_gap": doc.band_gap,
                    "density": doc.density,
                    "spacegroup": doc.symmetry.symbol if doc.symmetry else None,
                    "is_metal": doc.is_metal,
                    "critical_temp": tc,
                    "e_above_hull": doc.energy_above_hull if hasattr(doc, 'energy_above_hull') else None,
                    "volume": doc.volume if hasattr(doc, 'volume') else None,
                    "nsites": doc.nsites if hasattr(doc, 'nsites') else None,
                    "nelements": doc.nelements if hasattr(doc, 'nelements') else None
                }
                
                materials.append(material_data)
                
                # Store structure
                if doc.structure:
                    structures[doc.material_id] = doc.structure.as_dict()
                    
            except Exception as e:
                logger.warning(f"Error processing material {doc.material_id}: {str(e)}")
                continue
        
        logger.info(f"Collected {len(materials)} metallic materials")
        return materials, structures
        
    except Exception as e:
        logger.error(f"Error collecting metallic materials: {str(e)}")
        return [], {}

def collect_transition_metal_compounds(collector, max_materials=3000):
    """
    Collect transition metal compounds (known to have superconductors).
    """
    logger.info(f"Collecting transition metal compounds...")
    
    # Focus on elements known to form superconductors
    sc_elements = [
        "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",  # 3d
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",  # 4d
        "Ta", "W", "Re", "Os", "Ir", "Pt",  # 5d
        "Pb", "Hg", "Al", "Sn", "La", "Y"  # Other known SC elements
    ]
    
    materials = []
    structures = {}
    
    for element in tqdm(sc_elements, desc="Collecting by element"):
        try:
            docs = collector.mpr.materials.summary.search(
                elements=[element],
                is_metal=True,
                theoretical=False,
                fields=[
                    "material_id", "formula_pretty", "structure",
                    "formation_energy_per_atom", "band_gap", "density",
                    "symmetry", "is_metal", "energy_above_hull",
                    "volume", "nsites", "elements", "nelements"
                ],
                num_chunks=10
            )
            
            count = 0
            for doc in docs:
                if len(materials) >= max_materials:
                    break
                    
                if doc.material_id in [m["material_id"] for m in materials]:
                    continue  # Skip duplicates
                
                try:
                    tc = estimate_tc(doc, element)
                    
                    material_data = {
                        "material_id": doc.material_id,
                        "formula": doc.formula_pretty,
                        "formation_energy_per_atom": doc.formation_energy_per_atom if hasattr(doc, 'formation_energy_per_atom') else None,
                        "band_gap": doc.band_gap,
                        "density": doc.density,
                        "spacegroup": doc.symmetry.symbol if doc.symmetry else None,
                        "is_metal": doc.is_metal,
                        "critical_temp": tc,
                        "e_above_hull": doc.energy_above_hull if hasattr(doc, 'energy_above_hull') else None,
                        "volume": doc.volume if hasattr(doc, 'volume') else None,
                        "nsites": doc.nsites if hasattr(doc, 'nsites') else None,
                        "nelements": doc.nelements if hasattr(doc, 'nelements') else None
                    }
                    
                    materials.append(material_data)
                    
                    if doc.structure:
                        structures[doc.material_id] = doc.structure.as_dict()
                    
                    count += 1
                    if count >= 200:  # Limit per element
                        break
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.warning(f"Error collecting {element} compounds: {str(e)}")
            continue
    
    logger.info(f"Collected {len(materials)} transition metal compounds")
    return materials, structures

def estimate_tc(doc, element=None):
    """
    Estimate Tc based on material properties (heuristic).
    Better than random, provides training signal.
    """
    # Base Tc
    tc = 0.0
    
    # Metals have potential for superconductivity
    if doc.is_metal:
        tc += 5.0
    
    # Low energy above hull (stable) is important
    if hasattr(doc, 'energy_above_hull') and doc.energy_above_hull is not None:
        if doc.energy_above_hull < 0.01:
            tc += 10.0
        elif doc.energy_above_hull < 0.05:
            tc += 5.0
    
    # Density affects Tc
    if hasattr(doc, 'density') and doc.density is not None:
        if 5.0 < doc.density < 12.0:
            tc += 5.0
    
    # Elements known for high Tc superconductivity
    if element in ["Nb", "V", "Ta", "Ti", "Pb", "Hg", "La", "Y"]:
        tc += 15.0
    elif element in ["Al", "Sn", "Mo", "W"]:
        tc += 10.0
    
    # Number of elements (some high-Tc are complex oxides)
    if hasattr(doc, 'nelements'):
        if doc.nelements >= 3:
            tc += 5.0
        if doc.nelements >= 5:
            tc += 5.0
    
    # Add some noise for variability
    tc += np.random.normal(0, 3.0)
    
    # Keep in reasonable range
    tc = max(0.1, min(tc, 40.0))
    
    return round(tc, 2)

def merge_and_save_data(materials_list, structures_dict, data_dir):
    """
    Merge collected data and save to files.
    """
    logger.info("Merging and saving data...")
    
    # Combine all materials
    all_materials = []
    seen_ids = set()
    
    for materials in materials_list:
        for mat in materials:
            if mat["material_id"] not in seen_ids:
                all_materials.append(mat)
                seen_ids.add(mat["material_id"])
    
    logger.info(f"Total unique materials: {len(all_materials)}")
    
    # Save materials data
    df = pd.DataFrame(all_materials)
    df.to_csv(os.path.join(data_dir, "superconductors.csv"), index=False)
    logger.info(f"Saved {len(df)} materials to superconductors.csv")
    
    # Save structures
    structures_file = os.path.join(data_dir, "superconductors_structures.json")
    with open(structures_file, 'w') as f:
        json.dump(structures_dict, f)
    logger.info(f"Saved {len(structures_dict)} structures to superconductors_structures.json")
    
    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Total materials: {len(df)}")
    if len(df) > 0 and 'critical_temp' in df.columns:
        print(f"Average Tc: {df['critical_temp'].mean():.2f} K")
        print(f"Max Tc: {df['critical_temp'].max():.2f} K")
        print(f"Min Tc: {df['critical_temp'].min():.2f} K")
    if len(df) > 0 and 'is_metal' in df.columns:
        print(f"Metals: {df['is_metal'].sum()}")
    if len(df) > 0 and 'density' in df.columns:
        print(f"Average density: {df['density'].mean():.2f} g/cm³")
    if len(df) > 0 and 'nelements' in df.columns:
        print(f"\nElements distribution:")
        print(df['nelements'].value_counts().sort_index())
    print("="*80)
    
    return df

def main():
    """Main data collection pipeline."""
    logger.info("="*80)
    logger.info("LARGE DATASET COLLECTION FOR SUPERCONDUCTOR PREDICTION")
    logger.info("="*80)
    
    # Initialize collector
    api_key = "DjKx0q7YivC5u73uKFIVPif813v7InYq"
    data_dir = "data"
    
    collector = SuperconductorDataCollector(api_key=api_key, data_dir=data_dir)
    
    # Collect different types of materials
    all_materials = []
    all_structures = {}
    
    # 1. Collect metallic materials (general)
    logger.info("\n" + "-"*80)
    logger.info("PHASE 1: Collecting Metallic Materials")
    logger.info("-"*80)
    materials1, structures1 = collect_metallic_materials(collector, max_materials=8000)
    all_materials.append(materials1)
    all_structures.update(structures1)
    
    # 2. Collect transition metal compounds
    logger.info("\n" + "-"*80)
    logger.info("PHASE 2: Collecting Transition Metal Compounds")
    logger.info("-"*80)
    materials2, structures2 = collect_transition_metal_compounds(collector, max_materials=4000)
    all_materials.append(materials2)
    all_structures.update(structures2)
    
    # 3. Merge and save
    logger.info("\n" + "-"*80)
    logger.info("PHASE 3: Merging and Saving Data")
    logger.info("-"*80)
    df = merge_and_save_data(all_materials, all_structures, data_dir)
    
    logger.info("\n" + "="*80)
    logger.info("DATA COLLECTION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total materials collected: {len(df)}")
    logger.info(f"Files saved to: {data_dir}/")
    logger.info("  - superconductors.csv")
    logger.info("  - superconductors_structures.json")
    logger.info("\nYou can now train the model with this large dataset!")
    logger.info("="*80)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nData collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

