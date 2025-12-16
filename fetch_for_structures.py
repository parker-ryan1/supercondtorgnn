#!/usr/bin/env python3
"""
Fetch enhanced data specifically for the materials we have structures for.
This ensures maximum coverage of enhanced features.
"""

import pandas as pd
from pathlib import Path
import logging
from fetch_detailed_data import fetch_enhanced_properties, save_enhanced_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("FETCH ENHANCED DATA FOR AVAILABLE STRUCTURES")
    logger.info("="*70)
    
    # Get list of available structures
    structures_dir = Path('structures/superconductors')
    structure_files = list(structures_dir.glob("*.cif"))
    
    logger.info(f"Found {len(structure_files)} structure files")
    
    # Extract material IDs
    material_ids = [f.stem for f in structure_files]
    
    logger.info(f"Will fetch enhanced data for {len(material_ids)} materials")
    
    # Fetch in batches (100 at a time for safety)
    batch_size = 100
    max_total = 500  # Limit for reasonable time
    
    material_ids = material_ids[:max_total]
    
    logger.info(f"Limited to first {len(material_ids)} materials")
    logger.info(f"Estimated time: {len(material_ids) * 1.8 / 60:.1f} minutes")
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() != 'yes':
        logger.info("Cancelled")
        return
    
    # Fetch data
    enhanced_data = fetch_enhanced_properties(material_ids, max_materials=len(material_ids))
    
    # Save
    df = save_enhanced_data(enhanced_data, output_dir='data')
    
    # Also save as "for_structures" to distinguish
    df.to_csv('data/enhanced_for_structures.csv', index=False)
    logger.info(f"Also saved to: data/enhanced_for_structures.csv")
    
    logger.info("\nDONE! Now you can run training with much better coverage of enhanced features.")

if __name__ == "__main__":
    main()


