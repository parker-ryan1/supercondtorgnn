#!/usr/bin/env python3
"""
Large-scale enhanced data fetching with robust checkpointing.
Designed to fetch data for 10,000+ materials efficiently.
"""

import pandas as pd
from pymatgen.ext.matproj import MPRester
from pathlib import Path
import json
import logging
import time
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_api_key():
    """Get Materials Project API key"""
    api_key = os.getenv('MP_API_KEY')
    if not api_key:
        api_key = "jQ2P1lhV1Tla8lHmCJlINigb1j3B2Q8F"  # Default key
    return api_key

def fetch_batch(material_ids, mpr, batch_num, total_batches):
    """Fetch data for a batch of materials"""
    results = []
    
    for mat_id in tqdm(material_ids, desc=f"Batch {batch_num}/{total_batches}"):
        try:
            # Quick fetch with essential properties only
            data = {
                'material_id': mat_id,
                'success': True
            }
            
            # Get summary data (fast)
            try:
                summary = mpr.get_summary(mat_id)
                data['formula'] = str(summary.formula_pretty)
                data['formation_energy_per_atom'] = summary.formation_energy_per_atom
                data['energy_above_hull'] = summary.energy_above_hull
                data['band_gap'] = summary.band_gap
                data['is_stable'] = summary.is_stable
                data['is_metal'] = summary.is_metal
                data['density'] = summary.density
                data['volume'] = summary.structure.volume
                data['nsites'] = summary.structure.num_sites
                data['nelements'] = len(summary.structure.composition.elements)
                
                # Space group
                try:
                    data['spacegroup_symbol'] = summary.symmetry.symbol
                    data['spacegroup_number'] = summary.symmetry.number
                    data['crystal_system'] = summary.symmetry.crystal_system
                except:
                    pass
                
            except Exception as e:
                logger.debug(f"Summary failed for {mat_id}: {e}")
                data['success'] = False
                data['error'] = str(e)
            
            # Electronic structure (if available)
            try:
                elec = mpr.get_electronic_structure(mat_id)
                if elec:
                    data['efermi'] = elec.efermi if hasattr(elec, 'efermi') else None
                    data['is_gap_direct'] = elec.is_gap_direct if hasattr(elec, 'is_gap_direct') else False
                    data['is_magnetic'] = elec.is_magnetic if hasattr(elec, 'is_magnetic') else False
            except:
                pass
            
            results.append(data)
            
        except Exception as e:
            logger.warning(f"Failed to fetch {mat_id}: {e}")
            results.append({
                'material_id': mat_id,
                'success': False,
                'error': str(e)
            })
    
    return results

def main():
    logger.info("="*70)
    logger.info("LARGE-SCALE ENHANCED DATA FETCHING")
    logger.info("="*70)
    
    # Configuration
    TARGET_MATERIALS = 10000  # Target 10k materials
    BATCH_SIZE = 100  # Fetch 100 at a time
    CHECKPOINT_FILE = 'data/fetch_checkpoint_large.json'
    OUTPUT_CSV = 'data/enhanced_superconductors_large.csv'
    OUTPUT_JSON = 'data/enhanced_superconductors_large.json'
    
    # Get material IDs from structures
    structure_files = list(Path('structures/superconductors').glob('*.cif'))
    all_material_ids = [f.stem for f in structure_files]
    
    logger.info(f"\n📊 Dataset Information:")
    logger.info(f"  Total structures available: {len(all_material_ids):,}")
    logger.info(f"  Target to fetch: {TARGET_MATERIALS:,}")
    
    # Limit to target
    material_ids = all_material_ids[:TARGET_MATERIALS]
    logger.info(f"  Will fetch: {len(material_ids):,}")
    
    # Load checkpoint if exists
    fetched_ids = set()
    all_results = []
    
    if Path(CHECKPOINT_FILE).exists():
        logger.info(f"\n✅ Found checkpoint file")
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            all_results = checkpoint.get('results', [])
            fetched_ids = set(checkpoint.get('fetched_ids', []))
        logger.info(f"  Already fetched: {len(fetched_ids):,} materials")
    
    # Filter out already fetched
    remaining_ids = [mid for mid in material_ids if mid not in fetched_ids]
    
    if not remaining_ids:
        logger.info("\n✅ All materials already fetched!")
        logger.info(f"  Total: {len(all_results):,} materials")
    else:
        logger.info(f"\n🔄 Remaining to fetch: {len(remaining_ids):,}")
        
        # Get API key
        api_key = get_api_key()
        
        # Batch processing
        num_batches = (len(remaining_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"\n🚀 Starting batch processing:")
        logger.info(f"  Batches: {num_batches}")
        logger.info(f"  Batch size: {BATCH_SIZE}")
        logger.info(f"  Estimated time: ~{num_batches * 2} minutes")
        
        with MPRester(api_key) as mpr:
            for i in range(0, len(remaining_ids), BATCH_SIZE):
                batch_ids = remaining_ids[i:i+BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                
                logger.info(f"\n📦 Processing batch {batch_num}/{num_batches}")
                
                # Fetch batch
                batch_results = fetch_batch(batch_ids, mpr, batch_num, num_batches)
                all_results.extend(batch_results)
                fetched_ids.update(batch_ids)
                
                # Save checkpoint every batch
                checkpoint = {
                    'fetched_ids': list(fetched_ids),
                    'results': all_results,
                    'timestamp': time.time()
                }
                
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(checkpoint, f)
                
                logger.info(f"✅ Checkpoint saved ({len(fetched_ids):,}/{len(material_ids):,})")
                
                # Rate limiting (be nice to Materials Project)
                if batch_num < num_batches:
                    time.sleep(1)  # 1 second between batches
    
    # Save final results
    logger.info(f"\n💾 Saving final results...")
    
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"✅ Saved: {OUTPUT_CSV}")
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"✅ Saved: {OUTPUT_JSON}")
    
    # Statistics
    logger.info(f"\n" + "="*70)
    logger.info("FINAL STATISTICS")
    logger.info("="*70)
    
    successful = df[df['success'] == True]
    logger.info(f"  Total processed: {len(df):,}")
    logger.info(f"  Successful: {len(successful):,} ({len(successful)/len(df)*100:.1f}%)")
    logger.info(f"  Failed: {len(df) - len(successful):,}")
    
    # Feature coverage
    logger.info(f"\n📊 Feature Coverage:")
    for col in ['efermi', 'band_gap', 'formation_energy_per_atom', 'density']:
        if col in df.columns:
            coverage = df[col].notna().sum()
            logger.info(f"  {col:30s}: {coverage:6,}/{len(df):,} ({coverage/len(df)*100:5.1f}%)")
    
    # Clean up checkpoint
    if Path(CHECKPOINT_FILE).exists():
        os.remove(CHECKPOINT_FILE)
        logger.info(f"\n🧹 Checkpoint file removed")
    
    logger.info(f"\n" + "="*70)
    logger.info("✅ LARGE-SCALE FETCH COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nFetched {len(df):,} materials")
    logger.info(f"Ready for training!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Interrupted by user")
        logger.info("Progress saved in checkpoint file")
        logger.info("Run again to resume")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


