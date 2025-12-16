#!/usr/bin/env python3
"""
Fetch enhanced data for ALL materials in batches with progress tracking.
Handles interruptions gracefully and can resume from where it left off.
"""

import pandas as pd
import json
from pathlib import Path
import logging
from fetch_detailed_data import fetch_enhanced_properties, save_enhanced_data
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def fetch_all_materials_in_batches(
    batch_size=100,
    start_from=0,
    max_materials=None,
    checkpoint_file='data/fetch_checkpoint.json'
):
    """
    Fetch all materials in batches with checkpointing for resumption.
    
    Parameters:
    - batch_size: Number of materials to fetch per batch
    - start_from: Starting index (for resuming)
    - max_materials: Maximum number to fetch (None = all)
    - checkpoint_file: File to save progress
    """
    
    logger.info("="*70)
    logger.info("FETCH ALL ENHANCED MATERIAL DATA - BATCH MODE")
    logger.info("="*70)
    
    # Load existing material IDs
    csv_file = Path('data/superconductors.csv')
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    all_material_ids = df['material_id'].tolist()
    
    # Apply limits
    if max_materials:
        all_material_ids = all_material_ids[:max_materials]
    
    total_materials = len(all_material_ids)
    logger.info(f"Total materials to process: {total_materials}")
    
    # Check for existing checkpoint
    checkpoint_path = Path(checkpoint_file)
    if checkpoint_path.exists() and start_from == 0:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            start_from = checkpoint.get('last_completed_index', 0) + 1
            logger.info(f"📌 Resuming from checkpoint: index {start_from}")
    
    # Load existing data if any
    output_file = Path('data/enhanced_superconductors_full.json')
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            logger.info(f"📂 Loaded {len(existing_data)} existing records")
    else:
        existing_data = []
    
    # Process in batches
    current_idx = start_from
    all_enhanced_data = existing_data
    
    while current_idx < total_materials:
        batch_end = min(current_idx + batch_size, total_materials)
        batch_ids = all_material_ids[current_idx:batch_end]
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH {current_idx//batch_size + 1}: Processing materials {current_idx+1} to {batch_end}")
        logger.info(f"Progress: {batch_end}/{total_materials} ({batch_end/total_materials*100:.1f}%)")
        logger.info(f"{'='*70}\n")
        
        # Fetch batch
        try:
            batch_data = fetch_enhanced_properties(
                batch_ids,
                max_materials=len(batch_ids)
            )
            
            all_enhanced_data.extend(batch_data)
            
            # Save checkpoint
            checkpoint = {
                'last_completed_index': batch_end - 1,
                'total_processed': len(all_enhanced_data),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            # Save intermediate results
            with open(output_file, 'w') as f:
                json.dump(all_enhanced_data, f, indent=2, default=str)
            
            logger.info(f"✅ Batch complete. Checkpoint saved.")
            logger.info(f"💾 Intermediate results saved to {output_file}")
            
            current_idx = batch_end
            
            # Brief pause between batches to be nice to API
            if current_idx < total_materials:
                logger.info("⏸️ Pausing 5 seconds before next batch...")
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Process interrupted by user!")
            logger.info(f"Progress saved. You can resume from index {current_idx}")
            logger.info(f"Run again to continue from where you left off.")
            return
        except Exception as e:
            logger.error(f"❌ Error in batch: {e}")
            logger.info(f"Progress saved up to index {current_idx - 1}")
            logger.info(f"You can resume from index {current_idx}")
            raise
    
    # Final save
    logger.info("\n" + "="*70)
    logger.info("ALL BATCHES COMPLETE!")
    logger.info("="*70)
    
    # Save final results
    df_final = pd.DataFrame(all_enhanced_data)
    df_final.to_csv('data/enhanced_superconductors_full.csv', index=False)
    
    with open('data/enhanced_superconductors_full.json', 'w') as f:
        json.dump(all_enhanced_data, f, indent=2, default=str)
    
    logger.info(f"\n✅ Final results saved:")
    logger.info(f"  - data/enhanced_superconductors_full.csv")
    logger.info(f"  - data/enhanced_superconductors_full.json")
    
    logger.info(f"\n📊 FINAL STATISTICS:")
    logger.info(f"  Total materials processed: {len(all_enhanced_data)}")
    successful = sum(1 for d in all_enhanced_data if d.get('success', False))
    logger.info(f"  Successful: {successful} ({successful/len(all_enhanced_data)*100:.1f}%)")
    logger.info(f"  Failed: {len(all_enhanced_data) - successful}")
    
    # Clean up checkpoint file
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"\n🧹 Checkpoint file removed (processing complete)")
    
    logger.info("\n🎉 ALL DONE!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fetch enhanced data for all materials in batches'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of materials per batch (default: 100)'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Starting index (for manual resume, default: 0 = auto-resume from checkpoint)'
    )
    parser.add_argument(
        '--max-materials',
        type=int,
        default=None,
        help='Maximum number of materials to fetch (default: all)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: only process first 200 materials'
    )
    
    args = parser.parse_args()
    
    # Test mode
    if args.test:
        logger.info("🧪 TEST MODE: Processing first 200 materials only")
        args.max_materials = 200
    
    # Estimate time
    if args.max_materials:
        estimated_time_min = (args.max_materials * 1.8) / 60
        logger.info(f"\n⏱️ Estimated time: {estimated_time_min:.1f} minutes (~{estimated_time_min/60:.1f} hours)")
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Start from: {args.start_from}")
    logger.info(f"  Max materials: {args.max_materials or 'ALL'}")
    
    # Confirm before starting large job
    if not args.test and not args.max_materials:
        response = input("\n⚠️ This will fetch ALL materials (~10,000+) and may take several hours. Continue? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Cancelled by user.")
            return
    
    # Run
    try:
        fetch_all_materials_in_batches(
            batch_size=args.batch_size,
            start_from=args.start_from,
            max_materials=args.max_materials
        )
    except KeyboardInterrupt:
        logger.info("\n\nGracefully stopped. Run again to resume.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


