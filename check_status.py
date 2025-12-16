import pandas as pd
from pathlib import Path

# Check enhanced data status
enhanced_path = Path('data/enhanced_superconductors.csv')
full_path = Path('data/enhanced_superconductors_full.csv')
checkpoint_path = Path('data/fetch_checkpoint.json')

print("="*70)
print("DATA STATUS CHECK")
print("="*70)

if enhanced_path.exists():
    df = pd.read_csv(enhanced_path)
    print(f"[OK] Enhanced data (initial): {len(df)} materials")
else:
    print("[!] No enhanced data found")

if full_path.exists():
    df_full = pd.read_csv(full_path)
    print(f"[OK] Full enhanced dataset: {len(df_full)} materials")
else:
    print("[...] Full dataset not yet available (still fetching or not started)")

if checkpoint_path.exists():
    import json
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    print(f"[i] Checkpoint found: {checkpoint.get('total_processed', 0)} materials processed")
    print(f"    Last index: {checkpoint.get('last_completed_index', 0)}")
else:
    print("    No checkpoint found")

print("\nRecommendation:")
if full_path.exists():
    df_full = pd.read_csv(full_path)
    if len(df_full) >= 100:
        print(f"[OK] You have {len(df_full)} materials - READY TO TRAIN!")
    else:
        print(f"[...] Only {len(df_full)} materials - recommend waiting for more")
elif enhanced_path.exists():
    df = pd.read_csv(enhanced_path)
    print(f"[OK] You have {len(df)} materials (initial batch) - Can start training now!")
else:
    print("[!] No enhanced data available yet")

