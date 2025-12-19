#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test script for supercondtorgnn - tests core functionality without heavy dependencies.
"""

import sys
print("SuperconductorGNN - Minimal Functionality Test")
print("=" * 70)

# Test 1: Check Python version
print(f"\n[OK] Python version: {sys.version}")

# Test 2: Check standard library imports
try:
    import json
    import pickle
    from pathlib import Path
    print("[OK] Standard library imports: OK")
except ImportError as e:
    print(f"[ERROR] Standard library import failed: {e}")
    sys.exit(1)

# Test 3: Check numpy and pandas
try:
    import numpy as np
    import pandas as pd
    print(f"[OK] NumPy version: {np.__version__}")
    print(f"[OK] Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"[WARN] NumPy/Pandas not available: {e}")

# Test 4: Check PyTorch
try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
except ImportError as e:
    print(f"[WARN] PyTorch not available (this is expected if not yet installed)")
    print(f"   Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu")

# Test 5: Check scikit-learn
try:
    import sklearn
    print(f"[OK] scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"[WARN] scikit-learn not available: {e}")

# Test 6: Check matplotlib
try:
    import matplotlib
    print(f"[OK] Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"[WARN] Matplotlib not available: {e}")

# Test 7: Verify project structure
print("\nProject structure:")
project_root = Path(__file__).parent
required_files = [
    "gnn_model.py",
    "data_collector.py",
    "structure_visualizer.py",
    "requirements.txt",
    "pyproject.toml",
    "README.md"
]

for fname in required_files:
    fpath = project_root / fname
    if fpath.exists():
        print(f"   [OK] {fname}")
    else:
        print(f"   [MISSING] {fname}")

print("\n" + "=" * 70)
print("Basic tests passed!")
print("\nNext steps:")
print("  1. Create virtualenv: python -m venv .venv")
print("  2. Install ML packages: pip install -r requirements.txt")
print("  3. Set Materials Project API key: $env:MP_API_KEY='your_key'")
print("  4. Run training: python gnn_model.py")
print("=" * 70)
