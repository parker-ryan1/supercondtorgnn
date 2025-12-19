#!/usr/bin/env python3
"""
Quick installation script for supercondtorgnn.
Install dependencies without Poetry (for systems with permission issues).
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install all required dependencies."""
    
    print("=" * 70)
    print("SuperconductorGNN - Dependency Installation")
    print("=" * 70)
    
    # Core dependencies
    core_deps = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "requests>=2.26.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ]
    
    # Heavy packages (install separately)
    heavy_deps = [
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "pymatgen>=2023.0.0",
        "mp-api>=0.30.1",
        "ase>=3.22.1",
    ]
    
    print("\n📦 Installing core dependencies...")
    for dep in core_deps:
        print(f"  • {dep}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    print("\n⚙️  Installing heavy ML packages (this may take a while)...")
    
    # Install PyTorch first
    print("\n  • PyTorch (CPU)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch>=2.0.0",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    
    # Install remaining packages
    remaining = [
        "torch-geometric>=2.3.0",
        "pymatgen>=2023.0.0",
        "mp-api>=0.30.1",
        "ase>=3.22.1",
    ]
    
    for dep in remaining:
        print(f"  • {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError as e:
            print(f"    ⚠️  Warning: Could not install {dep}: {e}")
            print(f"    This package may require additional setup or dependencies.")
    
    print("\n✅ Installation complete!")
    print("\nYou can now run:")
    print("  python gnn_model.py")
    print("  python data_collector.py")
    print("  python structure_visualizer.py")

if __name__ == "__main__":
    try:
        install_dependencies()
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)
