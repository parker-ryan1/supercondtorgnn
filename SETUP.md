# Setup & Installation Guide

This document provides detailed installation instructions for **SuperconductorGNN** using Poetry or pip.

## Quick Summary

- ✅ **pyproject.toml** - Poetry configuration with all dependencies
- ✅ **requirements.txt** - Traditional pip requirements file
- ✅ **install.py** - Custom installation script
- ✅ **test_setup.py** - Validation script to check your setup

---

## Installation Methods

### Method 1: Using Poetry (Recommended)

Poetry provides deterministic, reproducible environments.

```powershell
# Install Poetry (if not already installed)
pip install poetry

# Install all dependencies (including dev dependencies)
cd supercondtorgnn
poetry install

# Activate the virtual environment
poetry shell

# Run the model
poetry run python gnn_model.py
```

**Note**: If Poetry installation fails due to permissions on Windows, use Method 2 or 3.

### Method 2: Using pip with virtual environment (Most reliable)

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install PyTorch separately (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or GPU version (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Run the model
python gnn_model.py
```

### Method 3: Using the custom install script

```powershell
# Run the installation script (handles dependencies in order)
python install.py

# This will:
# 1. Install core packages (numpy, pandas, sklearn, etc.)
# 2. Install PyTorch
# 3. Install remaining packages (torch-geometric, pymatgen, mp-api, etc.)

# Run the model
python gnn_model.py
```

---

## Verify Installation

Run the setup validation script:

```powershell
python test_setup.py
```

This will check:
- ✓ Python version and standard libraries
- ✓ NumPy, Pandas, Matplotlib, scikit-learn availability
- ✓ PyTorch installation and GPU support
- ✓ Project files present

Expected output:
```
SuperconductorGNN - Minimal Functionality Test
======================================================================
[OK] Python version: 3.10.6 ...
[OK] Standard library imports: OK
[OK] NumPy version: 2.0.2
[OK] Pandas version: 2.2.3
[OK] PyTorch version: 2.0.0
[OK] scikit-learn version: 1.3.0
[OK] Matplotlib version: 3.7.0
[OK] gnn_model.py
[OK] data_collector.py
[OK] structure_visualizer.py
[OK] requirements.txt
[OK] pyproject.toml
[OK] README.md

======================================================================
Basic tests passed!
```

---

## Key Dependencies

### Core Scientific Stack
- `numpy >= 1.21.0` - Numerical computing
- `pandas >= 1.3.0` - Data analysis
- `scipy >= 1.6.0` - Scientific algorithms
- `scikit-learn >= 0.24.2` - Machine learning utilities
- `matplotlib >= 3.4.0` - Plotting
- `seaborn >= 0.11.0` - Statistical visualization
- `tqdm >= 4.62.0` - Progress bars

### Deep Learning
- `torch >= 2.0.0` - PyTorch core
- `torch-geometric >= 2.3.0` - Graph neural networks
- `torch-scatter` - GNN scatter operations
- `torch-sparse` - Sparse tensor operations

### Materials Science
- `pymatgen >= 2023.0.0` - Crystal structure utilities
- `mp-api >= 0.30.1` - Materials Project API
- `ase >= 3.22.1` - Atomic Simulation Environment

### Other
- `requests >= 2.26.0` - HTTP library
- `tqdm >= 4.62.0` - Progress bars

---

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'torch'"

**Solution**: Install PyTorch explicitly
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### ❌ "ImportError: No module named 'torch_geometric'"

**Solution**: Install PyTorch Geometric
```powershell
pip install torch-geometric
```

### ❌ Permission denied errors on Windows

**Solution**: Use virtual environment or run PowerShell as Administrator
```powershell
# Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Then install normally
pip install -r requirements.txt
```

### ❌ "pymatgen" installation fails

**Solution**: Install build tools, then retry
```powershell
pip install --upgrade setuptools wheel
pip install pymatgen
```

### ❌ Poetry not found after installation

**Solution**: Use pip method directly (Poetry has Windows permission issues)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Development Setup

For development (includes testing and linting tools):

```powershell
poetry install --with dev

# Or with pip
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy jupyter
```

Then run tests:

```powershell
pytest tests/
black .
flake8 .
mypy .
```

---

## Setting Up Materials Project API

1. Register at: https://next-gen.materialsproject.org/
2. Get your API key from your account
3. Set environment variable:

```powershell
# Temporary (current session only)
$env:MP_API_KEY = "your_api_key_here"

# Permanent (add to PowerShell profile)
Add-Content $PROFILE 'export MP_API_KEY="your_api_key_here"'
```

Or create `.env` file in project root:
```
MP_API_KEY=your_api_key_here
```

---

## Running the Model

Once installed:

```powershell
# Train the GNN model
python gnn_model.py

# Collect data from Materials Project
python data_collector.py

# Visualize crystal structures
python structure_visualizer.py
```

---

## Next Steps

- Check the main [README.md](README.md) for usage and architecture details
- Review [large_scale_results.json](large_scale_results.json) for training benchmarks
- See [references.bib](references.bib) for citations
