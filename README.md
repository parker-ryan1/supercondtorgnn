# SuperconductorGNN

Advanced Graph Neural Network (GNN) models for predicting superconductor critical temperature (Tc) using crystal structure and material properties from the Materials Project database.

This repository contains machine learning pipelines for discovering and characterizing superconducting materials using graph neural networks, data collection utilities, structure visualization tools, and comprehensive analysis workflows.

---

## Overview

Superconductivity is a quantum phenomenon with tremendous technological potential. Finding new superconducting materials requires understanding complex relationships between crystal structure, elemental composition, and electronic properties. This project uses **Graph Neural Networks (GNNs)** to learn these relationships directly from crystallographic data.

### Key features

- **CrystalTcGNN**: Graph Convolutional Network (GCN) architecture for Tc prediction
- Multi-modal learning: Combines crystal graph features with material property descriptors
- Data collection from Materials Project: Automated fetching of Ti-based compounds and known superconductors
- Structure visualization: Tools to visualize and analyze crystal structures
- Large-scale training: Scalable pipeline tested on 10,000+ materials
- Performance metrics: MAE, RMSE, R² across train/val/test splits

---

## Quick start

### Requirements

- Python 3.7+ (tested with 3.8, 3.9, 3.10)
- PyTorch (CPU or GPU)
- PyTorch Geometric
- PyMatGen (crystal structure manipulation)
- Materials Project API key
- Common packages: numpy, pandas, scikit-learn, matplotlib, seaborn

**For detailed installation instructions, see [SETUP.md](SETUP.md)**

### Quick installation

1. Clone the repository:
   ```powershell
   git clone https://github.com/parker-ryan1/supercondtorgnn
   cd supercondtorgnn
   ```

2. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```powershell
   # Option A: Using pip (recommended for Windows)
   pip install -r requirements.txt
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   
   # Option B: Using Poetry
   pip install poetry
   poetry install
   
   # Option C: Using custom script
   python install.py
   ```

4. Verify installation:
   ```powershell
   python test_setup.py
   ```

5. Set up Materials Project API:
   ```powershell
   $env:MP_API_KEY = "your_api_key_here"
   ```

---

## Usage

### 1. Collect data from Materials Project

```bash
python data_collector.py
```

This script fetches:
- Ti-based compounds
- Known superconducting materials
- Crystal structures and material properties

Results are saved to JSON files for model training.

### 2. Train the model

```bash
python gnn_model.py
```

The training pipeline will:
- Load or generate training data
- Initialize the CrystalTcGNN model
- Train on GPU (if available) or CPU
- Evaluate on validation and test sets
- Save model checkpoints and results

Key outputs:
- Model weights (`.pth` file)
- Training history and metrics (JSON)
- Performance plots

### 3. Visualize structures

```bash
python structure_visualizer.py
```

Generate crystal structure visualizations:
- 3D atomic arrangements
- Unit cell geometry
- Bonding patterns
- Symmetry information

---

## Project structure

```
supercondtorgnn/
├── gnn_model.py              # Main GNN architecture and training pipeline
├── data_collector.py         # Materials Project API integration
├── structure_visualizer.py   # Crystal structure visualization
├── install.py                # Quick dependency installation script
├── test_setup.py             # Setup validation script
├── requirements.txt          # pip dependencies
├── pyproject.toml            # Poetry configuration
├── SETUP.md                  # Detailed setup and installation guide
├── references.bib            # Bibliography
├── large_scale_results.json  # Sample results from 10k+ training
└── README.md                 # This file
```

### Key files

**`gnn_model.py`** - Core model implementation
- `CrystalTcGNN`: Graph neural network using:
  - Graph Convolutional Network (GCN) layers
  - Global mean pooling for graph-level features
  - Multi-layer perceptron for prediction
  - Dropout regularization
- Handles batched crystal structures via PyTorch Geometric
- Material property integration (21 features per material)
- Training loop with validation and early stopping

**`data_collector.py`** - Data pipeline
- `SuperconductorDataCollector` class with Materials Project integration
- Fetches Ti-based compounds, metallic materials, known superconductors
- Extracts structure, formation energy, band gap, density, symmetry
- JSON/CSV export for training

**`structure_visualizer.py`** - Visualization utilities
- Plot 3D crystal structures
- Analyze bonding and coordination
- Generate symmetry reports
- Export images (PNG, PDF)

---

## Architecture details

### CrystalTcGNN

```
Crystal Structure (Graph)
    ↓
Node Features (atomic properties)
    ↓
GCN Layers (3 layers, 64 hidden dim)
    ├── GCNConv + ReLU + Dropout
    ├── GCNConv + ReLU + Dropout
    └── GCNConv + ReLU
    ↓
Global Mean Pooling
    ↓
Material Properties (21 features)
    ↓
Concatenate [Graph Features + Material Props]
    ↓
Dense Layers (FC1 → FC2 → FC3)
    ├── FC1: [85 → 64]
    ├── FC2: [64 → 32]
    └── FC3: [32 → 1] (Tc prediction)
    ↓
Critical Temperature (Tc) output
```

### Input features

- **Node features** (per atom): Atomic number, electronegativity, radius, etc.
- **Material features** (per structure): 
  - Formation energy
  - Band gap
  - Density
  - Symmetry information
  - Elemental composition descriptors
  - And 16 additional computed properties

---

## Training results

Sample results from large-scale training (10,000 materials):

- **Test R²**: 0.872
- **Test MAE**: 8.58 K
- **Test RMSE**: 19.0 K
- **Training convergence**: ~60 epochs
- **Enhanced coverage**: 1.14% (novel predictions)

See `large_scale_results.json` for full training history.

---

## Reproducibility

- Dataset splits use a fixed random seed (42)
- Model architecture and hyperparameters are hardcoded in scripts (can be refactored to use configs)
- Requires exact dependency versions specified in `requirements.txt`
- GPU results may vary slightly due to floating-point precision

---

## References

Bibliographic information and citations are available in `references.bib`. This work builds on:
- PyTorch and PyTorch Geometric frameworks
- PyMatGen for crystal structure utilities
- Materials Project database and API

---

## Contributing

Contributions welcome! Suggested improvements:
- Add attention mechanisms or transformer layers
- Integrate experimental Tc values for validation
- Develop uncertainty quantification
- Add hyperparameter tuning (grid search, Bayesian optimization)
- Create web interface for predictions

Workflow:
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Open a pull request

---

## License

No license currently specified. Add a LICENSE file (MIT/Apache/GPL) if you want to enable reuse.

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Check existing documentation and examples
- Refer to Materials Project API docs: https://docs.materialsproject.org/

---

## Future work

- [ ] Integrate experimental superconductor database
- [ ] Uncertainty quantification (Bayesian layers)
- [ ] Active learning for targeted material discovery
- [ ] Explainability analysis (attention weights, feature importance)
- [ ] Deployment as a web service (Flask/FastAPI)
- [ ] Pre-trained model checkpoints for transfer learning
