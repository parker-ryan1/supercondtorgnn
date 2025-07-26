# Superconductor Analysis and Prediction

This project focuses on analyzing and predicting superconducting materials using Graph Neural Networks (GNNs), with a special emphasis on titanium-based compounds.

## Project Structure

```
superconductor/
├── data/               # Raw and processed data
├── models/            # Trained models and model definitions
├── scripts/           # Data processing and training scripts
├── structures/        # Generated molecular structures
└── visualization/     # Visualization outputs
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Components

1. **Data Collection**: Scripts to fetch superconductor data from public databases
2. **Structure Generation**: PyMOL-based structure visualization
3. **GNN Model**: Graph Neural Network for superconductivity prediction
4. **Analysis Tools**: Utilities for data analysis and visualization

## Usage

Detailed usage instructions will be added as components are developed.

## Data Sources

- Materials Project Database
- SuperCon Database
- ICSD (Inorganic Crystal Structure Database) public data 