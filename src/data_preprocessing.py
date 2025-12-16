"""
Data preprocessing module for superconductor analysis.

This module handles data preprocessing tasks such as:
- Loading and cleaning data
- Feature extraction and engineering
- Data normalization and standardization
- Graph construction for GNN models
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import config

logger = logging.getLogger(__name__)

class StructureFeatureExtractor:
    """Extract features from crystal structures for machine learning."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        # Crystal nearest neighbor analyzer for bond detection
        self.crystal_nn = CrystalNN(weighted_cn=True, distance_cutoffs=(0.5, 1.5))
        
        # Element properties for node features
        self.element_properties = self._load_element_properties()
    
    def _load_element_properties(self) -> Dict[str, np.ndarray]:
        """
        Load element properties for feature generation.
        
        Returns:
            Dictionary mapping element symbols to property vectors
        """
        # Basic properties: atomic number, atomic mass, electronegativity, atomic radius
        properties = {
            'H': [1, 1.008, 2.20, 0.37],
            'He': [2, 4.003, 0.0, 0.32],
            'Li': [3, 6.941, 0.98, 1.34],
            'Be': [4, 9.012, 1.57, 0.90],
            'B': [5, 10.811, 2.04, 0.82],
            'C': [6, 12.011, 2.55, 0.77],
            'N': [7, 14.007, 3.04, 0.75],
            'O': [8, 15.999, 3.44, 0.73],
            'F': [9, 18.998, 3.98, 0.71],
            'Ne': [10, 20.180, 0.0, 0.69],
            'Na': [11, 22.990, 0.93, 1.54],
            'Mg': [12, 24.305, 1.31, 1.30],
            'Al': [13, 26.982, 1.61, 1.18],
            'Si': [14, 28.086, 1.90, 1.11],
            'P': [15, 30.974, 2.19, 1.06],
            'S': [16, 32.065, 2.58, 1.02],
            'Cl': [17, 35.453, 3.16, 0.99],
            'Ar': [18, 39.948, 0.0, 0.97],
            'K': [19, 39.098, 0.82, 1.96],
            'Ca': [20, 40.078, 1.00, 1.74],
            'Sc': [21, 44.956, 1.36, 1.44],
            'Ti': [22, 47.867, 1.54, 1.36],
            'V': [23, 50.942, 1.63, 1.25],
            'Cr': [24, 51.996, 1.66, 1.27],
            'Mn': [25, 54.938, 1.55, 1.39],
            'Fe': [26, 55.845, 1.83, 1.25],
            'Co': [27, 58.933, 1.88, 1.26],
            'Ni': [28, 58.693, 1.91, 1.21],
            'Cu': [29, 63.546, 1.90, 1.38],
            'Zn': [30, 65.38, 1.65, 1.31],
            'Ga': [31, 69.723, 1.81, 1.26],
            'Ge': [32, 72.64, 2.01, 1.22],
            'As': [33, 74.922, 2.18, 1.19],
            'Se': [34, 78.96, 2.55, 1.16],
            'Br': [35, 79.904, 2.96, 1.14],
            'Kr': [36, 83.798, 0.0, 1.10],
            'Rb': [37, 85.468, 0.82, 2.11],
            'Sr': [38, 87.62, 0.95, 1.92],
            'Y': [39, 88.906, 1.22, 1.62],
            'Zr': [40, 91.224, 1.33, 1.48],
            'Nb': [41, 92.906, 1.6, 1.37],
            'Mo': [42, 95.96, 2.16, 1.45],
            'Tc': [43, 98.0, 1.9, 1.56],
            'Ru': [44, 101.07, 2.2, 1.26],
            'Rh': [45, 102.906, 2.28, 1.35],
            'Pd': [46, 106.42, 2.20, 1.31],
            'Ag': [47, 107.868, 1.93, 1.53],
            'Cd': [48, 112.411, 1.69, 1.48],
            'In': [49, 114.818, 1.78, 1.44],
            'Sn': [50, 118.71, 1.96, 1.41],
            'Sb': [51, 121.76, 2.05, 1.38],
            'Te': [52, 127.6, 2.1, 1.35],
            'I': [53, 126.904, 2.66, 1.33],
            'Xe': [54, 131.293, 0.0, 1.30],
            'Cs': [55, 132.905, 0.79, 2.25],
            'Ba': [56, 137.327, 0.89, 1.98],
            'La': [57, 138.905, 1.1, 1.69],
            'Ce': [58, 140.116, 1.12, 1.65],
            'Pr': [59, 140.908, 1.13, 1.65],
            'Nd': [60, 144.242, 1.14, 1.64],
            'Pm': [61, 145.0, 1.13, 1.63],
            'Sm': [62, 150.36, 1.17, 1.62],
            'Eu': [63, 151.964, 1.2, 1.85],
            'Gd': [64, 157.25, 1.2, 1.61],
            'Tb': [65, 158.925, 1.1, 1.59],
            'Dy': [66, 162.5, 1.22, 1.59],
            'Ho': [67, 164.93, 1.23, 1.58],
            'Er': [68, 167.259, 1.24, 1.57],
            'Tm': [69, 168.934, 1.25, 1.56],
            'Yb': [70, 173.054, 1.1, 1.74],
            'Lu': [71, 174.967, 1.27, 1.56],
            'Hf': [72, 178.49, 1.3, 1.44],
            'Ta': [73, 180.948, 1.5, 1.34],
            'W': [74, 183.84, 2.36, 1.30],
            'Re': [75, 186.207, 1.9, 1.28],
            'Os': [76, 190.23, 2.2, 1.26],
            'Ir': [77, 192.217, 2.2, 1.27],
            'Pt': [78, 195.084, 2.28, 1.30],
            'Au': [79, 196.967, 2.54, 1.34],
            'Hg': [80, 200.59, 2.0, 1.49],
            'Tl': [81, 204.383, 1.62, 1.48],
            'Pb': [82, 207.2, 2.33, 1.47],
            'Bi': [83, 208.98, 2.02, 1.46],
            'Po': [84, 209.0, 2.0, 1.46],
            'At': [85, 210.0, 2.2, 1.45],
            'Rn': [86, 222.0, 0.0, 1.43],
            'Fr': [87, 223.0, 0.7, 2.23],
            'Ra': [88, 226.0, 0.9, 2.01],
            'Ac': [89, 227.0, 1.1, 1.86],
            'Th': [90, 232.038, 1.3, 1.75],
            'Pa': [91, 231.036, 1.5, 1.69],
            'U': [92, 238.029, 1.38, 1.70],
            'Np': [93, 237.0, 1.36, 1.71],
            'Pu': [94, 244.0, 1.28, 1.72],
            'Am': [95, 243.0, 1.3, 1.66],
            'Cm': [96, 247.0, 1.3, 1.66],
            'Bk': [97, 247.0, 1.3, 1.66],
            'Cf': [98, 251.0, 1.3, 1.68],
            'Es': [99, 252.0, 1.3, 1.65],
            'Fm': [100, 257.0, 1.3, 1.67],
            'Md': [101, 258.0, 1.3, 1.73],
            'No': [102, 259.0, 1.3, 1.76],
            'Lr': [103, 262.0, 1.3, 1.61],
            'Rf': [104, 267.0, 1.3, 1.57],
            'Db': [105, 268.0, 1.3, 1.49],
            'Sg': [106, 269.0, 1.3, 1.43],
            'Bh': [107, 270.0, 1.3, 1.41],
            'Hs': [108, 270.0, 1.3, 1.34],
            'Mt': [109, 278.0, 1.3, 1.29],
            'Ds': [110, 281.0, 1.3, 1.28],
            'Rg': [111, 282.0, 1.3, 1.21],
            'Cn': [112, 285.0, 1.3, 1.22],
            'Nh': [113, 286.0, 1.3, 1.36],
            'Fl': [114, 289.0, 1.3, 1.43],
            'Mc': [115, 290.0, 1.3, 1.62],
            'Lv': [116, 293.0, 1.3, 1.75],
            'Ts': [117, 294.0, 1.3, 1.65],
            'Og': [118, 294.0, 1.3, 1.57]
        }
        
        # Convert to numpy arrays
        return {k: np.array(v, dtype=np.float32) for k, v in properties.items()}
    
    def extract_node_features(self, structure: Structure) -> np.ndarray:
        """
        Extract node features for each atom in the structure.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Array of node features with shape [num_atoms, num_features]
        """
        num_atoms = len(structure)
        features = np.zeros((num_atoms, 4), dtype=np.float32)
        
        for i, site in enumerate(structure):
            element = site.specie.symbol
            if element in self.element_properties:
                features[i] = self.element_properties[element]
            else:
                logger.warning(f"Unknown element: {element}, using default features")
                features[i] = np.array([0, 0, 0, 0], dtype=np.float32)
        
        return features
    
    def extract_edge_index(self, structure: Structure) -> np.ndarray:
        """
        Extract edge indices (bonds) from the crystal structure.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Edge index array with shape [2, num_edges]
        """
        num_atoms = len(structure)
        edges = []
        
        # Get bonds using CrystalNN
        for i in range(num_atoms):
            try:
                nn_info = self.crystal_nn.get_nn_info(structure, i)
                for neighbor in nn_info:
                    j = neighbor["site_index"]
                    # Add both directions for undirected graph
                    edges.append([i, j])
                    edges.append([j, i])  # Ensure undirected graph
            except Exception as e:
                logger.warning(f"Error getting neighbors for atom {i}: {str(e)}")
        
        # If no edges were found, create self-loops
        if not edges:
            logger.warning(f"No edges found in structure, creating self-loops")
            edges = [[i, i] for i in range(num_atoms)]
        
        # Convert to numpy array and ensure proper shape
        edge_index = np.array(edges, dtype=np.int64).T
        return edge_index
    
    def extract_global_features(self, structure: Structure) -> np.ndarray:
        """
        Extract global features from the structure.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Array of global features
        """
        # Calculate basic structural properties
        volume = structure.volume
        density = structure.density
        num_atoms = len(structure)
        
        # Calculate composition features
        composition = structure.composition
        num_elements = len(composition.elements)
        
        # Calculate element statistics
        atomic_numbers = [site.specie.Z for site in structure]
        mean_atomic_number = np.mean(atomic_numbers)
        std_atomic_number = np.std(atomic_numbers)
        
        # Pack features
        features = np.array([
            volume,
            density,
            num_atoms,
            num_elements,
            mean_atomic_number,
            std_atomic_number
        ], dtype=np.float32)
        
        return features
    
    def structure_to_graph(self, structure: Structure) -> Data:
        """
        Convert a crystal structure to a graph for GNN processing.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            PyTorch Geometric Data object
        """
        # Extract node features
        node_features = self.extract_node_features(structure)
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Extract edge indices
        edge_index = self.extract_edge_index(structure)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # Extract global features
        global_features = self.extract_global_features(structure)
        global_features = torch.tensor(global_features, dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            global_features=global_features
        )
        
        return data

class SuperconductorDataset(Dataset):
    """Dataset for superconductor materials."""
    
    def __init__(self, 
                 structures_file: str,
                 properties_file: str,
                 target_property: str = "is_superconductor",
                 transform=None):
        """
        Initialize the dataset.
        
        Args:
            structures_file: Path to JSON file with structure data
            properties_file: Path to CSV file with material properties
            target_property: Name of the target property column
            transform: PyTorch Geometric transform
        """
        super().__init__(transform)
        
        self.structures_file = structures_file
        self.properties_file = properties_file
        self.target_property = target_property
        
        # Load data
        self.structures = self._load_structures()
        self.properties_df = self._load_properties()
        
        # Match structures with properties
        self.material_ids = list(self.structures.keys())
        
        # Feature extractor
        self.feature_extractor = StructureFeatureExtractor()
        
        logger.info(f"Loaded dataset with {len(self.material_ids)} materials")
    
    def _load_structures(self) -> Dict[str, Structure]:
        """Load structures from JSON file."""
        try:
            with open(self.structures_file, 'r') as f:
                structures_dict = json.load(f)
            
            # Convert to pymatgen Structure objects
            structures = {}
            for material_id, structure_dict in structures_dict.items():
                try:
                    structures[material_id] = Structure.from_dict(structure_dict)
                except Exception as e:
                    logger.warning(f"Error loading structure {material_id}: {str(e)}")
            
            return structures
            
        except Exception as e:
            logger.error(f"Error loading structures: {str(e)}")
            return {}
    
    def _load_properties(self) -> pd.DataFrame:
        """Load properties from CSV file."""
        try:
            df = pd.read_csv(self.properties_file)
            return df
        except Exception as e:
            logger.error(f"Error loading properties: {str(e)}")
            return pd.DataFrame()
    
    def len(self) -> int:
        """Return the number of materials in the dataset."""
        return len(self.material_ids)
    
    def get(self, idx: int) -> Data:
        """
        Get a single data point.
        
        Args:
            idx: Index of the material
            
        Returns:
            PyTorch Geometric Data object
        """
        material_id = self.material_ids[idx]
        
        # Get structure
        structure = self.structures[material_id]
        
        # Convert to graph
        data = self.feature_extractor.structure_to_graph(structure)
        
        # Add target property if available
        if not self.properties_df.empty:
            material_row = self.properties_df[self.properties_df['material_id'] == material_id]
            if not material_row.empty and self.target_property in material_row:
                target = material_row[self.target_property].values[0]
                data.y = torch.tensor([target], dtype=torch.float)
        
        # Add material ID for reference
        data.material_id = material_id
        
        return data

def normalize_features(dataset: SuperconductorDataset) -> Tuple[dict, SuperconductorDataset]:
    """
    Normalize node and global features in the dataset.
    
    Args:
        dataset: SuperconductorDataset to normalize
        
    Returns:
        Tuple of (normalization_stats, normalized_dataset)
    """
    # Collect all node features and global features
    all_node_features = []
    all_global_features = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        all_node_features.append(data.x.numpy())
        all_global_features.append(data.global_features.numpy())
    
    # Concatenate features
    all_node_features = np.vstack(all_node_features)
    all_global_features = np.vstack(all_global_features)
    
    # Calculate mean and std for normalization
    node_mean = np.mean(all_node_features, axis=0)
    node_std = np.std(all_node_features, axis=0)
    node_std[node_std == 0] = 1.0  # Avoid division by zero
    
    global_mean = np.mean(all_global_features, axis=0)
    global_std = np.std(all_global_features, axis=0)
    global_std[global_std == 0] = 1.0  # Avoid division by zero
    
    # Store normalization statistics
    norm_stats = {
        'node_mean': node_mean,
        'node_std': node_std,
        'global_mean': global_mean,
        'global_std': global_std
    }
    
    # Apply normalization to the dataset
    for i in range(len(dataset)):
        data = dataset[i]
        
        # Normalize node features
        x_normalized = (data.x.numpy() - node_mean) / node_std
        data.x = torch.tensor(x_normalized, dtype=torch.float)
        
        # Normalize global features
        global_normalized = (data.global_features.numpy() - global_mean) / global_std
        data.global_features = torch.tensor(global_normalized, dtype=torch.float)
    
    return norm_stats, dataset

def split_dataset(dataset: SuperconductorDataset, 
                 test_ratio: float = 0.2, 
                 val_ratio: float = 0.1,
                 random_state: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: SuperconductorDataset to split
        test_ratio: Ratio of test set
        val_ratio: Ratio of validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    n = len(dataset)
    indices = list(range(n))
    
    # First split: train+val and test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=random_state
    )
    
    # Second split: train and val
    val_size_relative = val_ratio / (1 - test_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size_relative, random_state=random_state
    )
    
    return train_indices, val_indices, test_indices

def prepare_superconductor_data() -> Tuple[SuperconductorDataset, dict, Tuple[List[int], List[int], List[int]]]:
    """
    Prepare superconductor dataset for training.
    
    Returns:
        Tuple of (dataset, normalization_stats, (train_indices, val_indices, test_indices))
    """
    # Load configuration
    data_dir = config.get("data_dir")
    structures_file = os.path.join(data_dir, config.get("superconductors_structures_file"))
    properties_file = os.path.join(data_dir, config.get("superconductors_file"))
    
    # Create dataset
    dataset = SuperconductorDataset(
        structures_file=structures_file,
        properties_file=properties_file,
        target_property="is_superconductor"
    )
    
    # Normalize features
    norm_stats, dataset = normalize_features(dataset)
    
    # Split dataset
    train_indices, val_indices, test_indices = split_dataset(
        dataset,
        test_ratio=config.get("test_split"),
        val_ratio=config.get("val_split")
    )
    
    return dataset, norm_stats, (train_indices, val_indices, test_indices)

if __name__ == "__main__":
    # Test the preprocessing module
    from config import load_config
    
    # Load configuration
    load_config()
    
    # Test feature extraction
    extractor = StructureFeatureExtractor()
    print("Feature extractor initialized")
    
    # Test dataset loading
    try:
        dataset, norm_stats, (train_idx, val_idx, test_idx) = prepare_superconductor_data()
        print(f"Dataset loaded with {len(dataset)} materials")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")