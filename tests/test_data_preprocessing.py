"""
Unit tests for the data preprocessing module.
"""

import os
import sys
import unittest
import tempfile
import json
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_preprocessing import StructureFeatureExtractor, SuperconductorDataset, normalize_features, split_dataset
from pymatgen.core import Structure, Lattice

class TestStructureFeatureExtractor(unittest.TestCase):
    """Test cases for the StructureFeatureExtractor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.extractor = StructureFeatureExtractor()
        
        # Create a simple test structure (TiO2)
        lattice = Lattice.cubic(4.0)
        coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
        species = ["Ti", "Ti", "O", "O"]
        self.test_structure = Structure(lattice, species, coords)
    
    def test_initialization(self):
        """Test initialization of feature extractor."""
        self.assertIsNotNone(self.extractor.crystal_nn)
        self.assertIsNotNone(self.extractor.element_properties)
        self.assertIn("Ti", self.extractor.element_properties)
        self.assertIn("O", self.extractor.element_properties)
    
    def test_extract_node_features(self):
        """Test extraction of node features."""
        features = self.extractor.extract_node_features(self.test_structure)
        
        # Check shape and content
        self.assertEqual(features.shape, (4, 4))  # 4 atoms, 4 features per atom
        
        # Check Ti features
        ti_features = features[0]
        self.assertEqual(ti_features[0], 22)  # Atomic number
        self.assertAlmostEqual(ti_features[1], 47.867, places=2)  # Atomic mass
        
        # Check O features
        o_features = features[2]
        self.assertEqual(o_features[0], 8)  # Atomic number
        self.assertAlmostEqual(o_features[1], 15.999, places=2)  # Atomic mass
    
    def test_extract_edge_index(self):
        """Test extraction of edge indices."""
        edge_index = self.extractor.extract_edge_index(self.test_structure)
        
        # Check shape
        self.assertEqual(edge_index.shape[0], 2)  # Source and target indices
        
        # Check for undirected graph (each edge appears in both directions)
        edges = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            edges.add((src, dst))
        
        # Check for symmetry in edges
        for src, dst in list(edges):
            self.assertIn((dst, src), edges)
    
    def test_extract_global_features(self):
        """Test extraction of global features."""
        global_features = self.extractor.extract_global_features(self.test_structure)
        
        # Check shape
        self.assertEqual(global_features.shape, (6,))
        
        # Check content
        self.assertAlmostEqual(global_features[0], 64.0, places=1)  # Volume
        self.assertGreater(global_features[1], 0)  # Density
        self.assertEqual(global_features[2], 4)  # Number of atoms
        self.assertEqual(global_features[3], 2)  # Number of elements
    
    def test_structure_to_graph(self):
        """Test conversion of structure to graph."""
        graph = self.extractor.structure_to_graph(self.test_structure)
        
        # Check graph properties
        self.assertIsNotNone(graph.x)
        self.assertIsNotNone(graph.edge_index)
        self.assertIsNotNone(graph.global_features)
        
        # Check shapes
        self.assertEqual(graph.x.shape[0], 4)  # 4 atoms
        self.assertEqual(graph.x.shape[1], 4)  # 4 features per atom
        self.assertEqual(graph.edge_index.shape[0], 2)  # Source and target indices
        self.assertEqual(graph.global_features.shape[0], 6)  # 6 global features

class TestSuperconductorDataset(unittest.TestCase):
    """Test cases for the SuperconductorDataset class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test structures
        structures = {
            "mp-123": {
                "lattice": {"a": 4.0, "b": 4.0, "c": 4.0, "alpha": 90, "beta": 90, "gamma": 90},
                "sites": [
                    {"species": [{"element": "Ti", "occu": 1.0}], "xyz": [0, 0, 0]},
                    {"species": [{"element": "Ti", "occu": 1.0}], "xyz": [2, 2, 2]},
                    {"species": [{"element": "O", "occu": 1.0}], "xyz": [2, 2, 0]},
                    {"species": [{"element": "O", "occu": 1.0}], "xyz": [0, 2, 2]}
                ]
            },
            "mp-456": {
                "lattice": {"a": 3.6, "b": 3.6, "c": 3.6, "alpha": 90, "beta": 90, "gamma": 90},
                "sites": [
                    {"species": [{"element": "Cu", "occu": 1.0}], "xyz": [0, 0, 0]},
                    {"species": [{"element": "Cu", "occu": 1.0}], "xyz": [1.8, 1.8, 1.8]}
                ]
            }
        }
        
        # Create properties dataframe
        properties = {
            "material_id": ["mp-123", "mp-456"],
            "formula": ["TiO2", "Cu"],
            "is_superconductor": [0, 1],
            "tc": [0, 1.2]
        }
        
        # Save to temporary files
        self.structures_file = os.path.join(self.temp_dir, "test_structures.json")
        with open(self.structures_file, 'w') as f:
            json.dump(structures, f)
        
        self.properties_file = os.path.join(self.temp_dir, "test_properties.csv")
        import pandas as pd
        pd.DataFrame(properties).to_csv(self.properties_file, index=False)
        
        # Create dataset
        self.dataset = SuperconductorDataset(
            structures_file=self.structures_file,
            properties_file=self.properties_file,
            target_property="is_superconductor"
        )
    
    def tearDown(self):
        """Tear down test environment."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization of dataset."""
        self.assertEqual(len(self.dataset), 2)
        self.assertEqual(len(self.dataset.material_ids), 2)
        self.assertIn("mp-123", self.dataset.structures)
        self.assertIn("mp-456", self.dataset.structures)
    
    def test_get_item(self):
        """Test getting an item from the dataset."""
        data = self.dataset[0]
        
        # Check data properties
        self.assertIsNotNone(data.x)
        self.assertIsNotNone(data.edge_index)
        self.assertIsNotNone(data.global_features)
        self.assertIsNotNone(data.y)
        self.assertIsNotNone(data.material_id)
        
        # Check shapes
        self.assertEqual(data.x.shape[1], 4)  # 4 features per atom
        self.assertEqual(data.edge_index.shape[0], 2)  # Source and target indices
        self.assertEqual(data.y.shape, (1,))  # Single target value

class TestDataPreprocessingFunctions(unittest.TestCase):
    """Test cases for the data preprocessing functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock dataset
        self.mock_dataset = MagicMock()
        self.mock_dataset.__len__.return_value = 2
        
        # Create mock data objects
        data1 = MagicMock()
        data1.x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)
        data1.global_features = torch.tensor([10.0, 20.0], dtype=torch.float)
        data1.y = torch.tensor([0.0], dtype=torch.float)
        
        data2 = MagicMock()
        data2.x = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float)
        data2.global_features = torch.tensor([30.0, 40.0], dtype=torch.float)
        data2.y = torch.tensor([1.0], dtype=torch.float)
        
        # Set up __getitem__
        self.mock_dataset.__getitem__.side_effect = [data1, data2]
    
    def test_normalize_features(self):
        """Test feature normalization."""
        norm_stats, normalized_dataset = normalize_features(self.mock_dataset)
        
        # Check normalization stats
        self.assertIn("node_mean", norm_stats)
        self.assertIn("node_std", norm_stats)
        self.assertIn("global_mean", norm_stats)
        self.assertIn("global_std", norm_stats)
        
        # Check normalized dataset
        self.assertEqual(normalized_dataset, self.mock_dataset)
        
        # Check that normalization was applied
        self.mock_dataset.__getitem__.assert_called()
    
    def test_split_dataset(self):
        """Test dataset splitting."""
        train_indices, val_indices, test_indices = split_dataset(
            self.mock_dataset,
            test_ratio=0.2,
            val_ratio=0.1,
            random_state=42
        )
        
        # Check indices
        self.assertEqual(len(train_indices) + len(val_indices) + len(test_indices), 2)
        
        # Check no overlap
        all_indices = set(train_indices + val_indices + test_indices)
        self.assertEqual(len(all_indices), 2)

if __name__ == "__main__":
    unittest.main()
