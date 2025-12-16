"""
Unit tests for the data collector module.
"""

import os
import sys
import unittest
import tempfile
import json
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_collector import SuperconductorDataCollector

class TestDataCollector(unittest.TestCase):
    """Test cases for the SuperconductorDataCollector class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for data
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock API key
        self.api_key = "test_api_key"
        
        # Mock environment
        patcher = patch.dict(os.environ, {"MP_API_KEY": self.api_key})
        patcher.start()
        self.addCleanup(patcher.stop)
        
        # Mock MPRester
        self.mp_patcher = patch("data_collector.MPRester")
        self.mock_mpr = self.mp_patcher.start()
        
        # Set up mock MPRester instance
        self.mock_mpr_instance = MagicMock()
        self.mock_mpr.return_value = self.mock_mpr_instance
        
        # Mock materials.summary.search
        self.mock_materials = MagicMock()
        self.mock_mpr_instance.materials.summary.search = self.mock_materials
        
        # Set up mock search results
        self.mock_doc = MagicMock()
        self.mock_doc.material_id = "mp-123"
        self.mock_doc.formula_pretty = "TiO2"
        self.mock_doc.formation_energy_per_atom = -3.0
        self.mock_doc.band_gap = 2.0
        self.mock_doc.density = 4.23
        self.mock_doc.symmetry.symbol = "P42/mnm"
        self.mock_doc.is_metal = False
        
        # Mock structure
        self.mock_doc.structure.as_dict.return_value = {"lattice": {"a": 4.59, "b": 4.59, "c": 2.96}}
        
        # Set up mock search results
        self.mock_materials.return_value = [self.mock_doc]
        
        # Create collector
        self.collector = SuperconductorDataCollector(api_key=self.api_key, data_dir=self.temp_dir)
    
    def tearDown(self):
        """Tear down test environment."""
        # Stop patches
        self.mp_patcher.stop()
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization of data collector."""
        self.assertEqual(self.collector.api_key, self.api_key)
        self.assertEqual(self.collector.data_dir, self.temp_dir)
        self.assertEqual(self.collector.mpr, self.mock_mpr_instance)
    
    def test_fetch_ti_compounds(self):
        """Test fetching titanium compounds."""
        # Call the method
        results = self.collector.fetch_ti_compounds()
        
        # Check that the API was called correctly
        self.mock_materials.assert_called_once()
        call_args = self.mock_materials.call_args[1]
        self.assertEqual(call_args["elements"], ["Ti"])
        self.assertEqual(call_args["num_elements"], (2, 5))
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["material_id"], "mp-123")
        self.assertEqual(results[0]["formula"], "TiO2")
        self.assertEqual(results[0]["formation_energy_per_atom"], -3.0)
        self.assertEqual(results[0]["band_gap"], 2.0)
        self.assertEqual(results[0]["density"], 4.23)
        self.assertEqual(results[0]["spacegroup"], "P42/mnm")
        self.assertFalse(results[0]["is_metal"])
        self.assertEqual(results[0]["structure"], {"lattice": {"a": 4.59, "b": 4.59, "c": 2.96}})
    
    def test_fetch_all_superconductors(self):
        """Test fetching all potential superconductors."""
        # Set up mock for metallic materials
        mock_metal_doc = MagicMock()
        mock_metal_doc.material_id = "mp-456"
        mock_metal_doc.formula_pretty = "Cu"
        mock_metal_doc.formation_energy_per_atom = -1.0
        mock_metal_doc.band_gap = 0.0
        mock_metal_doc.density = 8.96
        mock_metal_doc.symmetry.symbol = "Fm-3m"
        mock_metal_doc.is_metal = True
        mock_metal_doc.structure.as_dict.return_value = {"lattice": {"a": 3.61, "b": 3.61, "c": 3.61}}
        
        # Update mock search results
        self.mock_materials.return_value = [mock_metal_doc]
        
        # Call the method
        results = self.collector.fetch_all_superconductors()
        
        # Check that the API was called correctly
        call_args = self.mock_materials.call_args[1]
        self.assertTrue(call_args["is_metal"])
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["material_id"], "mp-456")
        self.assertEqual(results[0]["formula"], "Cu")
        self.assertTrue(results[0]["is_metal"])
    
    def test_save_data(self):
        """Test saving data to files."""
        # Create test data
        test_data = [
            {
                "material_id": "mp-123",
                "formula": "TiO2",
                "structure": {"lattice": {"a": 4.59, "b": 4.59, "c": 2.96}},
                "formation_energy_per_atom": -3.0,
                "band_gap": 2.0,
                "density": 4.23,
                "spacegroup": "P42/mnm",
                "is_metal": False
            }
        ]
        
        # Save data
        result_paths = self.collector.save_data(test_data, "test_data")
        
        # Check that files were created
        csv_path = os.path.join(self.temp_dir, "test_data.csv")
        json_path = os.path.join(self.temp_dir, "test_data_structures.json")
        
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        
        # Check CSV content
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["material_id"], "mp-123")
        self.assertEqual(df.iloc[0]["formula"], "TiO2")
        self.assertEqual(df.iloc[0]["formation_energy_per_atom"], -3.0)
        
        # Check JSON content
        with open(json_path, 'r') as f:
            structures = json.load(f)
        
        self.assertIn("mp-123", structures)
        self.assertEqual(structures["mp-123"]["lattice"]["a"], 4.59)
    
    def test_merge_datasets(self):
        """Test merging datasets."""
        # Create Materials Project data
        mp_data = [
            {
                "material_id": "mp-123",
                "formula": "TiO2",
                "structure": {"lattice": {"a": 4.59, "b": 4.59, "c": 2.96}},
                "is_metal": False
            },
            {
                "material_id": "mp-456",
                "formula": "Cu",
                "structure": {"lattice": {"a": 3.61, "b": 3.61, "c": 3.61}},
                "is_metal": True
            }
        ]
        
        # Create SuperCon data
        supercon_data = {
            "formula": ["Cu", "Nb3Ge"],
            "tc": [1.2, 23.2]
        }
        supercon_df = pd.DataFrame(supercon_data)
        
        # Merge datasets
        merged = self.collector.merge_datasets(mp_data, supercon_df)
        
        # Check results
        self.assertEqual(len(merged), 2)
        
        # Check Cu (should have Tc)
        cu_data = next(item for item in merged if item["formula"] == "Cu")
        self.assertEqual(cu_data["tc"], 1.2)
        self.assertEqual(cu_data["is_superconductor"], 1)
        
        # Check TiO2 (should not have Tc)
        tio2_data = next(item for item in merged if item["formula"] == "TiO2")
        self.assertEqual(tio2_data["tc"], 0)
        self.assertEqual(tio2_data["is_superconductor"], 0)
    
    @patch("data_collector.requests.get")
    def test_fetch_supercon_database(self, mock_get):
        """Test fetching SuperCon database."""
        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        
        # Create a temporary CSV file for the mock response
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as temp_file:
            temp_file.write("formula,tc,other_prop\nCu,1.2,value1\nNb3Ge,23.2,value2\n")
            temp_file_path = temp_file.name
        
        # Set up the mock to return the file content
        with open(temp_file_path, 'rb') as f:
            mock_response.iter_content.return_value = [f.read()]
        
        mock_get.return_value = mock_response
        
        try:
            # Call the method
            df = self.collector.fetch_supercon_database()
            
            # Check results
            self.assertEqual(len(df), 2)
            self.assertEqual(df.iloc[0]["formula"], "Cu")
            self.assertEqual(df.iloc[0]["tc"], 1.2)
            self.assertEqual(df.iloc[1]["formula"], "Nb3Ge")
            self.assertEqual(df.iloc[1]["tc"], 23.2)
            
        finally:
            # Clean up
            os.unlink(temp_file_path)

if __name__ == "__main__":
    unittest.main()
