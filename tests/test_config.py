"""
Unit tests for the configuration module.
"""

import os
import sys
import unittest
import tempfile
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config, load_config

class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment variables
        self.original_env = {}
        for key in ["MP_API_KEY", "SUPERCONDUCTOR_DATA_DIR"]:
            if key in os.environ:
                self.original_env[key] = os.environ[key]
                del os.environ[key]
    
    def tearDown(self):
        """Tear down test environment."""
        # Restore original environment variables
        for key, value in self.original_env.items():
            os.environ[key] = value
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        self.assertIsNone(config.get("mp_api_key"))
        self.assertEqual(config.get("data_dir"), "data")
        self.assertEqual(config.get("models_dir"), "models")
        self.assertTrue(config.get("use_gpu"))
    
    def test_config_from_file(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
            config_data = {
                "mp_api_key": "test_key",
                "data_dir": "test_data",
                "use_gpu": False
            }
            json.dump(config_data, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Load config from file
            config = Config(config_file=temp_file_path)
            
            # Check values
            self.assertEqual(config.get("mp_api_key"), "test_key")
            self.assertEqual(config.get("data_dir"), "test_data")
            self.assertFalse(config.get("use_gpu"))
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_config_from_env(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ["MP_API_KEY"] = "env_key"
        os.environ["SUPERCONDUCTOR_DATA_DIR"] = "env_data"
        os.environ["SUPERCONDUCTOR_USE_GPU"] = "false"
        
        # Load config
        config = Config()
        
        # Check values
        self.assertEqual(config.get("mp_api_key"), "env_key")
        self.assertEqual(config.get("data_dir"), "env_data")
        self.assertFalse(config.get("use_gpu"))
    
    def test_config_precedence(self):
        """Test configuration precedence (file overrides default, env overrides file)."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
            config_data = {
                "mp_api_key": "file_key",
                "data_dir": "file_data",
                "use_gpu": False
            }
            json.dump(config_data, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Set environment variables
            os.environ["MP_API_KEY"] = "env_key"
            
            # Load config from file and env
            config = Config(config_file=temp_file_path)
            
            # Check values (env should override file)
            self.assertEqual(config.get("mp_api_key"), "env_key")
            self.assertEqual(config.get("data_dir"), "file_data")
            self.assertFalse(config.get("use_gpu"))
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_set_and_get(self):
        """Test setting and getting configuration values."""
        config = Config()
        
        # Set values
        config.set("test_key", "test_value")
        config.set("test_number", 42)
        
        # Get values
        self.assertEqual(config.get("test_key"), "test_value")
        self.assertEqual(config.get("test_number"), 42)
        self.assertEqual(config.get("nonexistent_key", "default"), "default")
    
    def test_save_config(self):
        """Test saving configuration to a file."""
        config = Config()
        
        # Set some values
        config.set("mp_api_key", "save_test_key")
        config.set("data_dir", Path("test_data_dir"))
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            # Save config
            result = config.save(temp_file_path)
            self.assertTrue(result)
            
            # Load the saved config
            with open(temp_file_path, 'r') as f:
                saved_data = json.load(f)
            
            # Check values
            self.assertEqual(saved_data["mp_api_key"], "save_test_key")
            self.assertEqual(saved_data["data_dir"], "test_data_dir")
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_load_config_function(self):
        """Test the load_config function."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
            config_data = {
                "mp_api_key": "global_test_key",
                "data_dir": "global_test_data"
            }
            json.dump(config_data, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Load global config
            global_config = load_config(temp_file_path)
            
            # Check values
            self.assertEqual(global_config.get("mp_api_key"), "global_test_key")
            self.assertEqual(global_config.get("data_dir"), "global_test_data")
            
        finally:
            # Clean up
            os.unlink(temp_file_path)

if __name__ == "__main__":
    unittest.main()
