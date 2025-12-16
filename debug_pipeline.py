#!/usr/bin/env python3
"""
Debug script for the superconductor prediction pipeline.
This script helps diagnose issues in the pipeline components.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check Python environment and dependencies."""
    logger.info("Checking Python environment...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("PyTorch not installed")
    
    try:
        import torch_geometric
        logger.info(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        logger.error("PyTorch Geometric not installed")
    
    try:
        import pymatgen
        logger.info(f"Pymatgen version: {pymatgen.__version__}")
    except ImportError:
        logger.error("Pymatgen not installed")
    
    try:
        from mp_api.client import MPRester
        logger.info("Materials Project API client available")
    except ImportError:
        logger.error("Materials Project API client not installed")

def check_directory_structure():
    """Check project directory structure."""
    logger.info("Checking directory structure...")
    
    # Get current directory
    current_dir = Path.cwd()
    logger.info(f"Current directory: {current_dir}")
    
    # Check required directories
    required_dirs = ["data", "models", "src", "tests"]
    for dir_name in required_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            logger.info(f"✓ Directory exists: {dir_name}")
        else:
            logger.error(f"✗ Directory missing: {dir_name}")
    
    # Check src files
    src_dir = current_dir / "src"
    if src_dir.exists() and src_dir.is_dir():
        logger.info("Checking source files...")
        required_files = [
            "config.py", 
            "data_collector.py", 
            "data_preprocessing.py", 
            "gnn_model.py", 
            "structure_visualizer.py"
        ]
        
        for file_name in required_files:
            file_path = src_dir / file_name
            if file_path.exists() and file_path.is_file():
                logger.info(f"✓ Source file exists: {file_name}")
            else:
                logger.error(f"✗ Source file missing: {file_name}")

def test_mp_api():
    """Test Materials Project API connection."""
    logger.info("Testing Materials Project API connection...")
    
    api_key = "DjKx0q7YivC5u73uKFIVPif813v7InYq"
    
    try:
        from mp_api.client import MPRester
        
        with MPRester(api_key) as mpr:
            # Try a simple query
            logger.info("Querying Materials Project API...")
            materials = mpr.summary.search(elements=["Ti"], fields=["material_id"], limit=2)
            
            if materials and len(materials) > 0:
                logger.info(f"✓ API connection successful. Found {len(materials)} materials.")
                for material in materials:
                    logger.info(f"  - {material.material_id}")
            else:
                logger.error("✗ API query returned no results")
    
    except Exception as e:
        logger.error(f"✗ API connection failed: {str(e)}")

def create_minimal_config():
    """Create a minimal configuration file."""
    logger.info("Creating minimal configuration file...")
    
    config_path = Path.cwd() / "config.json"
    
    config = {
        "mp_api_key": "DjKx0q7YivC5u73uKFIVPif813v7InYq",
        "data_dir": "data",
        "models_dir": "models",
        "results_dir": "results",
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "hidden_dim": 64,
        "use_gpu": True
    }
    
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✓ Configuration file created: {config_path}")

def create_missing_files():
    """Create missing required files."""
    logger.info("Creating missing required files...")
    
    # Create src directory if it doesn't exist
    src_dir = Path.cwd() / "src"
    src_dir.mkdir(exist_ok=True)
    
    # Check and create config.py if missing
    config_file = src_dir / "config.py"
    if not config_file.exists():
        logger.info(f"Creating {config_file}...")
        with open(config_file, 'w') as f:
            f.write('''"""
Configuration system for the Superconductor Analysis and Prediction project.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the project."""
    
    DEFAULT_CONFIG = {
        "mp_api_key": "DjKx0q7YivC5u73uKFIVPif813v7InYq",
        "data_dir": "data",
        "models_dir": "models",
        "results_dir": "results",
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "hidden_dim": 64,
        "use_gpu": True
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize with default configuration."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self.config[key] = value

# Create a global configuration instance
config = Config()

def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration from a file."""
    global config
    config = Config(config_file)
    return config
''')
        logger.info(f"✓ Created {config_file}")
    
    # Create data_preprocessing.py if missing
    preproc_file = src_dir / "data_preprocessing.py"
    if not preproc_file.exists():
        logger.info(f"Creating {preproc_file}...")
        with open(preproc_file, 'w') as f:
            f.write('''"""
Data preprocessing module for superconductor analysis.
"""

import os
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
import logging

logger = logging.getLogger(__name__)

def load_structures(file_path):
    """Load structures from JSON file."""
    import json
    with open(file_path, 'r') as f:
        return json.load(f)

def load_properties(file_path):
    """Load properties from CSV file."""
    return pd.read_csv(file_path)

def extract_features(structure):
    """Extract features from a crystal structure."""
    # Simple feature extraction
    num_atoms = len(structure)
    elements = set(site.specie.symbol for site in structure)
    
    return {
        "num_atoms": num_atoms,
        "num_elements": len(elements),
        "volume": structure.volume,
        "density": structure.density
    }
''')
        logger.info(f"✓ Created {preproc_file}")

def run_basic_test():
    """Run a basic test of the pipeline components."""
    logger.info("Running basic test of pipeline components...")
    
    # Test importing modules
    try:
        logger.info("Importing modules...")
        
        # Add src to path
        src_dir = str(Path.cwd() / "src")
        if src_dir not in sys.path:
            sys.path.append(src_dir)
        
        # Try importing modules
        import_errors = []
        
        try:
            from config import config
            logger.info("✓ Imported config module")
        except ImportError as e:
            logger.error(f"✗ Failed to import config module: {str(e)}")
            import_errors.append("config")
        
        try:
            from data_collector import SuperconductorDataCollector
            logger.info("✓ Imported data_collector module")
        except ImportError as e:
            logger.error(f"✗ Failed to import data_collector module: {str(e)}")
            import_errors.append("data_collector")
        
        try:
            from gnn_model import SuperconductorTcPredictor
            logger.info("✓ Imported gnn_model module")
        except ImportError as e:
            logger.error(f"✗ Failed to import gnn_model module: {str(e)}")
            import_errors.append("gnn_model")
        
        if import_errors:
            logger.error(f"Import errors found in: {', '.join(import_errors)}")
            return False
        
        # Test data collector
        try:
            logger.info("Testing data collector...")
            collector = SuperconductorDataCollector(api_key="DjKx0q7YivC5u73uKFIVPif813v7InYq")
            logger.info("✓ Created data collector")
            
            # Test fetching a small amount of data
            logger.info("Fetching a small amount of data...")
            data = collector.fetch_ti_compounds(limit=2)
            logger.info(f"✓ Fetched {len(data)} compounds")
            
            return True
        except Exception as e:
            logger.error(f"✗ Error testing data collector: {str(e)}")
            return False
    
    except Exception as e:
        logger.error(f"✗ Error in basic test: {str(e)}")
        return False

def main():
    """Main entry point."""
    logger.info("Starting debug script...")
    
    # Check environment
    check_environment()
    
    # Check directory structure
    check_directory_structure()
    
    # Create minimal configuration
    create_minimal_config()
    
    # Create missing files
    create_missing_files()
    
    # Test Materials Project API
    test_mp_api()
    
    # Run basic test
    success = run_basic_test()
    
    if success:
        logger.info("✓ Basic test completed successfully")
        logger.info("The environment appears to be set up correctly.")
        logger.info("You can now try running the full pipeline.")
    else:
        logger.error("✗ Basic test failed")
        logger.error("Please fix the issues before running the full pipeline.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
