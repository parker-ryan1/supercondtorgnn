"""
Data collection module for superconductor analysis.

This module provides functionality to collect data from various sources,
including the Materials Project API and other databases.
"""

import os
import json
import pandas as pd
import numpy as np
from mp_api.client import MPRester
from typing import List, Dict, Any, Optional, Union
import logging
import time
from pathlib import Path
import requests
from tqdm import tqdm

# Import configuration
try:
    from config import config
except ImportError:
    # Fallback if config is not available
    config = {
        "data_dir": "data",
        "mp_api_key": os.getenv("MP_API_KEY")
    }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperconductorDataCollector:
    """Collects data from various sources for superconductor analysis."""
    
    def __init__(self, api_key: Optional[str] = None, data_dir: Optional[str] = None):
        """
        Initialize the data collector.
        
        Args:
            api_key: Materials Project API key (optional if set in environment)
            data_dir: Directory to save data (optional if set in config)
        """
        # Get API key from parameters, config, or environment
        self.api_key = api_key or config.get("mp_api_key") or os.getenv("MP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Materials Project API key is required. Set MP_API_KEY environment variable, "
                "pass it directly, or set it in the configuration."
            )
        
        # Set data directory
        self.data_dir = data_dir or config.get("data_dir", "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize Materials Project API client with retry logic
        self._init_mp_client()
        
        logger.info(f"SuperconductorDataCollector initialized with data directory: {self.data_dir}")
    
    def get_dft_data(self, material_id: str) -> Dict[str, Any]:
        """
        Get DFT data for a material from Materials Project.
        
        Args:
            material_id: Materials Project material ID
            
        Returns:
            Dictionary with DFT data
        """
        try:
            # Get material data
            material = self.mpr.materials.summary.get_data_by_id(material_id)
            
            if not material:
                raise ValueError(f"Material {material_id} not found")
            
            # Extract relevant properties
            dft_data = {
                "material_id": material_id,
                "formula": material.formula_pretty,
                "e_above_hull": material.energy_above_hull,
                "formation_energy_per_atom": material.formation_energy_per_atom,
                "band_gap": material.band_gap,
                "is_metal": material.is_metal,
                "density": material.density,
                "volume": material.volume,
                "spacegroup": material.symmetry.symbol if material.symmetry else None,
                "crystal_system": material.symmetry.crystal_system if material.symmetry else None,
                "is_stable": material.energy_above_hull < 0.1 if material.energy_above_hull is not None else False,
            }
            
            return dft_data
            
        except Exception as e:
            logger.error(f"Error fetching DFT data for {material_id}: {str(e)}")
            raise
    
    def _init_mp_client(self, max_retries: int = 3) -> None:
        """
        Initialize Materials Project API client with retry logic.
        
        Args:
            max_retries: Maximum number of retries on connection failure
        """
        for attempt in range(max_retries):
            try:
                self.mpr = MPRester(self.api_key)
                # Test connection - using correct API call without limit parameter
                _ = self.mpr.materials.summary.search(elements=["Si"], fields=["material_id"])[:1]
                logger.info("Successfully connected to Materials Project API")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Connection attempt {attempt+1} failed: {str(e)}. "
                                  f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect to Materials Project API after {max_retries} attempts")
                    raise ConnectionError(f"Could not connect to Materials Project API: {str(e)}")
    
    def fetch_ti_compounds(self, num_elements: tuple = (2, 5), max_results: Optional[int] = None) -> List[Dict[Any, Any]]:
        """
        Fetch titanium-based compounds from Materials Project.
        
        Args:
            num_elements: Range of number of elements in compounds (min, max)
            max_results: Maximum number of compounds to fetch (None for all)
            
        Returns:
            List of dictionaries with compound data
        """
        logger.info(f"Fetching Ti-based compounds (num_elements={num_elements}, max_results={max_results})...")
        
        try:
            # Query for Ti-containing materials - using correct API call
            docs = self.mpr.materials.summary.search(
                elements=["Ti"],
                num_elements=num_elements,
                fields=[
                    "material_id",
                    "formula_pretty",
                    "structure",
                    "formation_energy_per_atom",
                    "band_gap",
                    "density",
                    "symmetry",
                    "is_metal"
                ]
            )
            
            # Limit results if specified
            if max_results:
                docs = docs[:max_results]
            
            results = []
            for doc in docs:
                try:
                    result = {
                        "material_id": doc.material_id,
                        "formula": doc.formula_pretty,
                        "structure": doc.structure.as_dict(),
                        "formation_energy_per_atom": doc.formation_energy_per_atom,
                        "band_gap": doc.band_gap,
                        "density": doc.density,
                        "spacegroup": doc.symmetry.symbol if doc.symmetry else None,
                        "is_metal": doc.is_metal
                    }
                    results.append(result)
                except AttributeError as e:
                    logger.warning(f"Skipping material {doc.material_id if hasattr(doc, 'material_id') else 'unknown'}: {str(e)}")
                    continue
                
            logger.info(f"Found {len(results)} Ti-based compounds")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching Ti compounds: {str(e)}")
            raise
    
    def fetch_all_superconductors(self, is_metal: bool = True, max_results: Optional[int] = None) -> List[Dict[Any, Any]]:
        """
        Fetch known superconducting materials from Materials Project.
        
        Args:
            is_metal: Whether to filter for metallic materials
            max_results: Maximum number of materials to fetch (None for all)
            
        Returns:
            List of dictionaries with material data
        """
        logger.info(f"Fetching potential superconductors (is_metal={is_metal}, max_results={max_results})...")
        
        try:
            # Query for metallic materials (potential superconductors) - using correct API call
            docs = self.mpr.materials.summary.search(
                is_metal=is_metal,
                fields=[
                    "material_id",
                    "formula_pretty",
                    "structure",
                    "formation_energy_per_atom",
                    "band_gap",
                    "density",
                    "symmetry",
                    "is_metal"
                ]
            )
            
            # Limit results if specified
            if max_results:
                docs = docs[:max_results]
            
            results = []
            for doc in docs:
                try:
                    result = {
                        "material_id": doc.material_id,
                        "formula": doc.formula_pretty,
                        "structure": doc.structure.as_dict(),
                        "formation_energy_per_atom": doc.formation_energy_per_atom,
                        "band_gap": doc.band_gap,
                        "density": doc.density,
                        "spacegroup": doc.symmetry.symbol if doc.symmetry else None,
                        "is_metal": doc.is_metal
                    }
                    results.append(result)
                except AttributeError as e:
                    logger.warning(f"Skipping material {doc.material_id if hasattr(doc, 'material_id') else 'unknown'}: {str(e)}")
                    continue
                
            logger.info(f"Found {len(results)} potential superconducting materials")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching superconductors: {str(e)}")
            raise
    
    def fetch_supercon_database(self, url: str = "https://supercon.nims.go.jp/export/supercon_export.csv") -> pd.DataFrame:
        """
        Fetch data from the SuperCon database.
        
        Args:
            url: URL of the SuperCon database export
            
        Returns:
            DataFrame with SuperCon data
        """
        logger.info(f"Fetching data from SuperCon database: {url}")
        
        try:
            # Download the CSV file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save to temporary file
            temp_file = os.path.join(self.data_dir, "supercon_temp.csv")
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Read CSV file
            df = pd.read_csv(temp_file, encoding='utf-8')
            
            # Clean up temporary file
            os.remove(temp_file)
            
            logger.info(f"Successfully fetched {len(df)} entries from SuperCon database")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading SuperCon database: {str(e)}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing SuperCon data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching SuperCon data: {str(e)}")
            raise
    
    def save_data(self, data: List[Dict[Any, Any]], filename: str) -> Dict[str, str]:
        """
        Save the collected data to files.
        
        Args:
            data: List of dictionaries with material data
            filename: Base filename (without extension)
            
        Returns:
            Dictionary with paths to saved files
        """
        if not data:
            logger.warning(f"No data to save for {filename}")
            return {}
        
        try:
            # Create data directory if it doesn't exist
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Prepare paths
            csv_path = os.path.join(self.data_dir, f"{filename}.csv")
            json_path = os.path.join(self.data_dir, f"{filename}_structures.json")
            
            # Extract non-structure properties for CSV
            df = pd.DataFrame([
                {k: v for k, v in item.items() if k != 'structure'}
                for item in data
            ])
            
            # Save basic properties as CSV
            df.to_csv(csv_path, index=False)
            
            # Save structures separately as JSON
            structures = {item['material_id']: item['structure'] 
                         for item in data if 'structure' in item}
            
            with open(json_path, 'w') as f:
                json.dump(structures, f)
            
            logger.info(f"Data saved to {csv_path} and {json_path}")
            
            return {
                "csv": csv_path,
                "structures": json_path
            }
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def merge_datasets(self, mp_data: List[Dict[Any, Any]], supercon_df: pd.DataFrame) -> List[Dict[Any, Any]]:
        """
        Merge Materials Project data with SuperCon database data.
        
        Args:
            mp_data: List of dictionaries with Materials Project data
            supercon_df: DataFrame with SuperCon data
            
        Returns:
            List of merged dictionaries
        """
        logger.info("Merging Materials Project and SuperCon datasets...")
        
        try:
            # Create a mapping of formulas to SuperCon entries
            formula_to_tc = {}
            for _, row in supercon_df.iterrows():
                formula = row.get('formula', '')
                tc = row.get('tc', 0)
                if formula and tc > 0:
                    formula_to_tc[formula] = max(tc, formula_to_tc.get(formula, 0))
            
            # Merge data
            merged_data = []
            for item in mp_data:
                formula = item.get('formula', '')
                
                # Try to find a match in SuperCon data
                tc = formula_to_tc.get(formula, 0)
                
                # Add Tc to the item
                item_copy = item.copy()
                item_copy['tc'] = tc
                item_copy['is_superconductor'] = 1 if tc > 0 else 0
                
                merged_data.append(item_copy)
            
            logger.info(f"Merged dataset contains {len(merged_data)} materials")
            return merged_data
            
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            raise
    
    def collect_and_process_all(self, max_results: Optional[int] = None) -> Dict[str, str]:
        """
        Collect and process all data in one go.
        
        Args:
            max_results: Maximum number of materials to fetch from each source
            
        Returns:
            Dictionary with paths to saved files
        """
        logger.info("Starting complete data collection and processing...")
        
        try:
            # Fetch titanium compounds
            ti_compounds = self.fetch_ti_compounds(max_results=max_results)
            ti_files = self.save_data(ti_compounds, "ti_compounds")
            
            # Fetch potential superconductors
            superconductors = self.fetch_all_superconductors(max_results=max_results)
            sc_files = self.save_data(superconductors, "superconductors")
            
            # Try to fetch SuperCon database
            try:
                supercon_df = self.fetch_supercon_database()
                
                # Merge Materials Project data with SuperCon data
                merged_data = self.merge_datasets(superconductors, supercon_df)
                merged_files = self.save_data(merged_data, "merged_superconductors")
                
                # Save SuperCon data separately
                supercon_path = os.path.join(self.data_dir, "supercon_database.csv")
                supercon_df.to_csv(supercon_path, index=False)
                
                logger.info(f"SuperCon database saved to {supercon_path}")
                
            except Exception as e:
                logger.warning(f"Could not fetch or process SuperCon data: {str(e)}")
                merged_files = {}
            
            # Return all file paths
            result = {
                "ti_compounds": ti_files,
                "superconductors": sc_files
            }
            
            if merged_files:
                result["merged"] = merged_files
            
            logger.info("Data collection and processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in collect_and_process_all: {str(e)}")
            raise

def main():
    """Main function for command-line usage."""
    try:
        # Get API key from environment
        api_key = os.getenv("MP_API_KEY")
        if not api_key:
            logger.error("MP_API_KEY environment variable not set")
            return 1
        
        # Create data collector
        collector = SuperconductorDataCollector(api_key=api_key)
        
        # Collect all data
        collector.collect_and_process_all(max_results=100)  # Limit for testing
        
        return 0
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)