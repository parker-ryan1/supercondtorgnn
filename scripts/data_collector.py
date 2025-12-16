import os
import json
import pandas as pd
from mp_api.client import MPRester
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperconductorDataCollector:
    def __init__(self, api_key: str = None):
        """
        Initialize the data collector.
        :param api_key: Materials Project API key
        """
        self.api_key = api_key or os.getenv("MP_API_KEY")
        if not self.api_key:
            raise ValueError("Materials Project API key is required. Set MP_API_KEY environment variable or pass it directly.")
        
        self.mpr = MPRester(self.api_key)

    def fetch_ti_compounds(self) -> List[Dict[Any, Any]]:
        """
        Fetch titanium-based compounds from Materials Project.
        """
        logger.info("Fetching Ti-based compounds...")
        
        # Query for Ti-containing materials
        docs = self.mpr.materials.summary.search(
            elements=["Ti"],
            num_elements=(2, 5),  # compounds with 2-5 elements
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
        
        results = []
        for doc in docs:
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
            
        logger.info(f"Found {len(results)} Ti-based compounds")
        return results

    def fetch_all_superconductors(self) -> List[Dict[Any, Any]]:
        """
        Fetch known superconducting materials from Materials Project.
        """
        logger.info("Fetching potential superconductors...")
        
        # Query for metallic materials (potential superconductors)
        docs = self.mpr.materials.summary.search(
            is_metal=True,
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
        
        results = []
        for doc in docs:
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
            
        logger.info(f"Found {len(results)} potential superconducting materials")
        return results

    def save_data(self, data: List[Dict[Any, Any]], filename: str):
        """
        Save the collected data to a file.
        """
        df = pd.DataFrame([
            {k: v for k, v in item.items() if k != 'structure'}
            for item in data
        ])
        os.makedirs("data", exist_ok=True)
        
        # Save basic properties as CSV
        df.to_csv(f"data/{filename}.csv", index=False)
        
        # Save structures separately as JSON
        structures = {item['material_id']: item['structure'] 
                     for item in data if 'structure' in item}
        with open(f"data/{filename}_structures.json", 'w') as f:
            json.dump(structures, f)
        
        logger.info(f"Data saved to data/{filename}.csv and data/{filename}_structures.json")

def main():
    try:
        collector = SuperconductorDataCollector()
        
        # Collect Ti-based compounds
        ti_compounds = collector.fetch_ti_compounds()
        collector.save_data(ti_compounds, "ti_compounds")
        
        # Collect all potential superconductors
        superconductors = collector.fetch_all_superconductors()
        collector.save_data(superconductors, "superconductors")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 