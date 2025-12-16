import os
import json
import numpy as np
# Force matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import logging
from typing import Dict, Any
from ase.visualize.plot import plot_atoms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_rotation_matrix(angles):
    """
    Create a rotation matrix from angles in degrees.
    angles: [phi, theta, psi] in degrees
    """
    # Convert angles to radians
    phi, theta, psi = np.radians(angles)
    
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    
    # Combine rotations
    R = Rz @ Ry @ Rx
    return R

class StructureVisualizer:
    def __init__(self):
        """Initialize the structure visualizer"""
        self.adaptor = AseAtomsAdaptor()
        
    def load_structure(self, structure_dict: Dict[str, Any]) -> Structure:
        """Convert dictionary to pymatgen Structure"""
        return Structure.from_dict(structure_dict)
    
    def visualize_structure(self, structure: Structure, material_id: str, output_dir: str):
        """
        Generate structure visualization using ASE
        """
        try:
            # Convert to pymatgen structure to ASE atoms
            atoms = self.adaptor.get_atoms(structure)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save CIF file
            cif_file = os.path.join(output_dir, f"{material_id}.cif")
            structure.to(filename=cif_file)
            
            # Generate different views using ASE's write function
            views = [
                ("front", "0x,0y,0z"),
                ("top", "90x,0y,0z"),
                ("side", "0x,90y,0z")
            ]
            
            for view_name, rotation in views:
                try:
                    output_file = os.path.join(output_dir, f"{material_id}_{view_name}.png")
                    
                    # Use ASE's write function with rotation
                    write(output_file, atoms,
                          rotation=rotation,
                          show_unit_cell=2,
                          format='png')
                    
                    logger.info(f"Generated {view_name} view for {material_id}")
                except Exception as e:
                    logger.warning(f"Could not generate {view_name} view for {material_id}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to process structure {material_id}: {str(e)}")

def process_structures(structures_dict, output_dir, desc="structures"):
    """Helper function to process a batch of structures"""
    visualizer = StructureVisualizer()
    total = len(structures_dict)
    processed = 0
    failed = 0
    
    logger.info(f"Processing {total} {desc}...")
    
    for idx, (material_id, structure_dict) in enumerate(structures_dict.items(), 1):
        try:
            structure = visualizer.load_structure(structure_dict)
            visualizer.visualize_structure(
                structure,
                material_id,
                output_dir=output_dir
            )
            processed += 1
            
            # Progress update
            if idx % 10 == 0 or idx == total:
                logger.info(f"Processed {idx}/{total} {desc} ({(idx/total)*100:.1f}%) - "
                          f"Success: {processed}, Failed: {failed}")
                
        except Exception as e:
            logger.error(f"Error processing {material_id}: {str(e)}")
            failed += 1
            continue

def main():
    try:
        # Process Ti compounds
        with open("data/ti_compounds_structures.json", 'r') as f:
            ti_structures = json.load(f)
        process_structures(ti_structures, "structures/ti_compounds", "Ti compounds")
        
        # Process superconductors
        with open("data/superconductors_structures.json", 'r') as f:
            sc_structures = json.load(f)
        process_structures(sc_structures, "structures/superconductors", "superconductors")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 