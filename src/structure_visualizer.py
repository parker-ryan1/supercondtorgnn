"""
Structure visualization module for superconductor analysis.

This module provides functionality to visualize crystal structures using
various visualization methods, including ASE, Matplotlib, and export to
common 3D formats.
"""

import os
import json
import numpy as np
# Force matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.cm as cm
from ase.io import write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.vis.structure_vtk import StructureVis
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import time
from tqdm import tqdm
import shutil

# Try to import optional dependencies
try:
    from ase.visualize.plot import plot_atoms
    ASE_PLOT_AVAILABLE = True
except ImportError:
    ASE_PLOT_AVAILABLE = False

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

# Import configuration
try:
    from config import config
except ImportError:
    # Fallback if config is not available
    config = {
        "structures_dir": "structures",
        "visualization_dir": "visualization",
        "image_format": "png",
        "image_dpi": 300,
        "generate_3d_views": True
    }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructureVisualizer:
    """Visualizes crystal structures using various methods."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the structure visualizer.
        
        Args:
            output_dir: Directory to save visualizations (optional if set in config)
        """
        self.adaptor = AseAtomsAdaptor()
        
        # Set output directory
        self.output_dir = output_dir or config.get("visualization_dir", "visualization")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set image format and DPI
        self.image_format = config.get("image_format", "png")
        self.image_dpi = config.get("image_dpi", 300)
        
        # Set 3D views flag
        self.generate_3d_views = config.get("generate_3d_views", True)
        
        logger.info(f"StructureVisualizer initialized with output directory: {self.output_dir}")
    
    def load_structure(self, structure_dict: Dict[str, Any]) -> Structure:
        """
        Convert dictionary to pymatgen Structure.
        
        Args:
            structure_dict: Dictionary representation of a structure
            
        Returns:
            Pymatgen Structure object
        """
        return Structure.from_dict(structure_dict)
    
    def get_element_colors(self, structure: Structure) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Get a dictionary of element colors for visualization.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Dictionary mapping element symbols to RGBA colors
        """
        # Get unique elements in the structure
        elements = set(site.specie.symbol for site in structure)
        
        # Create a color map
        cmap = cm.get_cmap('tab10')
        colors = {}
        
        for i, element in enumerate(elements):
            # Use a color map to assign colors to elements
            color = cmap(i % 10)
            colors[element] = color
        
        return colors
    
    def visualize_structure(self, structure: Structure, material_id: str) -> Dict[str, str]:
        """
        Generate structure visualization using multiple methods.
        
        Args:
            structure: Pymatgen Structure object
            material_id: Identifier for the material
            
        Returns:
            Dictionary of generated file paths
        """
        output_files = {}
        
        try:
            # Create material-specific output directory
            material_dir = os.path.join(self.output_dir, material_id)
            os.makedirs(material_dir, exist_ok=True)
            
            # Convert to ASE atoms
            atoms = self.adaptor.get_atoms(structure)
            
            # Save CIF file
            cif_file = os.path.join(material_dir, f"{material_id}.cif")
            structure.to(filename=cif_file)
            output_files["cif"] = cif_file
            
            # Generate standard views using ASE
            views = self._generate_standard_views(atoms, material_id, material_dir)
            output_files.update(views)
            
            # Generate conventional cell view if symmetry is available
            try:
                symm_view = self._generate_conventional_cell_view(structure, material_id, material_dir)
                if symm_view:
                    output_files.update(symm_view)
            except Exception as e:
                logger.warning(f"Could not generate conventional cell view for {material_id}: {str(e)}")
            
            # Generate 3D visualization if enabled
            if self.generate_3d_views:
                try:
                    threejs_file = self._generate_3d_view(structure, material_id, material_dir)
                    if threejs_file:
                        output_files["3d"] = threejs_file
                except Exception as e:
                    logger.warning(f"Could not generate 3D view for {material_id}: {str(e)}")
            
            logger.info(f"Generated visualizations for {material_id}")
            return output_files
            
        except Exception as e:
            logger.error(f"Failed to process structure {material_id}: {str(e)}")
            return output_files
    
    def _generate_standard_views(self, atoms, material_id: str, output_dir: str) -> Dict[str, str]:
        """
        Generate standard views of the structure.
        
        Args:
            atoms: ASE Atoms object
            material_id: Identifier for the material
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary of generated file paths
        """
        views = {
            "front": "0x,0y,0z",
            "top": "90x,0y,0z",
            "side": "0x,90y,0z",
            "perspective": "30x,30y,30z"
        }
        
        output_files = {}
        
        for view_name, rotation in views.items():
            try:
                output_file = os.path.join(output_dir, f"{material_id}_{view_name}.{self.image_format}")
                
                # Use ASE's write function with rotation
                write(output_file, atoms,
                      rotation=rotation,
                      show_unit_cell=2,
                      format=self.image_format)
                
                output_files[view_name] = output_file
                
            except Exception as e:
                logger.warning(f"Could not generate {view_name} view for {material_id}: {str(e)}")
                continue
        
        # Generate matplotlib plot if available
        if ASE_PLOT_AVAILABLE:
            try:
                plot_file = os.path.join(output_dir, f"{material_id}_plot.{self.image_format}")
                
                fig, ax = plt.subplots(figsize=(8, 8))
                plot_atoms(atoms, ax, rotation='30x,30y,0z', show_unit_cell=2)
                plt.tight_layout()
                plt.savefig(plot_file, dpi=self.image_dpi)
                plt.close(fig)
                
                output_files["plot"] = plot_file
                
            except Exception as e:
                logger.warning(f"Could not generate matplotlib plot for {material_id}: {str(e)}")
        
        return output_files
    
    def _generate_conventional_cell_view(self, structure: Structure, material_id: str, output_dir: str) -> Dict[str, str]:
        """
        Generate view of the conventional unit cell.
        
        Args:
            structure: Pymatgen Structure object
            material_id: Identifier for the material
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary of generated file paths
        """
        output_files = {}
        
        try:
            # Get conventional structure
            analyzer = SpacegroupAnalyzer(structure)
            conv_structure = analyzer.get_conventional_standard_structure()
            
            # Convert to ASE atoms
            conv_atoms = self.adaptor.get_atoms(conv_structure)
            
            # Save CIF file
            conv_cif_file = os.path.join(output_dir, f"{material_id}_conventional.cif")
            conv_structure.to(filename=conv_cif_file)
            output_files["conventional_cif"] = conv_cif_file
            
            # Generate image
            conv_img_file = os.path.join(output_dir, f"{material_id}_conventional.{self.image_format}")
            write(conv_img_file, conv_atoms,
                  rotation="30x,30y,30z",
                  show_unit_cell=2,
                  format=self.image_format)
            
            output_files["conventional_image"] = conv_img_file
            
            # Get symmetry information
            symm_data = {
                "space_group_symbol": analyzer.get_space_group_symbol(),
                "space_group_number": analyzer.get_space_group_number(),
                "point_group": analyzer.get_point_group_symbol(),
                "crystal_system": analyzer.get_crystal_system(),
                "symmetry_dataset": str(analyzer.get_symmetry_dataset())
            }
            
            # Save symmetry information
            symm_file = os.path.join(output_dir, f"{material_id}_symmetry.json")
            with open(symm_file, 'w') as f:
                json.dump(symm_data, f, indent=2)
            
            output_files["symmetry_info"] = symm_file
            
            return output_files
            
        except Exception as e:
            logger.warning(f"Could not generate conventional cell view for {material_id}: {str(e)}")
            return {}
    
    def _generate_3d_view(self, structure: Structure, material_id: str, output_dir: str) -> Optional[str]:
        """
        Generate 3D interactive visualization.
        
        Args:
            structure: Pymatgen Structure object
            material_id: Identifier for the material
            output_dir: Directory to save visualizations
            
        Returns:
            Path to the generated file, or None if failed
        """
        if not PY3DMOL_AVAILABLE:
            return None
        
        try:
            # Create a py3Dmol view
            view = py3Dmol.view(width=800, height=600)
            
            # Convert structure to CIF and load it
            cif_file = os.path.join(output_dir, f"{material_id}.cif")
            if not os.path.exists(cif_file):
                structure.to(filename=cif_file)
            
            with open(cif_file, 'r') as f:
                cif_data = f.read()
            
            view.addModel(cif_data, 'cif')
            
            # Style the atoms and bonds
            view.setStyle({'sphere': {'radius': 0.3}, 'stick': {'radius': 0.15}})
            view.addUnitCell()
            view.zoomTo()
            
            # Save as HTML
            html_file = os.path.join(output_dir, f"{material_id}_3d.html")
            view.write_html(html_file)
            
            return html_file
            
        except Exception as e:
            logger.warning(f"Could not generate 3D view for {material_id}: {str(e)}")
            return None
    
    def generate_summary_page(self, material_ids: List[str], output_dir: Optional[str] = None) -> str:
        """
        Generate a summary HTML page with all visualizations.
        
        Args:
            material_ids: List of material IDs
            output_dir: Directory to save the summary page
            
        Returns:
            Path to the generated summary page
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        try:
            # Create HTML content
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Structure Visualization Summary</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .material { border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; }
                    .material-header { display: flex; justify-content: space-between; }
                    .views { display: flex; flex-wrap: wrap; margin-top: 10px; }
                    .view { margin: 10px; text-align: center; }
                    img { max-width: 300px; max-height: 300px; border: 1px solid #eee; }
                    h1, h2 { color: #333; }
                    a { color: #0066cc; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <h1>Structure Visualization Summary</h1>
                <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
            """
            
            # Add each material
            for material_id in material_ids:
                material_dir = os.path.join(self.output_dir, material_id)
                if not os.path.exists(material_dir):
                    continue
                
                html_content += f"""
                <div class="material">
                    <div class="material-header">
                        <h2>{material_id}</h2>
                        <div>
                            <a href="{material_id}/{material_id}.cif" download>Download CIF</a>
                """
                
                # Check for 3D view
                threejs_file = os.path.join(material_dir, f"{material_id}_3d.html")
                if os.path.exists(threejs_file):
                    html_content += f"""
                            | <a href="{material_id}/{material_id}_3d.html" target="_blank">View 3D</a>
                    """
                
                html_content += """
                        </div>
                    </div>
                    <div class="views">
                """
                
                # Add images
                view_types = ["front", "top", "side", "perspective", "plot"]
                for view_type in view_types:
                    img_file = os.path.join(material_dir, f"{material_id}_{view_type}.{self.image_format}")
                    if os.path.exists(img_file):
                        rel_path = f"{material_id}/{material_id}_{view_type}.{self.image_format}"
                        html_content += f"""
                        <div class="view">
                            <img src="{rel_path}" alt="{view_type} view">
                            <p>{view_type.capitalize()} View</p>
                        </div>
                        """
                
                # Check for conventional cell view
                conv_img_file = os.path.join(material_dir, f"{material_id}_conventional.{self.image_format}")
                if os.path.exists(conv_img_file):
                    rel_path = f"{material_id}/{material_id}_conventional.{self.image_format}"
                    html_content += f"""
                    <div class="view">
                        <img src="{rel_path}" alt="Conventional Cell">
                        <p>Conventional Cell</p>
                    </div>
                    """
                
                html_content += """
                    </div>
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            # Write HTML file
            summary_file = os.path.join(output_dir, "summary.html")
            with open(summary_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated summary page at {summary_file}")
            return summary_file
            
        except Exception as e:
            logger.error(f"Failed to generate summary page: {str(e)}")
            return ""

def process_structures(structures_dict: Dict[str, Dict], 
                      output_dir: Optional[str] = None, 
                      desc: str = "structures",
                      limit: Optional[int] = None) -> List[str]:
    """
    Process a batch of structures.
    
    Args:
        structures_dict: Dictionary mapping material IDs to structure dictionaries
        output_dir: Directory to save visualizations
        desc: Description for progress reporting
        limit: Maximum number of structures to process
        
    Returns:
        List of processed material IDs
    """
    visualizer = StructureVisualizer(output_dir=output_dir)
    
    # Limit the number of structures if specified
    if limit and limit > 0:
        material_ids = list(structures_dict.keys())[:limit]
    else:
        material_ids = list(structures_dict.keys())
    
    total = len(material_ids)
    processed = 0
    failed = 0
    processed_ids = []
    
    logger.info(f"Processing {total} {desc}...")
    
    for idx, material_id in enumerate(tqdm(material_ids, desc=f"Visualizing {desc}")):
        try:
            structure_dict = structures_dict[material_id]
            structure = visualizer.load_structure(structure_dict)
            
            visualizer.visualize_structure(
                structure,
                material_id
            )
            
            processed += 1
            processed_ids.append(material_id)
            
        except Exception as e:
            logger.error(f"Error processing {material_id}: {str(e)}")
            failed += 1
            continue
    
    # Generate summary page
    if processed_ids:
        visualizer.generate_summary_page(processed_ids)
    
    logger.info(f"Processed {processed}/{total} {desc} - Success: {processed}, Failed: {failed}")
    return processed_ids

def main():
    """Main function for command-line usage."""
    try:
        # Check if data directory exists
        data_dir = config.get("data_dir", "data")
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            return 1
        
        # Process Ti compounds
        ti_file = os.path.join(data_dir, "ti_compounds_structures.json")
        if os.path.exists(ti_file):
            with open(ti_file, 'r') as f:
                ti_structures = json.load(f)
            process_structures(
                ti_structures, 
                output_dir=os.path.join(config.get("visualization_dir", "visualization"), "ti_compounds"),
                desc="Ti compounds"
            )
        else:
            logger.warning(f"Ti compounds file not found: {ti_file}")
        
        # Process superconductors
        sc_file = os.path.join(data_dir, "superconductors_structures.json")
        if os.path.exists(sc_file):
            with open(sc_file, 'r') as f:
                sc_structures = json.load(f)
            process_structures(
                sc_structures,
                output_dir=os.path.join(config.get("visualization_dir", "visualization"), "superconductors"),
                desc="superconductors"
            )
        else:
            logger.warning(f"Superconductors file not found: {sc_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
