"""
Enhanced crystal graph feature extraction for superconductor prediction.

This module provides specialized feature extraction for crystal structures,
focusing on features relevant to superconductivity prediction.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN, BrunnerNN_real
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrystalGraphFeatureExtractor:
    """
    Enhanced feature extractor for crystal graphs with superconductivity-relevant features.
    
    This class extracts node features (atomic properties), edge features (bond properties),
    and global features (crystal properties) from crystal structures.
    """
    
    def __init__(self, 
                 nn_strategy: str = "crystal", 
                 edge_features: bool = True,
                 add_self_loops: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            nn_strategy: Nearest neighbor strategy ('crystal', 'voronoi', or 'brunner')
            edge_features: Whether to include edge features
            add_self_loops: Whether to add self-loops to the graph
        """
        self.nn_strategy = nn_strategy
        self.edge_features = edge_features
        self.add_self_loops = add_self_loops
        
        # Initialize nearest neighbor finder based on strategy
        if nn_strategy == "crystal":
            self.nn_finder = CrystalNN(weighted_cn=True, distance_cutoffs=(0.5, 1.5))
        elif nn_strategy == "voronoi":
            self.nn_finder = VoronoiNN(weight="area")
        elif nn_strategy == "brunner":
            self.nn_finder = BrunnerNN_real()
        else:
            raise ValueError(f"Unknown nearest neighbor strategy: {nn_strategy}")
        
        # Load element properties
        self.element_properties = self._load_element_properties()
        
        logger.info(f"CrystalGraphFeatureExtractor initialized with {nn_strategy} strategy")
    
    def _load_element_properties(self) -> Dict[str, np.ndarray]:
        """
        Load comprehensive element properties for feature generation.
        
        Returns:
            Dictionary mapping element symbols to property vectors
        """
        # Properties relevant for superconductivity:
        # 1. Atomic number (Z)
        # 2. Atomic mass
        # 3. Electronegativity (Pauling)
        # 4. Covalent radius
        # 5. Valence electrons
        # 6. First ionization energy
        # 7. Electron affinity
        # 8. Block (s=1, p=2, d=3, f=4)
        # 9. Group
        # 10. Row
        # 11. Metallic character (0=non-metal, 1=metalloid, 2=metal)
        # 12. Atomic volume
        
        properties = {}
        
        # Get all elements from pymatgen
        for z in range(1, 119):
            try:
                el = Element.from_Z(z)
                
                # Get block as numeric value
                block_map = {"s": 1, "p": 2, "d": 3, "f": 4}
                block = block_map.get(el.block, 0)
                
                # Get metallic character
                if el.is_metal:
                    metallic = 2
                elif el.is_metalloid:
                    metallic = 1
                else:
                    metallic = 0
                
                # Create feature vector
                props = [
                    el.Z,  # Atomic number
                    el.atomic_mass,  # Atomic mass
                    el.X if el.X is not None else 0,  # Electronegativity
                    el.atomic_radius if el.atomic_radius is not None else 0,  # Atomic radius
                    el.nvalence() if hasattr(el, 'nvalence') else 0,  # Valence electrons
                    el.ionization_energy if el.ionization_energy is not None else 0,  # Ionization energy
                    el.electron_affinity if el.electron_affinity is not None else 0,  # Electron affinity
                    block,  # Block (s, p, d, f)
                    el.group if el.group is not None else 0,  # Group
                    el.row if el.row is not None else 0,  # Row
                    metallic,  # Metallic character
                    el.atomic_volume if el.atomic_volume is not None else 0  # Atomic volume
                ]
                
                properties[el.symbol] = np.array(props, dtype=np.float32)
                
            except Exception as e:
                logger.warning(f"Error loading properties for element Z={z}: {str(e)}")
        
        return properties
    
    def extract_node_features(self, structure: Structure) -> np.ndarray:
        """
        Extract comprehensive node features for each atom in the structure.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Array of node features with shape [num_atoms, num_features]
        """
        num_atoms = len(structure)
        num_features = 12  # Number of element properties
        features = np.zeros((num_atoms, num_features), dtype=np.float32)
        
        for i, site in enumerate(structure):
            element = site.specie.symbol
            if element in self.element_properties:
                features[i] = self.element_properties[element]
            else:
                logger.warning(f"Unknown element: {element}, using default features")
        
        return features
    
    def extract_edge_index_and_features(self, structure: Structure) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract edge indices and features from the crystal structure.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Tuple of (edge_index, edge_features)
        """
        num_atoms = len(structure)
        edges = []
        edge_attrs = []
        
        # Get bonds using the selected nearest neighbor finder
        for i in range(num_atoms):
            try:
                nn_info = self.nn_finder.get_nn_info(structure, i)
                
                for neighbor in nn_info:
                    j = neighbor["site_index"]
                    
                    # Skip self-loops if not desired
                    if i == j and not self.add_self_loops:
                        continue
                    
                    # Add edge
                    edges.append([i, j])
                    
                    # Extract edge features if enabled
                    if self.edge_features:
                        # Distance between atoms
                        distance = neighbor.get("weight", 0.0)
                        
                        # Vector between atoms
                        site_i = structure[i]
                        site_j = structure[j]
                        vec = structure.lattice.get_distance_and_image(site_i.frac_coords, site_j.frac_coords)[1]
                        vec_normalized = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
                        
                        # Edge features: [distance, x, y, z]
                        edge_attr = [distance] + list(vec_normalized)
                        edge_attrs.append(edge_attr)
            
            except Exception as e:
                logger.warning(f"Error getting neighbors for atom {i}: {str(e)}")
        
        # If no edges were found, create self-loops
        if not edges:
            logger.warning(f"No edges found in structure, creating self-loops")
            edges = [[i, i] for i in range(num_atoms)]
            if self.edge_features:
                edge_attrs = [[0.0, 0.0, 0.0, 0.0] for _ in range(num_atoms)]
        
        # Convert to numpy arrays
        edge_index = np.array(edges, dtype=np.int64).T
        
        if self.edge_features:
            edge_attr = np.array(edge_attrs, dtype=np.float32)
            return edge_index, edge_attr
        else:
            return edge_index, None
    
    def extract_global_features(self, structure: Structure) -> np.ndarray:
        """
        Extract comprehensive global features from the structure.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Array of global features
        """
        # Basic structural properties
        volume = structure.volume
        density = structure.density
        num_atoms = len(structure)
        
        # Composition features
        composition = structure.composition
        num_elements = len(composition.elements)
        
        # Element statistics
        atomic_numbers = [site.specie.Z for site in structure]
        mean_atomic_number = np.mean(atomic_numbers)
        std_atomic_number = np.std(atomic_numbers)
        
        # Lattice features
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        
        # Symmetry features
        try:
            spg_analyzer = SpacegroupAnalyzer(structure)
            crystal_system = spg_analyzer.get_crystal_system_int()
            spg_number = spg_analyzer.get_space_group_number()
            point_group_order = len(spg_analyzer.get_point_group_operations())
        except Exception:
            crystal_system = 0
            spg_number = 0
            point_group_order = 0
        
        # Pack all features
        features = np.array([
            volume,
            density,
            num_atoms,
            num_elements,
            mean_atomic_number,
            std_atomic_number,
            a, b, c,
            alpha, beta, gamma,
            crystal_system,
            spg_number,
            point_group_order
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
        
        # Extract edge indices and features
        edge_index, edge_attr = self.extract_edge_index_and_features(structure)
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
        
        # Add edge features if available
        if edge_attr is not None:
            data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return data
    
    def extract_superconductivity_features(self, structure: Structure) -> Dict[str, float]:
        """
        Extract features specifically relevant to superconductivity.
        
        Args:
            structure: Pymatgen Structure object
            
        Returns:
            Dictionary of superconductivity-relevant features
        """
        # Composition-based features
        composition = structure.composition
        
        # Average atomic mass (lighter elements often have higher Tc)
        avg_atomic_mass = composition.weight / composition.num_atoms
        
        # Average electronegativity (important for electron-phonon coupling)
        electronegativities = [self.element_properties.get(el.symbol, np.zeros(12))[2] 
                              for el in composition.elements]
        avg_electronegativity = np.mean(electronegativities) if electronegativities else 0
        
        # Valence electron count (important for carrier concentration)
        valence_counts = [self.element_properties.get(el.symbol, np.zeros(12))[4] * composition[el]
                         for el in composition.elements]
        total_valence = sum(valence_counts)
        valence_per_atom = total_valence / composition.num_atoms
        
        # Metallic character ratio (metals are more likely to be superconductors)
        metallic_values = [self.element_properties.get(el.symbol, np.zeros(12))[10] * composition[el]
                          for el in composition.elements]
        metallic_ratio = sum(metallic_values) / composition.num_atoms
        
        # Density of states proxy (based on valence and metallic character)
        dos_proxy = valence_per_atom * metallic_ratio
        
        # Structural features
        volume_per_atom = structure.volume / composition.num_atoms
        
        # Symmetry features (higher symmetry often correlates with superconductivity)
        try:
            spg_analyzer = SpacegroupAnalyzer(structure)
            crystal_system = spg_analyzer.get_crystal_system_int()
            spg_number = spg_analyzer.get_space_group_number()
            point_group_order = len(spg_analyzer.get_point_group_operations())
        except Exception:
            crystal_system = 0
            spg_number = 0
            point_group_order = 0
        
        # Connectivity features (higher connectivity often correlates with superconductivity)
        try:
            vconn = VoronoiConnectivity(structure)
            connectivity = np.mean([len(vconn.get_connections(i)) for i in range(len(structure))])
        except Exception:
            connectivity = 0
        
        # Return all features
        return {
            "avg_atomic_mass": avg_atomic_mass,
            "avg_electronegativity": avg_electronegativity,
            "valence_per_atom": valence_per_atom,
            "metallic_ratio": metallic_ratio,
            "dos_proxy": dos_proxy,
            "volume_per_atom": volume_per_atom,
            "crystal_system": crystal_system,
            "spg_number": spg_number,
            "point_group_order": point_group_order,
            "connectivity": connectivity
        }

def test_feature_extraction():
    """Test the feature extraction on a simple structure."""
    from pymatgen.core import Lattice, Structure
    
    # Create a simple test structure (TiO2)
    lattice = Lattice.cubic(4.0)
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]
    species = ["Ti", "Ti", "O", "O"]
    test_structure = Structure(lattice, species, coords)
    
    # Create feature extractor
    extractor = CrystalGraphFeatureExtractor(edge_features=True)
    
    # Extract features
    node_features = extractor.extract_node_features(test_structure)
    edge_index, edge_attr = extractor.extract_edge_index_and_features(test_structure)
    global_features = extractor.extract_global_features(test_structure)
    
    # Convert to graph
    graph = extractor.structure_to_graph(test_structure)
    
    # Extract superconductivity features
    sc_features = extractor.extract_superconductivity_features(test_structure)
    
    # Print results
    print(f"Node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    if edge_attr is not None:
        print(f"Edge features shape: {edge_attr.shape}")
    print(f"Global features shape: {global_features.shape}")
    print(f"Graph: {graph}")
    print(f"Superconductivity features: {sc_features}")
    
    return graph

if __name__ == "__main__":
    test_feature_extraction()
