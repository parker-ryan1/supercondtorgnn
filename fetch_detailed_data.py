#!/usr/bin/env python3
"""
Fetch detailed material data from Materials Project API.
Enriches dataset with electronic structure, phonon, and other advanced properties.
"""

from mp_api.client import MPRester
import pandas as pd
import json
from pathlib import Path
import logging
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Materials Project API key
API_KEY = "DjKx0q7YivC5u73uKFIVPif813v7InYq"

def fetch_enhanced_properties(material_ids, api_key=API_KEY, max_materials=100):
    """
    Fetch comprehensive material properties from Materials Project
    
    Enhanced properties include:
    - Electronic structure (band structure, DOS)
    - Magnetic properties
    - Elastic properties  
    - Phonon properties (if available)
    - Detailed atomic site information
    - Oxidation states
    - Coordination numbers
    - Bond lengths and angles
    """
    
    logger.info(f"Fetching enhanced data for {len(material_ids)} materials...")
    
    enhanced_data = []
    
    with MPRester(api_key) as mpr:
        for i, mat_id in enumerate(tqdm(material_ids[:max_materials], desc="Fetching")):
            try:
                # Give API a break every 50 requests
                if i > 0 and i % 50 == 0:
                    logger.info(f"Processed {i} materials, taking a short break...")
                    time.sleep(2)
                
                material_data = {
                    'material_id': mat_id,
                    'success': True
                }
                
                # 1. Basic summary data
                try:
                    summary = mpr.materials.summary.search(material_ids=[mat_id])
                    if summary:
                        summary_doc = summary[0]
                        material_data.update({
                            'formula': summary_doc.formula_pretty,
                            'formation_energy_per_atom': summary_doc.formation_energy_per_atom,
                            'energy_above_hull': summary_doc.energy_above_hull,
                            'band_gap': summary_doc.band_gap,
                            'is_stable': summary_doc.is_stable,
                            'is_metal': summary_doc.is_metal if hasattr(summary_doc, 'is_metal') else (summary_doc.band_gap == 0),
                            'density': summary_doc.density,
                            'volume': summary_doc.volume,
                            'nsites': summary_doc.nsites,
                            'nelements': summary_doc.nelements,
                            'spacegroup_symbol': summary_doc.symmetry.symbol if hasattr(summary_doc, 'symmetry') else None,
                            'spacegroup_number': summary_doc.symmetry.number if hasattr(summary_doc, 'symmetry') else None,
                            'crystal_system': summary_doc.symmetry.crystal_system if hasattr(summary_doc, 'symmetry') else None,
                        })
                except Exception as e:
                    logger.debug(f"Summary data failed for {mat_id}: {e}")
                
                # 2. Electronic structure properties
                try:
                    electronic = mpr.materials.electronic_structure.search(material_ids=[mat_id])
                    if electronic:
                        elec_doc = electronic[0]
                        material_data.update({
                            'efermi': elec_doc.efermi if hasattr(elec_doc, 'efermi') else None,
                            'is_gap_direct': elec_doc.is_gap_direct if hasattr(elec_doc, 'is_gap_direct') else None,
                            'is_magnetic': elec_doc.is_magnetic if hasattr(elec_doc, 'is_magnetic') else False,
                            'total_magnetization': elec_doc.total_magnetization if hasattr(elec_doc, 'total_magnetization') else None,
                        })
                except Exception as e:
                    logger.debug(f"Electronic structure failed for {mat_id}: {e}")
                
                # 3. Magnetic properties (if magnetic)
                try:
                    if material_data.get('is_magnetic', False):
                        magnetism = mpr.materials.magnetism.search(material_ids=[mat_id])
                        if magnetism:
                            mag_doc = magnetism[0]
                            material_data.update({
                                'total_magnetization_per_cell': mag_doc.total_magnetization if hasattr(mag_doc, 'total_magnetization') else None,
                                'num_magnetic_sites': mag_doc.num_magnetic_sites if hasattr(mag_doc, 'num_magnetic_sites') else None,
                                'num_unique_magnetic_sites': mag_doc.num_unique_magnetic_sites if hasattr(mag_doc, 'num_unique_magnetic_sites') else None,
                            })
                except Exception as e:
                    logger.debug(f"Magnetism data failed for {mat_id}: {e}")
                
                # 4. Elastic properties
                try:
                    elastic = mpr.materials.elasticity.search(material_ids=[mat_id])
                    if elastic:
                        elast_doc = elastic[0]
                        material_data.update({
                            'bulk_modulus_vrh': elast_doc.k_vrh if hasattr(elast_doc, 'k_vrh') else None,
                            'shear_modulus_vrh': elast_doc.g_vrh if hasattr(elast_doc, 'g_vrh') else None,
                            'elastic_anisotropy': elast_doc.universal_anisotropy if hasattr(elast_doc, 'universal_anisotropy') else None,
                            'poisson_ratio': elast_doc.homogeneous_poisson if hasattr(elast_doc, 'homogeneous_poisson') else None,
                        })
                except Exception as e:
                    logger.debug(f"Elastic properties failed for {mat_id}: {e}")
                
                # 5. Phonon properties (if available)
                try:
                    phonon = mpr.materials.phonon.search(material_ids=[mat_id])
                    if phonon:
                        phon_doc = phonon[0]
                        material_data.update({
                            'has_phonon_data': True,
                            'phonon_bandstructure_available': hasattr(phon_doc, 'phonon_bandstructure'),
                        })
                except Exception as e:
                    logger.debug(f"Phonon data failed for {mat_id}: {e}")
                    material_data['has_phonon_data'] = False
                
                # 6. Oxidation states
                try:
                    oxidation = mpr.materials.oxidation_states.search(material_ids=[mat_id])
                    if oxidation:
                        ox_doc = oxidation[0]
                        if hasattr(ox_doc, 'possible_species'):
                            material_data['oxidation_states'] = str(ox_doc.possible_species)
                except Exception as e:
                    logger.debug(f"Oxidation states failed for {mat_id}: {e}")
                
                # 7. Detailed structure with coordination
                try:
                    structure = mpr.materials.summary.search(material_ids=[mat_id])[0].structure
                    
                    # Calculate average coordination number
                    from pymatgen.analysis.local_env import CrystalNN
                    try:
                        nn = CrystalNN()
                        coord_numbers = []
                        for i, site in enumerate(structure):
                            try:
                                cn = nn.get_cn(structure, i)
                                coord_numbers.append(cn)
                            except:
                                pass
                        
                        if coord_numbers:
                            material_data.update({
                                'avg_coordination_number': sum(coord_numbers) / len(coord_numbers),
                                'min_coordination_number': min(coord_numbers),
                                'max_coordination_number': max(coord_numbers),
                            })
                    except Exception as e:
                        logger.debug(f"Coordination calculation failed for {mat_id}: {e}")
                    
                    # Bond length statistics
                    try:
                        from scipy.spatial.distance import pdist
                        import numpy as np
                        
                        coords = np.array([site.coords for site in structure])
                        if len(coords) > 1:
                            distances = pdist(coords)
                            # Filter out very long distances (non-bonding)
                            bond_distances = distances[distances < 5.0]  # Assume bonds < 5 Angstrom
                            if len(bond_distances) > 0:
                                material_data.update({
                                    'avg_bond_length': float(np.mean(bond_distances)),
                                    'min_bond_length': float(np.min(bond_distances)),
                                    'max_bond_length': float(np.max(bond_distances)),
                                    'bond_length_std': float(np.std(bond_distances)),
                                })
                    except Exception as e:
                        logger.debug(f"Bond length calculation failed for {mat_id}: {e}")
                    
                    # Element diversity
                    composition = structure.composition
                    elements = list(composition.keys())
                    
                    # Count transition metals, rare earths, etc.
                    transition_metals = sum(1 for el in elements if 21 <= el.Z <= 30 or 39 <= el.Z <= 48 or 72 <= el.Z <= 80)
                    rare_earths = sum(1 for el in elements if 57 <= el.Z <= 71 or 89 <= el.Z <= 103)
                    alkali = sum(1 for el in elements if el.Z in [3, 11, 19, 37, 55, 87])
                    alkaline_earth = sum(1 for el in elements if el.Z in [4, 12, 20, 38, 56, 88])
                    noble_metals = sum(1 for el in elements if el.Z in [44, 45, 46, 76, 77, 78])  # Ru, Rh, Pd, Os, Ir, Pt
                    
                    material_data.update({
                        'num_transition_metals': transition_metals,
                        'num_rare_earths': rare_earths,
                        'num_alkali': alkali,
                        'num_alkaline_earth': alkaline_earth,
                        'num_noble_metals': noble_metals,
                    })
                    
                    # Electronegativity statistics
                    electronegativities = [el.X for el in elements if el.X is not None]
                    if electronegativities:
                        material_data.update({
                            'avg_electronegativity': np.mean(electronegativities),
                            'electronegativity_variance': np.var(electronegativities),
                            'electronegativity_range': max(electronegativities) - min(electronegativities),
                        })
                    
                    # Atomic mass statistics
                    atomic_masses = [el.atomic_mass for el in elements]
                    material_data.update({
                        'avg_atomic_mass': np.mean(atomic_masses),
                        'total_atomic_mass': sum(atomic_masses),
                        'atomic_mass_variance': np.var(atomic_masses),
                    })
                    
                except Exception as e:
                    logger.debug(f"Structure analysis failed for {mat_id}: {e}")
                
                # 8. Theoretical data (if available)
                try:
                    theoretical = mpr.materials.thermo.search(material_ids=[mat_id])
                    if theoretical:
                        thermo_doc = theoretical[0]
                        material_data.update({
                            'debye_temperature': thermo_doc.debye_temperature if hasattr(thermo_doc, 'debye_temperature') else None,
                        })
                except Exception as e:
                    logger.debug(f"Thermo data failed for {mat_id}: {e}")
                
                enhanced_data.append(material_data)
                
            except Exception as e:
                logger.warning(f"Failed to fetch data for {mat_id}: {e}")
                enhanced_data.append({
                    'material_id': mat_id,
                    'success': False,
                    'error': str(e)
                })
    
    logger.info(f"✅ Successfully fetched data for {len([d for d in enhanced_data if d.get('success', False)])} materials")
    return enhanced_data

def save_enhanced_data(enhanced_data, output_dir='data'):
    """Save enhanced data to files"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(enhanced_data)
    
    # Save to CSV
    csv_file = Path(output_dir) / 'enhanced_superconductors.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"✅ Saved enhanced data to {csv_file}")
    
    # Save to JSON (preserves more detail)
    json_file = Path(output_dir) / 'enhanced_superconductors.json'
    with open(json_file, 'w') as f:
        json.dump(enhanced_data, f, indent=2, default=str)
    logger.info(f"✅ Saved enhanced data to {json_file}")
    
    # Print statistics
    logger.info("\n" + "="*70)
    logger.info("ENHANCED DATA STATISTICS")
    logger.info("="*70)
    
    successful = df[df['success'] == True]
    logger.info(f"Total materials: {len(df)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(df) - len(successful)}")
    
    # Count available properties
    property_counts = {}
    for col in df.columns:
        if col not in ['material_id', 'success', 'error']:
            non_null = df[col].notna().sum()
            if non_null > 0:
                property_counts[col] = non_null
    
    logger.info(f"\nProperties available:")
    for prop, count in sorted(property_counts.items(), key=lambda x: -x[1])[:20]:
        logger.info(f"  {prop:35s}: {count:4d} / {len(df)} ({count/len(df)*100:5.1f}%)")
    
    logger.info("="*70)
    
    return df

def main():
    logger.info("="*70)
    logger.info("FETCH ENHANCED MATERIAL DATA FROM MATERIALS PROJECT")
    logger.info("="*70)
    
    # Load existing material IDs
    csv_file = Path('data/superconductors.csv')
    
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        logger.info("Please ensure data/superconductors.csv exists")
        return
    
    df = pd.read_csv(csv_file)
    material_ids = df['material_id'].tolist()
    
    logger.info(f"Loaded {len(material_ids)} material IDs from existing data")
    
    # Fetch enhanced data (start with first 100 for testing)
    max_materials = 100  # Change to len(material_ids) for all
    
    logger.info(f"\nFetching detailed data for first {max_materials} materials...")
    logger.info("This may take a while (~2-5 seconds per material)")
    logger.info(f"Estimated time: {max_materials * 3 / 60:.1f} minutes\n")
    
    enhanced_data = fetch_enhanced_properties(material_ids, max_materials=max_materials)
    
    # Save results
    df_enhanced = save_enhanced_data(enhanced_data)
    
    logger.info("\n✅ COMPLETE!")
    logger.info(f"\nEnhanced data saved to:")
    logger.info(f"  - data/enhanced_superconductors.csv")
    logger.info(f"  - data/enhanced_superconductors.json")
    
    logger.info(f"\n📊 New features added:")
    logger.info(f"  • Electronic structure (Fermi energy, band gap type)")
    logger.info(f"  • Magnetic properties (magnetization, magnetic sites)")
    logger.info(f"  • Elastic properties (bulk/shear modulus, Poisson ratio)")
    logger.info(f"  • Phonon data availability")
    logger.info(f"  • Coordination numbers (avg, min, max)")
    logger.info(f"  • Bond lengths (avg, min, max, std)")
    logger.info(f"  • Element counts (transition metals, rare earths, etc.)")
    logger.info(f"  • Electronegativity statistics")
    logger.info(f"  • Atomic mass statistics")
    logger.info(f"  • Debye temperature")
    
    logger.info(f"\n🎯 Next steps:")
    logger.info(f"  1. Review data/enhanced_superconductors.csv")
    logger.info(f"  2. Update training script to use enhanced features")
    logger.info(f"  3. Re-train model with richer data")
    logger.info(f"  4. Compare performance improvements")
    
    # Show sample of enhanced data
    logger.info(f"\n📋 Sample of enhanced data (first 3 materials):")
    for i, row in df_enhanced.head(3).iterrows():
        logger.info(f"\n{i+1}. {row.get('material_id', 'N/A')} - {row.get('formula', 'N/A')}")
        logger.info(f"   Band gap: {row.get('band_gap', 'N/A')} eV")
        logger.info(f"   Fermi energy: {row.get('efermi', 'N/A')} eV")
        logger.info(f"   Avg coordination: {row.get('avg_coordination_number', 'N/A'):.2f}" if pd.notna(row.get('avg_coordination_number')) else "   Avg coordination: N/A")
        logger.info(f"   Avg bond length: {row.get('avg_bond_length', 'N/A'):.3f} Å" if pd.notna(row.get('avg_bond_length')) else "   Avg bond length: N/A")
        logger.info(f"   Transition metals: {row.get('num_transition_metals', 0)}")
        logger.info(f"   Noble metals: {row.get('num_noble_metals', 0)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nData fetching interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


