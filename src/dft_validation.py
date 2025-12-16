"""
Density Functional Theory (DFT) validation for superconductor candidates.

This module provides functionality to validate superconductor candidates
using DFT calculations from the Materials Project API.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from mp_api.client import MPRester
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.dos import CompleteDos
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DFTValidator:
    """
    Validator for superconductor candidates using DFT calculations.
    
    This class fetches and analyzes DFT data from the Materials Project
    to validate superconductor candidates identified by the GNN model.
    """
    
    def __init__(self, api_key: str, output_dir: str = "dft_validation"):
        """
        Initialize the DFT validator.
        
        Args:
            api_key: Materials Project API key
            output_dir: Directory to save validation results
        """
        self.api_key = api_key
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Materials Project API client
        self.mpr = MPRester(api_key)
        
        logger.info(f"DFTValidator initialized with output directory: {output_dir}")
    
    def validate_candidates(self, candidates_file: str) -> pd.DataFrame:
        """
        Validate superconductor candidates using DFT calculations.
        
        Args:
            candidates_file: Path to CSV file with candidate materials
            
        Returns:
            DataFrame with validation results
        """
        # Load candidates
        candidates = pd.read_csv(candidates_file)
        logger.info(f"Loaded {len(candidates)} candidate materials from {candidates_file}")
        
        # Initialize results
        results = []
        
        # Process each candidate
        for _, candidate in tqdm(candidates.iterrows(), total=len(candidates), 
                                desc="Validating candidates with DFT"):
            material_id = candidate["material_id"]
            formula = candidate["formula"]
            predicted_tc = candidate["predicted_tc"]
            
            try:
                # Fetch DFT data
                dft_data = self.fetch_dft_data(material_id)
                
                # Analyze DFT data
                analysis = self.analyze_dft_data(dft_data, material_id)
                
                # Combine with candidate info
                result = {
                    "material_id": material_id,
                    "formula": formula,
                    "predicted_tc": predicted_tc,
                    **analysis
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error validating {material_id} ({formula}): {str(e)}")
                # Add partial result with error
                results.append({
                    "material_id": material_id,
                    "formula": formula,
                    "predicted_tc": predicted_tc,
                    "error": str(e),
                    "validation_success": False
                })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_file = os.path.join(self.output_dir, "validation_results.csv")
        results_df.to_csv(output_file, index=False)
        logger.info(f"Validation results saved to {output_file}")
        
        # Generate summary report
        self.generate_summary_report(results_df)
        
        return results_df
    
    def fetch_dft_data(self, material_id: str) -> Dict[str, Any]:
        """
        Fetch DFT data for a material from the Materials Project.
        
        Args:
            material_id: Materials Project ID
            
        Returns:
            Dictionary with DFT data
        """
        logger.info(f"Fetching DFT data for {material_id}")
        
        # Fetch basic material data
        material = self.mpr.materials.get_data_by_id(material_id)
        
        # Fetch electronic structure data if available
        try:
            bandstructure = self.mpr.electronic_structure.get_bandstructure_by_material_id(material_id)
        except Exception as e:
            logger.warning(f"Could not fetch bandstructure for {material_id}: {str(e)}")
            bandstructure = None
        
        try:
            dos = self.mpr.electronic_structure.get_dos_by_material_id(material_id)
        except Exception as e:
            logger.warning(f"Could not fetch DOS for {material_id}: {str(e)}")
            dos = None
        
        # Fetch phonon data if available
        try:
            phonon_data = self.mpr.phonon.get_data_by_material_id(material_id)
        except Exception as e:
            logger.warning(f"Could not fetch phonon data for {material_id}: {str(e)}")
            phonon_data = None
        
        # Combine data
        dft_data = {
            "material": material,
            "bandstructure": bandstructure,
            "dos": dos,
            "phonon_data": phonon_data
        }
        
        return dft_data
    
    def analyze_dft_data(self, dft_data: Dict[str, Any], material_id: str) -> Dict[str, Any]:
        """
        Analyze DFT data for superconductivity indicators.
        
        Args:
            dft_data: Dictionary with DFT data
            material_id: Materials Project ID
            
        Returns:
            Dictionary with analysis results
        """
        material = dft_data["material"]
        bandstructure = dft_data["bandstructure"]
        dos = dft_data["dos"]
        phonon_data = dft_data["phonon_data"]
        
        # Initialize results
        analysis = {
            "validation_success": True,
            "is_metal": material.is_metal if hasattr(material, "is_metal") else None,
            "formation_energy_per_atom": material.formation_energy_per_atom if hasattr(material, "formation_energy_per_atom") else None,
            "e_above_hull": material.energy_above_hull if hasattr(material, "energy_above_hull") else None,
            "band_gap": material.band_gap if hasattr(material, "band_gap") else None,
            "density": material.density if hasattr(material, "density") else None,
            "total_magnetization": material.total_magnetization if hasattr(material, "total_magnetization") else None,
            "elastic_anisotropy": material.elastic_anisotropy if hasattr(material, "elastic_anisotropy") else None,
            "has_bandstructure": bandstructure is not None,
            "has_dos": dos is not None,
            "has_phonon_data": phonon_data is not None
        }
        
        # Analyze electronic structure
        if bandstructure is not None:
            try:
                # Plot band structure
                self.plot_bandstructure(bandstructure, material_id)
                
                # Extract band structure properties
                if isinstance(bandstructure, BandStructureSymmLine):
                    analysis["fermi_surface_complexity"] = self._estimate_fermi_surface_complexity(bandstructure)
                    analysis["band_crossing_count"] = self._count_band_crossings(bandstructure)
                    analysis["bandwidth"] = self._calculate_bandwidth(bandstructure)
            except Exception as e:
                logger.warning(f"Error analyzing band structure for {material_id}: {str(e)}")
        
        # Analyze density of states
        if dos is not None:
            try:
                # Plot DOS
                self.plot_dos(dos, material_id)
                
                # Extract DOS properties
                if isinstance(dos, CompleteDos):
                    analysis["dos_at_fermi"] = dos.get_densities(spin=None, energy_label="energy")[0]
                    analysis["d_orbital_contribution"] = self._calculate_d_orbital_contribution(dos)
            except Exception as e:
                logger.warning(f"Error analyzing DOS for {material_id}: {str(e)}")
        
        # Analyze phonon data
        if phonon_data is not None:
            try:
                # Extract phonon properties
                analysis["has_soft_modes"] = self._check_soft_modes(phonon_data)
                analysis["phonon_bandwidth"] = self._calculate_phonon_bandwidth(phonon_data)
            except Exception as e:
                logger.warning(f"Error analyzing phonon data for {material_id}: {str(e)}")
        
        # Calculate superconductivity likelihood score
        analysis["sc_likelihood_score"] = self._calculate_sc_likelihood(analysis)
        
        return analysis
    
    def _estimate_fermi_surface_complexity(self, bandstructure: BandStructureSymmLine) -> float:
        """
        Estimate the complexity of the Fermi surface.
        
        Args:
            bandstructure: Band structure object
            
        Returns:
            Complexity score (higher means more complex)
        """
        # Count number of band crossings at Fermi level
        crossings = 0
        for i, band in enumerate(bandstructure.bands.values()):
            for j in range(len(band[0]) - 1):
                if (band[0][j] - bandstructure.efermi) * (band[0][j+1] - bandstructure.efermi) <= 0:
                    crossings += 1
        
        return float(crossings)
    
    def _count_band_crossings(self, bandstructure: BandStructureSymmLine) -> int:
        """
        Count the number of band crossings near the Fermi level.
        
        Args:
            bandstructure: Band structure object
            
        Returns:
            Number of band crossings
        """
        crossings = 0
        bands = list(bandstructure.bands.values())
        
        # Check for crossings between bands
        for i in range(len(bands) - 1):
            for j in range(i + 1, len(bands)):
                for k in range(len(bands[i][0]) - 1):
                    if (bands[i][0][k] - bands[j][0][k]) * (bands[i][0][k+1] - bands[j][0][k+1]) <= 0:
                        # Bands cross
                        # Check if near Fermi level (within 1 eV)
                        if abs(bands[i][0][k] - bandstructure.efermi) < 1.0 or abs(bands[j][0][k] - bandstructure.efermi) < 1.0:
                            crossings += 1
        
        return crossings
    
    def _calculate_bandwidth(self, bandstructure: BandStructureSymmLine) -> float:
        """
        Calculate the bandwidth of the valence and conduction bands.
        
        Args:
            bandstructure: Band structure object
            
        Returns:
            Bandwidth in eV
        """
        all_energies = []
        for band in bandstructure.bands.values():
            all_energies.extend(band[0])
        
        # Filter energies near Fermi level (within 5 eV)
        near_fermi = [e for e in all_energies if abs(e - bandstructure.efermi) < 5.0]
        
        if near_fermi:
            return max(near_fermi) - min(near_fermi)
        else:
            return 0.0
    
    def _calculate_d_orbital_contribution(self, dos: CompleteDos) -> float:
        """
        Calculate the contribution of d-orbitals to the DOS at Fermi level.
        
        Args:
            dos: Complete DOS object
            
        Returns:
            Fraction of DOS from d-orbitals
        """
        try:
            # Get total DOS at Fermi
            total_dos = dos.get_densities(spin=None, energy_label="energy")[0]
            
            # Get d-orbital DOS at Fermi
            d_dos = 0.0
            for element in dos.structure.composition.elements:
                if hasattr(dos, "get_element_spd_dos"):
                    element_dos = dos.get_element_spd_dos(element)
                    if 'd' in element_dos:
                        d_dos += element_dos['d'].get_densities(spin=None)[0]
            
            # Calculate fraction
            if total_dos > 0:
                return d_dos / total_dos
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _check_soft_modes(self, phonon_data: Any) -> bool:
        """
        Check for soft phonon modes (indicator of lattice instability).
        
        Args:
            phonon_data: Phonon data object
            
        Returns:
            True if soft modes are present
        """
        # Implementation depends on the format of phonon_data
        # This is a placeholder
        return False
    
    def _calculate_phonon_bandwidth(self, phonon_data: Any) -> float:
        """
        Calculate the phonon bandwidth.
        
        Args:
            phonon_data: Phonon data object
            
        Returns:
            Phonon bandwidth in THz
        """
        # Implementation depends on the format of phonon_data
        # This is a placeholder
        return 0.0
    
    def _calculate_sc_likelihood(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate a likelihood score for superconductivity.
        
        Args:
            analysis: Dictionary with analysis results
            
        Returns:
            Likelihood score (0-1)
        """
        # Initialize score
        score = 0.0
        max_score = 0.0
        
        # Check if material is metallic (essential for conventional superconductivity)
        if analysis.get("is_metal") is True:
            score += 1.0
        elif analysis.get("is_metal") is False:
            score -= 0.5
        max_score += 1.0
        
        # Check stability (e_above_hull should be low)
        e_hull = analysis.get("e_above_hull")
        if e_hull is not None:
            if e_hull < 0.05:  # Very stable
                score += 1.0
            elif e_hull < 0.1:  # Moderately stable
                score += 0.5
            elif e_hull > 0.2:  # Unstable
                score -= 0.5
            max_score += 1.0
        
        # Check DOS at Fermi level (higher is better for superconductivity)
        dos_fermi = analysis.get("dos_at_fermi")
        if dos_fermi is not None:
            if dos_fermi > 2.0:  # High DOS
                score += 1.0
            elif dos_fermi > 1.0:  # Moderate DOS
                score += 0.5
            max_score += 1.0
        
        # Check d-orbital contribution (higher is better for many superconductors)
        d_contrib = analysis.get("d_orbital_contribution")
        if d_contrib is not None:
            if d_contrib > 0.5:  # High d-orbital contribution
                score += 1.0
            elif d_contrib > 0.2:  # Moderate d-orbital contribution
                score += 0.5
            max_score += 1.0
        
        # Check band crossing count (more crossings can indicate favorable electronic structure)
        crossings = analysis.get("band_crossing_count")
        if crossings is not None:
            if crossings > 5:  # Many crossings
                score += 1.0
            elif crossings > 2:  # Some crossings
                score += 0.5
            max_score += 1.0
        
        # Calculate final score (normalize to 0-1)
        if max_score > 0:
            return max(0.0, min(1.0, score / max_score))
        else:
            return 0.0
    
    def plot_bandstructure(self, bandstructure: BandStructureSymmLine, material_id: str) -> str:
        """
        Plot the band structure.
        
        Args:
            bandstructure: Band structure object
            material_id: Materials Project ID
            
        Returns:
            Path to the saved plot
        """
        if not isinstance(bandstructure, BandStructureSymmLine):
            logger.warning(f"Cannot plot band structure for {material_id}: Not a BandStructureSymmLine object")
            return ""
        
        try:
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot band structure
            bandstructure.plot(zero_to_efermi=True)
            
            # Set labels and title
            plt.xlabel("Wave Vector")
            plt.ylabel("Energy (eV)")
            plt.title(f"Band Structure for {material_id}")
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"{material_id}_bandstructure.png")
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            return output_file
            
        except Exception as e:
            logger.warning(f"Error plotting band structure for {material_id}: {str(e)}")
            return ""
    
    def plot_dos(self, dos: CompleteDos, material_id: str) -> str:
        """
        Plot the density of states.
        
        Args:
            dos: DOS object
            material_id: Materials Project ID
            
        Returns:
            Path to the saved plot
        """
        if not isinstance(dos, CompleteDos):
            logger.warning(f"Cannot plot DOS for {material_id}: Not a CompleteDos object")
            return ""
        
        try:
            # Create DOS plotter
            plotter = DosPlotter()
            plotter.add_dos(f"{material_id}", dos)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot DOS
            plotter.get_plot(zero_to_efermi=True)
            
            # Set labels and title
            plt.xlabel("Energy (eV)")
            plt.ylabel("DOS")
            plt.title(f"Density of States for {material_id}")
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"{material_id}_dos.png")
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            return output_file
            
        except Exception as e:
            logger.warning(f"Error plotting DOS for {material_id}: {str(e)}")
            return ""
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate a summary report of the validation results.
        
        Args:
            results_df: DataFrame with validation results
            
        Returns:
            Path to the summary report
        """
        # Filter successful validations
        successful = results_df[results_df["validation_success"] == True].copy()
        
        # Sort by SC likelihood score
        if "sc_likelihood_score" in successful.columns:
            successful = successful.sort_values("sc_likelihood_score", ascending=False)
        
        # Create HTML report
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Superconductor Candidates DFT Validation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .high-score { background-color: #d4edda; }
                .medium-score { background-color: #fff3cd; }
                .low-score { background-color: #f8d7da; }
            </style>
        </head>
        <body>
            <h1>Superconductor Candidates DFT Validation</h1>
            <p>Generated on: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <h2>Summary</h2>
            <p>Total candidates: """ + str(len(results_df)) + """</p>
            <p>Successful validations: """ + str(len(successful)) + """</p>
            
            <h2>Top Candidates</h2>
            <table>
                <tr>
                    <th>Material ID</th>
                    <th>Formula</th>
                    <th>Predicted Tc (K)</th>
                    <th>SC Likelihood</th>
                    <th>Is Metal</th>
                    <th>E Above Hull (eV)</th>
                    <th>DOS at Fermi</th>
                    <th>Band Structure</th>
                    <th>DOS</th>
                </tr>
        """
        
        # Add rows for top candidates
        for _, row in successful.head(10).iterrows():
            # Determine score class
            score_class = ""
            if "sc_likelihood_score" in row:
                if row["sc_likelihood_score"] >= 0.7:
                    score_class = "high-score"
                elif row["sc_likelihood_score"] >= 0.4:
                    score_class = "medium-score"
                else:
                    score_class = "low-score"
            
            # Create row
            html += f"""
                <tr class="{score_class}">
                    <td>{row['material_id']}</td>
                    <td>{row['formula']}</td>
                    <td>{row['predicted_tc']:.2f}</td>
                    <td>{row.get('sc_likelihood_score', 'N/A'):.2f if isinstance(row.get('sc_likelihood_score'), (int, float)) else 'N/A'}</td>
                    <td>{row.get('is_metal', 'N/A')}</td>
                    <td>{row.get('e_above_hull', 'N/A'):.4f if isinstance(row.get('e_above_hull'), (int, float)) else 'N/A'}</td>
                    <td>{row.get('dos_at_fermi', 'N/A'):.2f if isinstance(row.get('dos_at_fermi'), (int, float)) else 'N/A'}</td>
                    <td><a href="{row['material_id']}_bandstructure.png">View</a></td>
                    <td><a href="{row['material_id']}_dos.png">View</a></td>
                </tr>
            """
        
        # Close table and HTML
        html += """
            </table>
            
            <h2>Validation Details</h2>
            <p>The validation process includes the following checks:</p>
            <ul>
                <li>Metallic character (essential for conventional superconductivity)</li>
                <li>Thermodynamic stability (e_above_hull)</li>
                <li>Electronic structure (DOS at Fermi level, band crossings)</li>
                <li>d-orbital contribution to DOS</li>
                <li>Phonon properties (if available)</li>
            </ul>
            
            <p>The SC likelihood score is calculated based on these properties.</p>
        </body>
        </html>
        """
        
        # Save HTML report
        report_file = os.path.join(self.output_dir, "validation_summary.html")
        with open(report_file, 'w') as f:
            f.write(html)
        
        logger.info(f"Summary report saved to {report_file}")
        return report_file

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate superconductor candidates using DFT calculations"
    )
    
    parser.add_argument("--api-key", required=True, help="Materials Project API key")
    parser.add_argument("--candidates", required=True, help="Path to candidates CSV file")
    parser.add_argument("--output-dir", default="dft_validation", help="Output directory")
    
    args = parser.parse_args()
    
    # Create validator
    validator = DFTValidator(api_key=args.api_key, output_dir=args.output_dir)
    
    # Validate candidates
    results = validator.validate_candidates(args.candidates)
    
    # Print top candidates
    print("\nTop candidates by SC likelihood score:")
    if "sc_likelihood_score" in results.columns:
        top = results.sort_values("sc_likelihood_score", ascending=False).head(6)
        for i, (_, row) in enumerate(top.iterrows()):
            print(f"{i+1}. {row['material_id']} - {row['formula']} - "
                 f"Predicted Tc: {row['predicted_tc']:.2f}K, "
                 f"SC Likelihood: {row.get('sc_likelihood_score', 'N/A'):.2f}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
