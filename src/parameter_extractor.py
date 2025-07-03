import os
import re
import zipfile
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

from .environment import get_environment

logger = logging.getLogger(__name__)


class ParameterExtractor:
    """Extract docking parameters from protein structures"""
    
    def __init__(self, output_dir: Optional[str] = None, padding: float = 10.0):
        """
        Initialize parameter extractor
        
        Args:
            output_dir: Output directory for parameter files
            padding: Padding around protein structure (Angstroms)
        """
        self.env = get_environment()
        self.output_dir = Path(output_dir) if output_dir else self.env.get_work_dir() / "docking_parameters"
        self.output_dir.mkdir(exist_ok=True)
        self.padding = padding
    
    def extract_coordinates_from_pdb(self, pdb_file: str) -> Optional[np.ndarray]:
        """
        Extract atomic coordinates from PDB file
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            Array of coordinates or None if failed
        """
        coordinates = []

        try:
            with open(pdb_file, 'r') as file:
                for line in file:
                    if line.startswith(('ATOM', 'HETATM')):
                        try:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            coordinates.append([x, y, z])
                        except (ValueError, IndexError):
                            continue
                            
        except Exception as e:
            logger.error(f"Error reading {pdb_file}: {e}")
            return None

        if not coordinates:
            logger.error(f"No valid coordinates found in {pdb_file}")
            return None

        return np.array(coordinates)

    def calculate_docking_parameters(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate docking box center and size from coordinates
        
        Args:
            coordinates: Array of atomic coordinates
            
        Returns:
            Tuple of (center, size)
        """
        min_coords = np.min(coordinates, axis=0)
        max_coords = np.max(coordinates, axis=0)

        # Calculate center
        center = (min_coords + max_coords) / 2

        # Calculate size with padding
        size = (max_coords - min_coords) + self.padding

        return center, size

    def clean_filename(self, filename: str) -> str:
        """
        Clean filename by removing unwanted suffixes
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned filename
        """
        # Remove pattern like (1), (2), etc.
        cleaned = re.sub(r'\s*\(\d+\)', '', filename)
        # Remove _docking_params if present
        cleaned = cleaned.replace('_docking_params', '')
        return cleaned.strip()

    def process_pdb_file(self, pdb_file: str) -> Tuple[bool, str]:
        """
        Process single PDB file and generate parameter file
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            Tuple of (success, cleaned_filename)
        """
        original_filename = Path(pdb_file).stem
        cleaned_filename = self.clean_filename(original_filename)

        logger.info(f"🔄 Processing {Path(pdb_file).name}...")

        # Extract coordinates
        coordinates = self.extract_coordinates_from_pdb(pdb_file)
        if coordinates is None:
            return False, cleaned_filename

        # Calculate parameters
        center, size = self.calculate_docking_parameters(coordinates)

        # Write parameter file with cleaned filename
        param_file = self.output_dir / f"{cleaned_filename}.txt"

        try:
            with open(param_file, 'w') as f:
                f.write("# AutoDock Vina Docking Parameters\n")
                f.write(f"# Generated from: {Path(pdb_file).name}\n")
                f.write(f"# Atoms processed: {len(coordinates)}\n\n")

                f.write("# Docking box center coordinates\n")
                f.write(f"center_x = {center[0]:.3f}\n")
                f.write(f"center_y = {center[1]:.3f}\n")
                f.write(f"center_z = {center[2]:.3f}\n\n")

                f.write("# Docking box size\n")
                f.write(f"size_x = {size[0]:.3f}\n")
                f.write(f"size_y = {size[1]:.3f}\n")
                f.write(f"size_z = {size[2]:.3f}\n")

            logger.info(f"   ✅ Generated parameters ({len(coordinates)} atoms) -> {cleaned_filename}.txt")
            return True, cleaned_filename

        except Exception as e:
            logger.error(f"   ❌ Failed to write parameter file: {e}")
            return False, cleaned_filename

    def extract_pdb_from_zip(self, zip_file: str, extract_dir: str) -> List[str]:
        """
        Extract PDB files from ZIP archive
        
        Args:
            zip_file: Path to ZIP file
            extract_dir: Directory to extract to
            
        Returns:
            List of extracted PDB file paths
        """
        pdb_files = []

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for filename in zip_ref.namelist():
                    if filename.lower().endswith('.pdb'):
                        zip_ref.extract(filename, extract_dir)
                        pdb_files.append(os.path.join(extract_dir, filename))

            logger.info(f"   📦 Extracted {len(pdb_files)} PDB files from ZIP")
            return pdb_files

        except zipfile.BadZipFile:
            logger.error(f"   ❌ Invalid ZIP file: {zip_file}")
            return []
        except Exception as e:
            logger.error(f"   ❌ Error extracting ZIP: {e}")
            return []

    def create_download_package(self, archive_name: str = "ezdock_docking_parameters.zip") -> Optional[str]:
        """
        Create downloadable package of parameter files
        
        Args:
            archive_name: Name of the archive file
            
        Returns:
            Path to created archive or None if failed
        """
        param_files = list(self.output_dir.glob("*.txt"))

        if not param_files:
            return None

        logger.info(f"📦 Packaging {len(param_files)} parameter files...")

        try:
            import shutil
            archive_path = self.env.get_work_dir() / archive_name
            shutil.make_archive(str(archive_path.with_suffix('')), 'zip', self.output_dir)

            if archive_path.exists():
                return str(archive_path)
                
        except Exception as e:
            logger.error(f"❌ Packaging failed: {e}")

        return None

    def show_summary(self, successful_files: List[str], failed_files: List[str]) -> None:
        """Display processing summary"""
        logger.info("=" * 50)
        logger.info("📋 Parameter Generation Summary")
        logger.info("=" * 50)
        logger.info(f"✅ Successfully processed: {len(successful_files)} files")
        
        if len(failed_files) > 0:
            logger.info(f"❌ Failed: {len(failed_files)} files")

        if successful_files:
            logger.info("📄 Generated parameter files:")
            for filename in successful_files:
                logger.info(f"   • {filename}.txt")

        if failed_files:
            logger.warning("⚠️ Failed files:")
            for filename in failed_files:
                logger.warning(f"   • {filename}")

    def process_files(self, input_files: Dict[str, bytes]) -> Dict[str, Any]:
        """
        Process uploaded files and generate parameters
        
        Args:
            input_files: Dictionary of filename -> file_content
            
        Returns:
            Dictionary with processing results
        """
        # Setup temporary directory
        temp_dir = self.env.get_work_dir() / 'temp_extracts'
        temp_dir.mkdir(exist_ok=True)

        successful_files = []
        failed_files = []

        try:
            # Process uploaded files
            for filename, file_content in input_files.items():
                file_path = temp_dir / filename

                # Save uploaded file
                with open(file_path, 'wb') as f:
                    f.write(file_content)

                file_ext = Path(filename).suffix.lower()

                if file_ext == '.zip':
                    # Extract and process PDB files from ZIP
                    logger.info(f"📦 Extracting {filename}...")
                    pdb_files = self.extract_pdb_from_zip(str(file_path), str(temp_dir))

                    for pdb_file in pdb_files:
                        success, cleaned_name = self.process_pdb_file(pdb_file)
                        if success:
                            successful_files.append(cleaned_name)
                        else:
                            failed_files.append(cleaned_name)

                elif file_ext == '.pdb':
                    # Process single PDB file
                    success, cleaned_name = self.process_pdb_file(str(file_path))
                    if success:
                        successful_files.append(cleaned_name)
                    else:
                        failed_files.append(cleaned_name)

                else:
                    logger.warning(f"⚠️ Skipping {filename}: Not a PDB or ZIP file")
                    failed_files.append(Path(filename).stem)

            # Show summary
            self.show_summary(successful_files, failed_files)

            # Create download package
            package_file = None
            if successful_files:
                package_file = self.create_download_package()

            return {
                'successful_files': successful_files,
                'failed_files': failed_files,
                'package_path': package_file,
                'output_directory': str(self.output_dir)
            }

        finally:
            # Cleanup
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


def extract_parameters(input_files: Dict[str, bytes], 
                      output_dir: Optional[str] = None,
                      padding: float = 10.0) -> Dict[str, Any]:
    """
    Convenience function for parameter extraction
    
    Args:
        input_files: Dictionary of filename -> file_content
        output_dir: Output directory for parameter files
        padding: Padding around protein structure (Angstroms)
        
    Returns:
        Dictionary with processing results
    """
    extractor = ParameterExtractor(output_dir, padding)
    return extractor.process_files(input_files)
