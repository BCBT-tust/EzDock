"""
Receptor Preprocessing Module
Handles batch processing of receptor files for molecular docking
"""

import os
import subprocess
import threading
import concurrent.futures
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

from .environment import get_environment

logger = logging.getLogger(__name__)


class ProcessingProgress:
    """Thread-safe progress tracker"""
    
    def __init__(self):
        self.completed = 0
        self.total = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, success: bool = True):
        """Update progress counter"""
        with self.lock:
            self.completed += 1
            if not success:
                self.failed += 1

            if self.total > 0:
                progress = (self.completed / self.total) * 100
                elapsed = time.time() - self.start_time
                logger.info(f"Progress: {self.completed}/{self.total} ({progress:.0f}%) | "
                           f"✅ {self.completed - self.failed} success | "
                           f"❌ {self.failed} failed | "
                           f"⏱️ {elapsed:.1f}s")


class ReceptorProcessor:
    """Batch receptor preprocessing for AutoDock"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize receptor processor
        
        Args:
            output_dir: Output directory for processed files
        """
        self.env = get_environment()
        self.output_dir = Path(output_dir) if output_dir else self.env.get_work_dir() / "processed_receptors"
        self.output_dir.mkdir(exist_ok=True)
        
        self.progress = ProcessingProgress()
    
    def process_single_receptor(self, receptor_file: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Process a single receptor file through the AutoDock pipeline
        
        Args:
            receptor_file: Path to input PDB file
            
        Returns:
            Tuple of (output_path, error_message)
        """
        try:
            filename = Path(receptor_file).stem
            output_path = self.output_dir / f"{filename}.pdbqt"

            # AutoDock receptor preparation command
            cmd = [
                self.env.mgltools_path, 
                self.env.prepare_receptor4_path,
                "-r", receptor_file,
                "-o", str(output_path),
                "-A", "hydrogens",  # Add hydrogens
                "-U", "nphs_lps_waters"  # Remove non-polar hydrogens, lone pairs, waters
            ]

            # Set environment variables
            env = os.environ.copy()
            env.update(self.env.env_vars)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout per file
            )

            if result.returncode == 0 and output_path.exists():
                self.progress.update(success=True)
                return str(output_path), None
            else:
                error_msg = result.stderr.strip() or "Unknown processing error"
                self.progress.update(success=False)
                return None, error_msg

        except subprocess.TimeoutExpired:
            self.progress.update(success=False)
            return None, "Processing timeout (>5 minutes)"
        except Exception as e:
            self.progress.update(success=False)
            return None, str(e)

    def batch_process_receptors(self, receptor_files: List[str], 
                               max_workers: int = 4) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Process multiple receptor files in parallel
        
        Args:
            receptor_files: List of receptor file paths
            max_workers: Maximum number of parallel workers
            
        Returns:
            Tuple of (successful_files, failed_files_with_errors)
        """
        if not receptor_files:
            logger.error("No receptor files provided")
            return [], []

        self.progress.total = len(receptor_files)
        self.progress.start_time = time.time()

        logger.info(f"🚀 Processing {len(receptor_files)} receptor files...")
        logger.info("⚙️ Adding hydrogens, charges, and removing waters...")

        successful_files = []
        failed_files = []

        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_receptor, filename): filename
                for filename in receptor_files
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    output_path, error = future.result()
                    if output_path:
                        successful_files.append(output_path)
                    else:
                        failed_files.append((filename, error))
                except Exception as e:
                    failed_files.append((filename, f"Unexpected error: {str(e)}"))

        return successful_files, failed_files

    def create_output_archive(self, successful_files: List[str], 
                             archive_name: str = "ezdock_receptors.zip") -> Optional[str]:
        """
        Create downloadable archive of processed files
        
        Args:
            successful_files: List of successfully processed files
            archive_name: Name of the output archive
            
        Returns:
            Path to created archive or None if failed
        """
        if not successful_files:
            return None

        import zipfile
        
        archive_path = self.env.get_work_dir() / archive_name
        
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in successful_files:
                    zipf.write(file_path, Path(file_path).name)
            
            logger.info(f"📦 Created archive: {archive_path}")
            return str(archive_path)
            
        except Exception as e:
            logger.error(f"Failed to create archive: {str(e)}")
            return None

    def show_summary(self, successful_files: List[str], 
                    failed_files: List[Tuple[str, str]]) -> None:
        """Display processing summary"""
        total_time = time.time() - self.progress.start_time

        logger.info("=" * 50)
        logger.info("📋 Processing Summary")
        logger.info("=" * 50)
        logger.info(f"✅ Successfully processed: {len(successful_files)} files")
        logger.info(f"❌ Failed: {len(failed_files)} files")
        logger.info(f"⏱️ Total time: {total_time:.1f} seconds")

        if failed_files:
            logger.warning("⚠️ Failed Files:")
            for filename, error in failed_files[:5]:  # Show first 5 failures
                logger.warning(f"   • {filename}: {error}")
            if len(failed_files) > 5:
                logger.warning(f"   ... and {len(failed_files) - 5} more")


def process_receptors(receptor_files: List[str], 
                     output_dir: Optional[str] = None,
                     max_workers: int = 4) -> Dict[str, Any]:
    """
    Convenience function for batch receptor processing
    
    Args:
        receptor_files: List of receptor file paths
        output_dir: Output directory for processed files
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with processing results
    """
    processor = ReceptorProcessor(output_dir)
    
    successful_files, failed_files = processor.batch_process_receptors(
        receptor_files, max_workers
    )
    
    processor.show_summary(successful_files, failed_files)
    
    # Create output archive
    archive_path = processor.create_output_archive(successful_files)
    
    return {
        'successful_files': successful_files,
        'failed_files': failed_files,
        'archive_path': archive_path,
        'processor': processor
    }