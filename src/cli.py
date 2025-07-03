"""
EzDock Command Line Interface
Provides command-line access to EzDock functionality
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from . import __version__
from .environment import setup_environment
from .receptor_processor import process_receptors
from .parameter_extractor import extract_parameters
from .docking_pipeline import run_docking_pipeline
from .enzyme_analyzer import analyze_enzyme_docking
from .parameter_optimizer import optimize_docking_parameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description='EzDock: Automated Molecular Docking and Enzyme Engineering Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process receptor files
  ezdock process-receptors *.pdb --output-dir ./processed

  # Extract docking parameters
  ezdock extract-parameters protein.pdb --output-dir ./params

  # Run complete docking pipeline
  ezdock run-docking --receptors ./receptors/ --ligands ./ligands/ --parameters ./params/

  # Analyze enzyme engineering results
  ezdock analyze-enzyme results.zip --receptor protein.pdb

  # Optimize docking parameters
  ezdock optimize-parameters --receptor protein.pdbqt --ligand ligand.pdbqt

For more information, visit: https://github.com/yourusername/ezdock
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'EzDock {__version__}'
    )
    
    parser.add_argument(
        '--work-dir',
        type=str,
        help='Working directory for EzDock operations (default: ./ezdock_workspace)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup subcommand
    setup_cmd = subparsers.add_parser('setup', help='Setup EzDock environment')
    setup_cmd.add_argument(
        '--check-only',
        action='store_true',
        help='Only check environment, do not setup'
    )
    
    # Process receptors subcommand
    process_cmd = subparsers.add_parser('process-receptors', help='Process receptor PDB files')
    process_cmd.add_argument(
        'files',
        nargs='+',
        help='Receptor PDB files to process'
    )
    process_cmd.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for processed files'
    )
    process_cmd.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    
    # Extract parameters subcommand
    extract_cmd = subparsers.add_parser('extract-parameters', help='Extract docking parameters')
    extract_cmd.add_argument(
        'files',
        nargs='+',
        help='PDB files or ZIP archives to process'
    )
    extract_cmd.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for parameter files'
    )
    extract_cmd.add_argument(
        '--padding',
        type=float,
        default=10.0,
        help='Padding around protein structure in Angstroms (default: 10.0)'
    )
    
    # Run docking subcommand
    docking_cmd = subparsers.add_parser('run-docking', help='Run parallel molecular docking')
    docking_cmd.add_argument(
        '--receptors',
        type=str,
        required=True,