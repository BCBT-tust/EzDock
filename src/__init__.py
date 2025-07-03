"""
EzDock: Automated Molecular Docking and Enzyme Engineering Analysis Platform

A comprehensive toolkit for molecular docking, parameter optimization,
and enzyme engineering analysis with machine learning integration.
"""

__version__ = "1.0.0"
__author__ = "Chunru Zhou"
__email__ = "zhoucr2023@163.com"
__license__ = "MIT"

# Core modules
from .environment import setup_environment, get_environment
from .receptor_processor import process_receptors, ReceptorProcessor
from .parameter_extractor import extract_parameters, ParameterExtractor
from .docking_pipeline import run_docking_pipeline, EzDockPipeline
from .enzyme_analyzer import analyze_enzyme_docking, EnzymeDockingAnalyzer
from .parameter_optimizer import optimize_docking_parameters, DockingParameterOptimizer

# Convenience imports
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Environment management
    "setup_environment",
    "get_environment",
    
    # Receptor processing
    "process_receptors",
    "ReceptorProcessor",
    
    # Parameter extraction
    "extract_parameters", 
    "ParameterExtractor",
    
    # Docking pipeline
    "run_docking_pipeline",
    "EzDockPipeline",
    
    # Enzyme analysis
    "analyze_enzyme_docking",
    "EnzymeDockingAnalyzer",
    
    # Parameter optimization
    "optimize_docking_parameters",
    "DockingParameterOptimizer",
]


def get_version():
    """Return the current version string"""
    return __version__


def print_info():
    """Print package information"""
    print(f"""
EzDock {__version__}
Automated Molecular Docking and Enzyme Engineering Platform

Author: {__author__}
Email: {__email__}
License: {__license__}

Features:
• Batch receptor preprocessing
• Automated parameter extraction
• High-throughput parallel docking
• Machine learning-based enzyme analysis
• Bayesian parameter optimization
• Comprehensive reporting and visualization

For documentation and examples, visit:
https://github.com/yourusername/ezdock
    """.strip())


# Package-level configuration
import logging

# Configure default logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Set up package-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if no handlers exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)


def configure_logging(level=logging.INFO, format_string=None):
    """
    Configure package-wide logging
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger for the package
    package_logger = logging.getLogger(__name__)
    package_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in package_logger.handlers[:]:
        package_logger.removeHandler(handler)
    
    # Add new console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    
    package_logger.addHandler(console_handler)
    
    logger.info(f"EzDock logging configured at {logging.getLevelName(level)} level")


# Package startup message
logger.info(f"EzDock {__version__} initialized")

# Optional: Check for required dependencies on import
def _check_dependencies():
    """Check for required dependencies and warn if missing"""
    required_packages = [
        'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing optional dependencies: {', '.join(missing_packages)}")
        logger.warning("Some features may not be available. Install with: pip install ezdock[complete]")

# Run dependency check (optional)
try:
    _check_dependencies()
except Exception:
    pass  # Silently ignore errors during dependency checking