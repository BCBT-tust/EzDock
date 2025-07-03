import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EzDockEnvironment:
    """Environment setup and management for EzDock"""
    
    def __init__(self, work_dir: Optional[str] = None):
        """
        Initialize environment manager
        
        Args:
            work_dir: Working directory for EzDock operations
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd() / "ezdock_workspace"
        self.work_dir.mkdir(exist_ok=True)
        
        # Tool paths
        self.mgltools_path = "/usr/local/autodocktools/bin/pythonsh"
        self.prepare_receptor4_path = "/usr/local/autodocktools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
        self.prepare_ligand4_path = "/usr/local/autodocktools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
        self.vina_path = "vina"
        
        # Environment variables
        self.env_vars = {
            'PYTHONPATH': "/usr/local/autodocktools/MGLToolsPckgs",
            'PATH': '/usr/local/autodocktools/bin:' + os.environ.get('PATH', '')
        }
    
    def run_command(self, command: str, verbose: bool = False) -> Tuple[bool, str, str]:
        """Execute system command safely"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0 and verbose:
                logger.warning(f"Command failed: {command}")
                logger.warning(f"Error: {result.stderr}")
                
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {command}")
            return False, "", "Command timed out"
        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            return False, "", str(e)
    
    def check_tool_availability(self) -> dict:
        """Check availability of required tools"""
        tools_status = {}
        
        # Check Python
        tools_status['python'] = sys.version_info >= (3, 7)
        
        # Check AutoDock Vina
        success, _, _ = self.run_command("vina --version")
        tools_status['vina'] = success
        
        # Check MGLTools (if available)
        tools_status['mgltools'] = Path(self.mgltools_path).exists()
        
        # Check required Python packages
        required_packages = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib']
        for package in required_packages:
            try:
                __import__(package)
                tools_status[package] = True
            except ImportError:
                tools_status[package] = False
        
        return tools_status
    
    def setup_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            'input',
            'output', 
            'receptors',
            'ligands',
            'parameters',
            'results',
            'logs',
            'temp'
        ]
        
        for dir_name in directories:
            (self.work_dir / dir_name).mkdir(exist_ok=True)
    
    def get_work_dir(self) -> Path:
        """Get the working directory path"""
        return self.work_dir
    
    def set_environment(self) -> None:
        """Set up environment variables"""
        for key, value in self.env_vars.items():
            os.environ[key] = value
    
    def validate_installation(self) -> bool:
        """Validate the complete installation"""
        status = self.check_tool_availability()
        
        critical_tools = ['python', 'numpy', 'pandas', 'scipy', 'sklearn']
        missing_tools = [tool for tool in critical_tools if not status.get(tool, False)]
        
        if missing_tools:
            logger.error(f"Missing critical tools: {missing_tools}")
            return False
        
        logger.info("✅ EzDock environment validation successful!")
        return True
    
    def get_system_info(self) -> dict:
        """Get system information for diagnostics"""
        import platform
        
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'work_directory': str(self.work_dir),
            'tool_status': self.check_tool_availability()
        }
        
        return info


def setup_environment(work_dir: Optional[str] = None) -> EzDockEnvironment:
    """
    Quick setup function for EzDock environment
    
    Args:
        work_dir: Working directory path
        
    Returns:
        Configured EzDockEnvironment instance
    """
    logger.info("🧬 Setting up EzDock environment...")
    
    env = EzDockEnvironment(work_dir)
    env.setup_directories()
    env.set_environment()
    
    if env.validate_installation():
        logger.info("🎉 EzDock environment ready!")
    else:
        logger.warning("⚠️ Environment setup completed with warnings")
    
    return env


# Global environment instance
_env = None

def get_environment() -> EzDockEnvironment:
    """Get or create global environment instance"""
    global _env
    if _env is None:
        _env = setup_environment()
    return _env
