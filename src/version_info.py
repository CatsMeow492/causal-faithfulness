"""
Version information and tracking utilities.
Provides comprehensive version information for reproducibility.
"""

import sys
import platform
import subprocess
from typing import Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime


# Framework version
__version__ = "1.0.0"
__version_info__ = (1, 0, 0)


def get_framework_version() -> str:
    """Get the framework version."""
    return __version__


def get_framework_version_info() -> tuple:
    """Get the framework version as a tuple."""
    return __version_info__


def get_git_info() -> Dict[str, Optional[str]]:
    """Get git repository information if available."""
    git_info = {
        'commit_hash': None,
        'branch': None,
        'is_dirty': None,
        'remote_url': None
    }
    
    try:
        # Get current directory (should be project root)
        repo_path = Path(__file__).parent.parent
        
        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info['commit_hash'] = result.stdout.strip()
        
        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()
        
        # Check if repository is dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info['is_dirty'] = len(result.stdout.strip()) > 0
        
        # Get remote URL
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info['remote_url'] = result.stdout.strip()
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        # Git not available or not a git repository
        pass
    
    return git_info


def get_python_info() -> Dict[str, str]:
    """Get Python interpreter information."""
    return {
        'version': sys.version,
        'version_info': '.'.join(map(str, sys.version_info[:3])),
        'implementation': platform.python_implementation(),
        'compiler': platform.python_compiler(),
        'executable': sys.executable,
        'platform': platform.platform(),
        'architecture': platform.architecture()[0]
    }


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = {}
    
    # Core packages with version attributes
    package_modules = {
        'torch': 'torch',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'transformers': 'transformers',
        'datasets': 'datasets'
    }
    
    for package_name, module_name in package_modules.items():
        try:
            module = __import__(module_name)
            if hasattr(module, '__version__'):
                packages[package_name] = module.__version__
            elif hasattr(module, 'version'):
                packages[package_name] = str(module.version)
            else:
                packages[package_name] = 'unknown'
        except ImportError:
            packages[package_name] = 'not_installed'
    
    # Special handling for optional packages
    optional_packages = {
        'shap': 'shap',
        'lime': 'lime',
        'captum': 'captum'
    }
    
    for package_name, module_name in optional_packages.items():
        try:
            module = __import__(module_name)
            if hasattr(module, '__version__'):
                packages[package_name] = module.__version__
            else:
                packages[package_name] = 'unknown'
        except ImportError:
            packages[package_name] = 'not_installed'
    
    return packages


def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information."""
    hardware_info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor()
    }
    
    # CPU information
    try:
        import psutil
        hardware_info['cpu_count_physical'] = psutil.cpu_count(logical=False)
        hardware_info['cpu_count_logical'] = psutil.cpu_count(logical=True)
        hardware_info['memory_total_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        hardware_info['cpu_count_logical'] = None
        hardware_info['memory_total_gb'] = None
    
    # GPU information
    try:
        import torch
        if torch.cuda.is_available():
            hardware_info['cuda_available'] = True
            hardware_info['cuda_version'] = torch.version.cuda
            hardware_info['cudnn_version'] = torch.backends.cudnn.version()
            hardware_info['gpu_count'] = torch.cuda.device_count()
            
            # GPU details
            gpu_details = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_details.append({
                    'name': props.name,
                    'memory_total_gb': round(props.total_memory / (1024**3), 2),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multiprocessor_count': props.multi_processor_count
                })
            hardware_info['gpu_details'] = gpu_details
        else:
            hardware_info['cuda_available'] = False
        
        # MPS information (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            hardware_info['mps_available'] = torch.backends.mps.is_available()
            hardware_info['mps_built'] = torch.backends.mps.is_built()
        
    except ImportError:
        hardware_info['cuda_available'] = None
        hardware_info['mps_available'] = None
    
    return hardware_info


def get_environment_info() -> Dict[str, Any]:
    """Get environment information."""
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'framework_version': get_framework_version(),
        'python_info': get_python_info(),
        'package_versions': get_package_versions(),
        'hardware_info': get_hardware_info(),
        'git_info': get_git_info()
    }
    
    return env_info


def save_version_info(filepath: str):
    """Save complete version information to a file."""
    version_info = get_environment_info()
    
    with open(filepath, 'w') as f:
        json.dump(version_info, f, indent=2, default=str)


def load_version_info(filepath: str) -> Dict[str, Any]:
    """Load version information from a file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_versions(current_info: Dict[str, Any], reference_info: Dict[str, Any]) -> Dict[str, bool]:
    """Compare current version info with reference."""
    comparison = {}
    
    # Compare framework version
    comparison['framework_version_match'] = (
        current_info.get('framework_version') == reference_info.get('framework_version')
    )
    
    # Compare Python version
    current_python = current_info.get('python_info', {}).get('version_info')
    reference_python = reference_info.get('python_info', {}).get('version_info')
    comparison['python_version_match'] = current_python == reference_python
    
    # Compare key package versions
    current_packages = current_info.get('package_versions', {})
    reference_packages = reference_info.get('package_versions', {})
    
    key_packages = ['torch', 'numpy', 'scipy', 'scikit-learn', 'transformers']
    package_matches = []
    
    for package in key_packages:
        current_version = current_packages.get(package)
        reference_version = reference_packages.get(package)
        
        if current_version and reference_version:
            # Compare major.minor versions (ignore patch)
            current_parts = current_version.split('.')[:2]
            reference_parts = reference_version.split('.')[:2]
            package_matches.append(current_parts == reference_parts)
        else:
            package_matches.append(current_version == reference_version)
    
    comparison['key_packages_match'] = all(package_matches)
    
    # Compare git commit (if available)
    current_commit = current_info.get('git_info', {}).get('commit_hash')
    reference_commit = reference_info.get('git_info', {}).get('commit_hash')
    
    if current_commit and reference_commit:
        comparison['git_commit_match'] = current_commit == reference_commit
    else:
        comparison['git_commit_match'] = None
    
    return comparison


def print_version_info():
    """Print formatted version information."""
    info = get_environment_info()
    
    print("=== Version Information ===")
    print(f"Framework Version: {info['framework_version']}")
    print(f"Timestamp: {info['timestamp']}")
    
    print(f"\nPython Information:")
    python_info = info['python_info']
    print(f"  Version: {python_info['version_info']}")
    print(f"  Implementation: {python_info['implementation']}")
    print(f"  Platform: {python_info['platform']}")
    print(f"  Architecture: {python_info['architecture']}")
    
    print(f"\nPackage Versions:")
    for package, version in sorted(info['package_versions'].items()):
        print(f"  {package}: {version}")
    
    print(f"\nHardware Information:")
    hardware = info['hardware_info']
    print(f"  Platform: {hardware['platform']} {hardware['platform_release']}")
    print(f"  Architecture: {hardware['architecture']}")
    print(f"  CPU Count: {hardware.get('cpu_count_logical', 'unknown')}")
    print(f"  Memory: {hardware.get('memory_total_gb', 'unknown')} GB")
    
    if hardware.get('cuda_available'):
        print(f"  CUDA: Available (version {hardware.get('cuda_version')})")
        print(f"  GPU Count: {hardware.get('gpu_count')}")
        for i, gpu in enumerate(hardware.get('gpu_details', [])):
            print(f"    GPU {i}: {gpu['name']} ({gpu['memory_total_gb']} GB)")
    else:
        print(f"  CUDA: Not available")
    
    if hardware.get('mps_available'):
        print(f"  MPS: Available")
    elif hardware.get('mps_available') is False:
        print(f"  MPS: Not available")
    
    git_info = info['git_info']
    if git_info.get('commit_hash'):
        print(f"\nGit Information:")
        print(f"  Commit: {git_info['commit_hash'][:8]}...")
        print(f"  Branch: {git_info.get('branch', 'unknown')}")
        print(f"  Dirty: {git_info.get('is_dirty', 'unknown')}")
        if git_info.get('remote_url'):
            print(f"  Remote: {git_info['remote_url']}")


def check_version_compatibility() -> Dict[str, bool]:
    """Check if current environment meets minimum requirements."""
    compatibility = {}
    
    # Check Python version
    python_version = sys.version_info
    compatibility['python_version_ok'] = python_version >= (3, 10)
    
    # Check key packages
    packages = get_package_versions()
    
    # PyTorch version check
    torch_version = packages.get('torch', '0.0.0')
    if torch_version != 'not_installed':
        try:
            torch_parts = [int(x) for x in torch_version.split('.')[:2]]
            compatibility['torch_version_ok'] = torch_parts >= [2, 1]
        except (ValueError, IndexError):
            compatibility['torch_version_ok'] = False
    else:
        compatibility['torch_version_ok'] = False
    
    # NumPy version check
    numpy_version = packages.get('numpy', '0.0.0')
    if numpy_version != 'not_installed':
        try:
            numpy_parts = [int(x) for x in numpy_version.split('.')[:2]]
            compatibility['numpy_version_ok'] = numpy_parts >= [1, 24]
        except (ValueError, IndexError):
            compatibility['numpy_version_ok'] = False
    else:
        compatibility['numpy_version_ok'] = False
    
    # Overall compatibility
    compatibility['overall_compatible'] = all([
        compatibility['python_version_ok'],
        compatibility['torch_version_ok'],
        compatibility['numpy_version_ok']
    ])
    
    return compatibility


if __name__ == "__main__":
    # Print version information when run as script
    print_version_info()
    
    # Check compatibility
    print(f"\n{'='*40}")
    print("COMPATIBILITY CHECK")
    print(f"{'='*40}")
    
    compatibility = check_version_compatibility()
    for check, result in compatibility.items():
        status = "✓" if result else "✗"
        print(f"{check}: {status}")