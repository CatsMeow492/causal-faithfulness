"""
Reproducibility utilities for ensuring consistent and reproducible results.
Handles random seed management, version tracking, and experiment configuration.
"""

import os
import random
import numpy as np
import torch
import json
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import platform
import sys
from datetime import datetime


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility settings."""
    
    # Random seeds
    global_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42
    python_seed: int = 42
    
    # Deterministic settings
    torch_deterministic: bool = True
    torch_benchmark: bool = False
    
    # Environment settings
    cuda_deterministic: bool = True
    mps_deterministic: bool = True
    
    # Version tracking
    track_versions: bool = True
    track_environment: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.global_seed < 0:
            raise ValueError("global_seed must be non-negative")
        
        # Sync all seeds to global_seed if they're not explicitly set
        if self.numpy_seed == 42 and self.global_seed != 42:
            self.numpy_seed = self.global_seed
        if self.torch_seed == 42 and self.global_seed != 42:
            self.torch_seed = self.global_seed
        if self.python_seed == 42 and self.global_seed != 42:
            self.python_seed = self.global_seed


@dataclass
class ExperimentMetadata:
    """Metadata for experiment reproducibility."""
    
    # Experiment identification
    experiment_id: str
    timestamp: str
    description: Optional[str] = None
    
    # Reproducibility settings
    seeds: Dict[str, int] = None
    deterministic_settings: Dict[str, bool] = None
    
    # System information
    system_info: Dict[str, str] = None
    python_version: str = None
    
    # Package versions
    package_versions: Dict[str, str] = None
    
    # Hardware information
    hardware_info: Dict[str, Any] = None
    
    # Configuration hash
    config_hash: Optional[str] = None


class ReproducibilityManager:
    """Manager for ensuring reproducible experiments."""
    
    def __init__(self, config: Optional[ReproducibilityConfig] = None):
        """
        Initialize reproducibility manager.
        
        Args:
            config: Reproducibility configuration (uses default if None)
        """
        self.config = config or ReproducibilityConfig()
        self._is_initialized = False
        self._metadata = None
        
    def initialize_reproducibility(self, experiment_id: Optional[str] = None) -> ExperimentMetadata:
        """
        Initialize reproducibility settings for an experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            
        Returns:
            ExperimentMetadata with all reproducibility information
        """
        if experiment_id is None:
            experiment_id = self._generate_experiment_id()
        
        # Set all random seeds
        self._set_random_seeds()
        
        # Configure deterministic behavior
        self._configure_deterministic_behavior()
        
        # Collect metadata
        self._metadata = self._collect_experiment_metadata(experiment_id)
        
        self._is_initialized = True
        
        return self._metadata
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{random_suffix}"
    
    def _set_random_seeds(self):
        """Set all random seeds for reproducibility."""
        # Python random
        random.seed(self.config.python_seed)
        
        # NumPy random
        np.random.seed(self.config.numpy_seed)
        
        # PyTorch random
        torch.manual_seed(self.config.torch_seed)
        
        # CUDA random (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.torch_seed)
            torch.cuda.manual_seed_all(self.config.torch_seed)
        
        # MPS random (if available)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.manual_seed(self.config.torch_seed)
        
        # Set environment variable for Python hash seed
        os.environ['PYTHONHASHSEED'] = str(self.config.python_seed)
    
    def _configure_deterministic_behavior(self):
        """Configure deterministic behavior for PyTorch and other libraries."""
        if self.config.torch_deterministic:
            # Enable deterministic algorithms
            torch.use_deterministic_algorithms(True, warn_only=True)
            
            # Set deterministic behavior for CUDA
            if torch.cuda.is_available() and self.config.cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = self.config.torch_benchmark
                
                # Set CUDA environment variables
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            
            # Set deterministic behavior for MPS
            if (hasattr(torch.backends, 'mps') and 
                torch.backends.mps.is_available() and 
                self.config.mps_deterministic):
                # MPS deterministic settings (if available in future PyTorch versions)
                pass
    
    def _collect_experiment_metadata(self, experiment_id: str) -> ExperimentMetadata:
        """Collect comprehensive experiment metadata."""
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            seeds={
                'global_seed': self.config.global_seed,
                'numpy_seed': self.config.numpy_seed,
                'torch_seed': self.config.torch_seed,
                'python_seed': self.config.python_seed
            },
            deterministic_settings={
                'torch_deterministic': self.config.torch_deterministic,
                'torch_benchmark': self.config.torch_benchmark,
                'cuda_deterministic': self.config.cuda_deterministic,
                'mps_deterministic': self.config.mps_deterministic
            }
        )
        
        if self.config.track_environment:
            metadata.system_info = self._get_system_info()
            metadata.python_version = sys.version
            metadata.hardware_info = self._get_hardware_info()
        
        if self.config.track_versions:
            metadata.package_versions = self._get_package_versions()
        
        # Generate configuration hash
        metadata.config_hash = self._generate_config_hash()
        
        return metadata
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_implementation': platform.python_implementation(),
            'python_version': platform.python_version()
        }
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        hardware_info = {}
        
        # CPU information
        try:
            import psutil
            hardware_info['cpu_count'] = psutil.cpu_count()
            hardware_info['memory_total_gb'] = psutil.virtual_memory().total / 1e9
        except ImportError:
            hardware_info['cpu_count'] = os.cpu_count()
        
        # GPU information
        if torch.cuda.is_available():
            hardware_info['cuda_available'] = True
            hardware_info['cuda_version'] = torch.version.cuda
            hardware_info['gpu_count'] = torch.cuda.device_count()
            
            # Get GPU details
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'name': props.name,
                    'memory_total_gb': props.total_memory / 1e9,
                    'compute_capability': f"{props.major}.{props.minor}"
                })
            hardware_info['gpu_details'] = gpu_info
        else:
            hardware_info['cuda_available'] = False
        
        # MPS information
        if hasattr(torch.backends, 'mps'):
            hardware_info['mps_available'] = torch.backends.mps.is_available()
            hardware_info['mps_built'] = torch.backends.mps.is_built()
        
        return hardware_info
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        versions = {}
        
        # Core packages
        packages_to_check = [
            'torch', 'numpy', 'scipy', 'scikit-learn', 'pandas',
            'matplotlib', 'seaborn', 'transformers', 'datasets',
            'shap', 'lime', 'captum'
        ]
        
        for package in packages_to_check:
            try:
                if package == 'torch':
                    versions[package] = torch.__version__
                elif package == 'numpy':
                    versions[package] = np.__version__
                else:
                    # Try to import and get version
                    module = __import__(package)
                    if hasattr(module, '__version__'):
                        versions[package] = module.__version__
                    elif hasattr(module, 'version'):
                        versions[package] = module.version
                    else:
                        versions[package] = 'unknown'
            except ImportError:
                versions[package] = 'not_installed'
        
        return versions
    
    def _generate_config_hash(self) -> str:
        """Generate a hash of the current configuration."""
        config_dict = asdict(self.config)
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def save_metadata(self, filepath: str):
        """Save experiment metadata to file."""
        if self._metadata is None:
            raise RuntimeError("Reproducibility not initialized. Call initialize_reproducibility() first.")
        
        metadata_dict = asdict(self._metadata)
        
        with open(filepath, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
    
    def load_metadata(self, filepath: str) -> ExperimentMetadata:
        """Load experiment metadata from file."""
        with open(filepath, 'r') as f:
            metadata_dict = json.load(f)
        
        return ExperimentMetadata(**metadata_dict)
    
    def verify_reproducibility(self, reference_metadata: ExperimentMetadata) -> Dict[str, bool]:
        """
        Verify that current environment matches reference metadata.
        
        Args:
            reference_metadata: Reference metadata to compare against
            
        Returns:
            Dictionary of verification results
        """
        if self._metadata is None:
            raise RuntimeError("Reproducibility not initialized. Call initialize_reproducibility() first.")
        
        verification = {}
        
        # Check seeds
        verification['seeds_match'] = (
            self._metadata.seeds == reference_metadata.seeds
        )
        
        # Check deterministic settings
        verification['deterministic_settings_match'] = (
            self._metadata.deterministic_settings == reference_metadata.deterministic_settings
        )
        
        # Check Python version
        verification['python_version_match'] = (
            self._metadata.python_version == reference_metadata.python_version
        )
        
        # Check key package versions
        if (self._metadata.package_versions and reference_metadata.package_versions):
            key_packages = ['torch', 'numpy', 'scipy', 'scikit-learn']
            package_matches = []
            
            for package in key_packages:
                current_version = self._metadata.package_versions.get(package)
                reference_version = reference_metadata.package_versions.get(package)
                package_matches.append(current_version == reference_version)
            
            verification['key_packages_match'] = all(package_matches)
        else:
            verification['key_packages_match'] = None
        
        # Check configuration hash
        verification['config_hash_match'] = (
            self._metadata.config_hash == reference_metadata.config_hash
        )
        
        return verification
    
    def get_reproducibility_report(self) -> str:
        """Generate a human-readable reproducibility report."""
        if self._metadata is None:
            raise RuntimeError("Reproducibility not initialized. Call initialize_reproducibility() first.")
        
        report = []
        report.append("=== Reproducibility Report ===")
        report.append(f"Experiment ID: {self._metadata.experiment_id}")
        report.append(f"Timestamp: {self._metadata.timestamp}")
        report.append("")
        
        # Seeds
        report.append("Random Seeds:")
        for seed_type, seed_value in self._metadata.seeds.items():
            report.append(f"  {seed_type}: {seed_value}")
        report.append("")
        
        # Deterministic settings
        report.append("Deterministic Settings:")
        for setting, value in self._metadata.deterministic_settings.items():
            report.append(f"  {setting}: {value}")
        report.append("")
        
        # System information
        if self._metadata.system_info:
            report.append("System Information:")
            for key, value in self._metadata.system_info.items():
                report.append(f"  {key}: {value}")
            report.append("")
        
        # Hardware information
        if self._metadata.hardware_info:
            report.append("Hardware Information:")
            for key, value in self._metadata.hardware_info.items():
                if isinstance(value, list):
                    report.append(f"  {key}:")
                    for item in value:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                report.append(f"    {k}: {v}")
                        else:
                            report.append(f"    {item}")
                else:
                    report.append(f"  {key}: {value}")
            report.append("")
        
        # Package versions
        if self._metadata.package_versions:
            report.append("Package Versions:")
            for package, version in sorted(self._metadata.package_versions.items()):
                report.append(f"  {package}: {version}")
            report.append("")
        
        report.append(f"Configuration Hash: {self._metadata.config_hash}")
        
        return "\n".join(report)


# Global reproducibility manager instance
_global_manager = None


def set_global_reproducibility(config: Optional[ReproducibilityConfig] = None) -> ExperimentMetadata:
    """
    Set global reproducibility settings.
    
    Args:
        config: Reproducibility configuration (uses default if None)
        
    Returns:
        ExperimentMetadata with reproducibility information
    """
    global _global_manager
    _global_manager = ReproducibilityManager(config)
    return _global_manager.initialize_reproducibility()


def get_global_manager() -> Optional[ReproducibilityManager]:
    """Get the global reproducibility manager."""
    return _global_manager


def ensure_reproducibility(seed: int = 42) -> ExperimentMetadata:
    """
    Convenience function to ensure reproducibility with a given seed.
    
    Args:
        seed: Random seed to use for all random number generators
        
    Returns:
        ExperimentMetadata with reproducibility information
    """
    config = ReproducibilityConfig(global_seed=seed)
    return set_global_reproducibility(config)


def create_reproducible_config(
    seed: int = 42,
    deterministic: bool = True,
    track_versions: bool = True
) -> ReproducibilityConfig:
    """
    Create a reproducibility configuration with common settings.
    
    Args:
        seed: Random seed for all generators
        deterministic: Whether to enable deterministic algorithms
        track_versions: Whether to track package versions
        
    Returns:
        ReproducibilityConfig with specified settings
    """
    return ReproducibilityConfig(
        global_seed=seed,
        numpy_seed=seed,
        torch_seed=seed,
        python_seed=seed,
        torch_deterministic=deterministic,
        torch_benchmark=not deterministic,  # Disable benchmark for determinism
        cuda_deterministic=deterministic,
        mps_deterministic=deterministic,
        track_versions=track_versions,
        track_environment=True
    )


def save_reproducibility_info(filepath: str, experiment_id: Optional[str] = None):
    """
    Save current reproducibility information to file.
    
    Args:
        filepath: Path to save the metadata file
        experiment_id: Optional experiment ID (generated if None)
    """
    if _global_manager is None:
        # Initialize with default settings
        set_global_reproducibility()
    
    _global_manager.save_metadata(filepath)


def load_and_verify_reproducibility(filepath: str) -> Dict[str, bool]:
    """
    Load reference metadata and verify current environment matches.
    
    Args:
        filepath: Path to reference metadata file
        
    Returns:
        Dictionary of verification results
    """
    if _global_manager is None:
        set_global_reproducibility()
    
    reference_metadata = _global_manager.load_metadata(filepath)
    return _global_manager.verify_reproducibility(reference_metadata)