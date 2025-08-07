"""
Configuration module for hardware compatibility and system settings.
Optimized for Mac M-series with MPS support and CPU fallbacks.
"""

import torch
import platform
import warnings
import psutil
import gc
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager


class HardwareConfig:
    """Hardware configuration manager for Mac M-series compatibility."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.supports_mps = self._check_mps_support()
        self.device = self._get_optimal_device()
        self.memory_info = self._get_memory_info()
        self._mps_fallback_ops = self._get_mps_fallback_operations()
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for compatibility checks."""
        return {
            'platform': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cpu_count': str(psutil.cpu_count()),
            'memory_gb': f"{psutil.virtual_memory().total / 1e9:.1f}"
        }
    
    def _check_mps_support(self) -> bool:
        """Check if MPS (Metal Performance Shaders) is available."""
        if not torch.backends.mps.is_available():
            return False
        
        if not torch.backends.mps.is_built():
            warnings.warn("MPS not built with PyTorch installation")
            return False
            
        return True
    
    def _get_optimal_device(self) -> torch.device:
        """Determine the optimal device for computation."""
        # Priority: MPS > CUDA > CPU
        if self.supports_mps and torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get memory information for different devices."""
        memory_info = {
            'system_memory_gb': psutil.virtual_memory().total / 1e9,
            'available_memory_gb': psutil.virtual_memory().available / 1e9
        }
        
        if torch.cuda.is_available():
            try:
                memory_info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
                memory_info['cuda_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            except Exception:
                pass
        
        # MPS memory is shared with system memory on Apple Silicon
        if self.supports_mps:
            memory_info['mps_shared_memory'] = True
            
        return memory_info
    
    def _get_mps_fallback_operations(self) -> List[str]:
        """Get list of operations that may need CPU fallback on MPS."""
        return [
            'torch.linalg.svd',
            'torch.symeig', 
            'torch.eig',
            'torch.qr',
            'torch.cholesky',
            'torch.triangular_solve',
            'torch.solve',
            'torch.inverse',
            'torch.det',
            'torch.logdet',
            'torch.slogdet',
            'torch.pinverse',
            'torch.matrix_rank',
            'torch.matrix_power',
            'torch.chain_matmul',
            'torch.cross',
            'torch.histc',
            'torch.bincount',
            'torch.multinomial',
            'torch.unique',
            'torch.sort',
            'torch.argsort',
            'torch.topk',
            'torch.kthvalue'
        ]
    
    def get_device(self, fallback_cpu: bool = True, operation_name: Optional[str] = None) -> torch.device:
        """
        Get device with optional CPU fallback for specific operations.
        
        Args:
            fallback_cpu: Whether to fallback to CPU if operation fails
            operation_name: Name of the operation to check for MPS compatibility
        """
        if fallback_cpu and self.device.type == 'mps':
            # Check if specific operation needs CPU fallback
            if operation_name and any(op in operation_name for op in self._mps_fallback_ops):
                return torch.device('cpu')
            
            # Test basic tensor operation
            try:
                test_tensor = torch.randn(2, 2, device=self.device)
                _ = test_tensor @ test_tensor
                return self.device
            except Exception as e:
                warnings.warn(f"MPS operation failed ({e}), falling back to CPU")
                return torch.device('cpu')
        
        return self.device
    
    def get_batch_size(self, default: int = 32, memory_factor: float = 0.8) -> int:
        """
        Get optimal batch size based on available memory with dynamic adjustment.
        
        Args:
            default: Default batch size
            memory_factor: Fraction of available memory to use
        """
        if self.device.type == 'mps':
            # MPS shares system memory - be conservative
            available_gb = self.memory_info['available_memory_gb'] * memory_factor
            # Rough estimate: 1GB can handle batch size of ~64 for typical models
            estimated_batch_size = max(1, int(available_gb * 64))
            return min(default, estimated_batch_size, 32)  # Cap at 32 for MPS
            
        elif self.device.type == 'cuda':
            try:
                # Get GPU memory info
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                available_memory = (total_memory - allocated_memory) * memory_factor
                
                # Rough estimate: 1GB can handle batch size of ~128 for typical models
                estimated_batch_size = max(1, int(available_memory / 1e9 * 128))
                return min(default, estimated_batch_size)
                
            except Exception:
                return min(default, 16)  # Conservative fallback
        else:
            # CPU - use system memory but be conservative
            available_gb = self.memory_info['available_memory_gb'] * memory_factor
            # CPU is slower, use smaller batches: 1GB -> batch size of ~16
            estimated_batch_size = max(1, int(available_gb * 16))
            return min(default, estimated_batch_size, 16)  # Cap at 16 for CPU
    
    def adjust_batch_size_for_oom(self, current_batch_size: int, reduction_factor: float = 0.5) -> int:
        """
        Reduce batch size after OOM error.
        
        Args:
            current_batch_size: Current batch size that caused OOM
            reduction_factor: Factor to reduce batch size by
            
        Returns:
            New smaller batch size
        """
        new_batch_size = max(1, int(current_batch_size * reduction_factor))
        warnings.warn(f"Reducing batch size from {current_batch_size} to {new_batch_size} due to OOM")
        return new_batch_size
    
    @contextmanager
    def memory_efficient_context(self):
        """Context manager for memory-efficient computation."""
        # Clear cache before computation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        try:
            yield
        finally:
            # Clean up after computation
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()
    
    def safe_operation(self, operation: Callable, *args, fallback_device: str = 'cpu', **kwargs):
        """
        Execute operation with automatic fallback to CPU if needed.
        
        Args:
            operation: Function to execute
            *args: Arguments for the operation
            fallback_device: Device to fallback to if operation fails
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
        """
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            if self.device.type == 'mps' and fallback_device == 'cpu':
                warnings.warn(f"MPS operation failed ({e}), retrying on CPU")
                
                # Move tensors to CPU
                cpu_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        cpu_args.append(arg.cpu())
                    else:
                        cpu_args.append(arg)
                
                cpu_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        cpu_kwargs[key] = value.cpu()
                    else:
                        cpu_kwargs[key] = value
                
                # Execute on CPU
                result = operation(*cpu_args, **cpu_kwargs)
                
                # Move result back to original device if it's a tensor
                if isinstance(result, torch.Tensor):
                    return result.to(self.device)
                else:
                    return result
            else:
                raise e


# Global hardware configuration instance
hardware_config = HardwareConfig()


def get_device(fallback_cpu: bool = True, operation_name: Optional[str] = None) -> torch.device:
    """Get the optimal device for computation with operation-specific fallbacks."""
    return hardware_config.get_device(fallback_cpu, operation_name)


def get_batch_size(default: int = 32, memory_factor: float = 0.8) -> int:
    """Get optimal batch size for the current hardware with memory consideration."""
    return hardware_config.get_batch_size(default, memory_factor)


def adjust_batch_size_for_oom(current_batch_size: int, reduction_factor: float = 0.5) -> int:
    """Reduce batch size after OOM error."""
    return hardware_config.adjust_batch_size_for_oom(current_batch_size, reduction_factor)


def memory_efficient_context():
    """Context manager for memory-efficient computation."""
    return hardware_config.memory_efficient_context()


def safe_operation(operation: Callable, *args, fallback_device: str = 'cpu', **kwargs):
    """Execute operation with automatic fallback to CPU if needed."""
    return hardware_config.safe_operation(operation, *args, fallback_device=fallback_device, **kwargs)


def print_system_info():
    """Print system and hardware information."""
    print("=== System Information ===")
    for key, value in hardware_config.system_info.items():
        print(f"{key}: {value}")
    
    print(f"\nOptimal device: {hardware_config.device}")
    print(f"MPS support: {hardware_config.supports_mps}")
    print(f"Recommended batch size: {get_batch_size()}")
    
    print(f"\n=== Memory Information ===")
    for key, value in hardware_config.memory_info.items():
        if isinstance(value, float):
            print(f"{key}: {value:.1f}")
        else:
            print(f"{key}: {value}")


def get_memory_info() -> Dict[str, Any]:
    """Get current memory information."""
    return hardware_config.memory_info.copy()


def check_mps_operation_support(operation_name: str) -> bool:
    """Check if an operation is supported on MPS."""
    if not hardware_config.supports_mps:
        return False
    return not any(op in operation_name for op in hardware_config._mps_fallback_ops)


# Configuration constants
DEFAULT_CONFIG = {
    'random_seed': 42,
    'n_monte_carlo_samples': 1000,
    'confidence_level': 0.95,
    'batch_size': get_batch_size(),
    'device': get_device(),
    'numerical_epsilon': 1e-8,
    'max_sequence_length': 512,  # For text models
    'memory_factor': 0.8,  # Fraction of available memory to use
    'oom_reduction_factor': 0.5,  # Factor to reduce batch size on OOM
}