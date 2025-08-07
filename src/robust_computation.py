"""
Computational robustness utilities for memory management, numerical stability,
and streaming computation for large datasets.
"""

import torch
import numpy as np
import warnings
import gc
import psutil
from typing import Iterator, Callable, Any, Optional, Union, Tuple, List
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class ComputationLimits:
    """Configuration for computational limits and robustness."""
    max_memory_usage_gb: float = 8.0  # Maximum memory usage in GB
    numerical_epsilon: float = 1e-8   # Epsilon for numerical stability
    gradient_clip_value: float = 1.0  # Gradient clipping threshold
    min_batch_size: int = 1           # Minimum allowed batch size
    max_retries: int = 3              # Maximum retries for failed operations
    memory_check_interval: int = 100  # Check memory every N operations


class MemoryManager:
    """Memory management utilities with OOM handling."""
    
    def __init__(self, limits: ComputationLimits):
        self.limits = limits
        self.operation_count = 0
        
    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1e9
    
    def get_available_memory_gb(self) -> float:
        """Get available system memory in GB."""
        return psutil.virtual_memory().available / 1e9
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        current_usage = self.get_memory_usage_gb()
        return current_usage < self.limits.max_memory_usage_gb
    
    def force_cleanup(self):
        """Force garbage collection and cache cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    @contextmanager
    def memory_monitor(self):
        """Context manager that monitors memory usage."""
        initial_memory = self.get_memory_usage_gb()
        try:
            yield
        finally:
            final_memory = self.get_memory_usage_gb()
            memory_increase = final_memory - initial_memory
            
            if memory_increase > 1.0:  # More than 1GB increase
                warnings.warn(f"Memory usage increased by {memory_increase:.1f}GB during operation")
                self.force_cleanup()
    
    def adaptive_batch_size(self, current_batch_size: int, target_memory_gb: float = None) -> int:
        """
        Adaptively adjust batch size based on memory usage.
        
        Args:
            current_batch_size: Current batch size
            target_memory_gb: Target memory usage (uses limit if None)
            
        Returns:
            Adjusted batch size
        """
        if target_memory_gb is None:
            target_memory_gb = self.limits.max_memory_usage_gb * 0.8  # Use 80% of limit
        
        current_memory = self.get_memory_usage_gb()
        available_memory = self.get_available_memory_gb()
        
        if current_memory > target_memory_gb:
            # Reduce batch size
            reduction_factor = target_memory_gb / current_memory
            new_batch_size = max(self.limits.min_batch_size, 
                               int(current_batch_size * reduction_factor))
            return new_batch_size
        elif available_memory > target_memory_gb * 2:
            # Can potentially increase batch size
            increase_factor = min(2.0, available_memory / target_memory_gb)
            new_batch_size = int(current_batch_size * increase_factor)
            return new_batch_size
        else:
            return current_batch_size


class NumericalStabilizer:
    """Numerical stability utilities."""
    
    def __init__(self, limits: ComputationLimits):
        self.limits = limits
    
    def safe_divide(self, numerator: Union[torch.Tensor, np.ndarray, float], 
                   denominator: Union[torch.Tensor, np.ndarray, float]) -> Union[torch.Tensor, np.ndarray, float]:
        """Safe division with epsilon handling."""
        if isinstance(denominator, torch.Tensor):
            # Handle zero denominators by replacing with epsilon
            safe_denominator = torch.where(
                torch.abs(denominator) < self.limits.numerical_epsilon,
                torch.where(denominator >= 0, self.limits.numerical_epsilon, -self.limits.numerical_epsilon),
                denominator
            )
        elif isinstance(denominator, np.ndarray):
            # Handle zero denominators by replacing with epsilon
            safe_denominator = np.where(
                np.abs(denominator) < self.limits.numerical_epsilon,
                np.where(denominator >= 0, self.limits.numerical_epsilon, -self.limits.numerical_epsilon),
                denominator
            )
        else:
            # Scalar case
            if abs(denominator) < self.limits.numerical_epsilon:
                safe_denominator = self.limits.numerical_epsilon if denominator >= 0 else -self.limits.numerical_epsilon
            else:
                safe_denominator = denominator
        
        return numerator / safe_denominator
    
    def clip_gradients(self, parameters, max_norm: Optional[float] = None) -> float:
        """Clip gradients to prevent exploding gradients."""
        if max_norm is None:
            max_norm = self.limits.gradient_clip_value
        
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        # Compute total norm
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Clip gradients
        clip_coef = max_norm / (total_norm + self.limits.numerical_epsilon)
        if clip_coef < 1:
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def stabilize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Stabilize tensor values to prevent numerical issues."""
        # Clamp extreme values
        max_val = 1e6
        min_val = -1e6
        
        # Replace NaN and Inf values
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        tensor = torch.where(torch.isinf(tensor), torch.sign(tensor) * max_val, tensor)
        
        # Clamp to reasonable range
        tensor = torch.clamp(tensor, min_val, max_val)
        
        return tensor
    
    def safe_log(self, tensor: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Safe logarithm with epsilon handling."""
        if isinstance(tensor, torch.Tensor):
            return torch.log(torch.clamp(tensor, min=self.limits.numerical_epsilon))
        else:
            return np.log(np.clip(tensor, a_min=self.limits.numerical_epsilon, a_max=None))
    
    def safe_sqrt(self, tensor: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Safe square root with epsilon handling."""
        if isinstance(tensor, torch.Tensor):
            return torch.sqrt(torch.clamp(tensor, min=self.limits.numerical_epsilon))
        else:
            return np.sqrt(np.clip(tensor, a_min=self.limits.numerical_epsilon, a_max=None))


class StreamingComputation:
    """Streaming computation utilities for large datasets."""
    
    def __init__(self, limits: ComputationLimits):
        self.limits = limits
        self.memory_manager = MemoryManager(limits)
    
    def stream_batches(self, data: Union[torch.Tensor, np.ndarray, List], 
                      batch_size: int) -> Iterator[Union[torch.Tensor, np.ndarray, List]]:
        """
        Stream data in batches with adaptive batch sizing.
        
        Args:
            data: Input data to stream
            batch_size: Initial batch size
            
        Yields:
            Batches of data
        """
        if isinstance(data, (torch.Tensor, np.ndarray)):
            total_size = len(data)
        else:
            total_size = len(data)
        
        current_batch_size = batch_size
        i = 0
        batch_count = 0
        
        while i < total_size:
            # Check memory and adjust batch size periodically (not every iteration)
            if batch_count > 0 and batch_count % max(1, self.limits.memory_check_interval // batch_size) == 0:
                new_batch_size = self.memory_manager.adaptive_batch_size(current_batch_size)
                # Only reduce batch size, don't increase during streaming to maintain consistency
                if new_batch_size < current_batch_size:
                    current_batch_size = new_batch_size
            
            end_idx = min(i + current_batch_size, total_size)
            
            if isinstance(data, torch.Tensor):
                batch = data[i:end_idx]
            elif isinstance(data, np.ndarray):
                batch = data[i:end_idx]
            else:
                batch = data[i:end_idx]
            
            yield batch
            i = end_idx
            batch_count += 1
    
    def streaming_mean(self, data_stream: Iterator, axis: Optional[int] = None) -> Union[torch.Tensor, np.ndarray, float]:
        """Compute mean over streaming data."""
        running_sum = None
        count = 0
        
        for batch in data_stream:
            if running_sum is None:
                if isinstance(batch, torch.Tensor):
                    running_sum = torch.sum(batch, dim=axis, keepdim=True if axis is not None else False)
                else:
                    running_sum = np.sum(batch, axis=axis, keepdims=True if axis is not None else False)
            else:
                if isinstance(batch, torch.Tensor):
                    running_sum += torch.sum(batch, dim=axis, keepdim=True if axis is not None else False)
                else:
                    running_sum += np.sum(batch, axis=axis, keepdims=True if axis is not None else False)
            
            if isinstance(batch, (torch.Tensor, np.ndarray)):
                if axis is None:
                    count += batch.numel() if isinstance(batch, torch.Tensor) else batch.size
                else:
                    count += batch.shape[axis]
            else:
                count += len(batch)
        
        if count == 0:
            return 0.0
        
        return running_sum / count
    
    def streaming_std(self, data_stream: Iterator, axis: Optional[int] = None) -> Union[torch.Tensor, np.ndarray, float]:
        """Compute standard deviation over streaming data using Welford's algorithm."""
        count = 0
        mean = None
        m2 = None
        
        for batch in data_stream:
            if isinstance(batch, torch.Tensor):
                batch_flat = batch.flatten() if axis is None else batch
            else:
                batch_flat = batch.flatten() if axis is None else batch
            
            for value in batch_flat:
                count += 1
                if mean is None:
                    mean = value
                    m2 = 0 if isinstance(value, (int, float)) else torch.zeros_like(value) if isinstance(value, torch.Tensor) else np.zeros_like(value)
                else:
                    delta = value - mean
                    mean += delta / count
                    delta2 = value - mean
                    m2 += delta * delta2
        
        if count < 2:
            return 0.0
        
        variance = m2 / (count - 1)
        if isinstance(variance, torch.Tensor):
            return torch.sqrt(variance)
        elif isinstance(variance, np.ndarray):
            return np.sqrt(variance)
        else:
            return np.sqrt(variance)


class RobustOperationWrapper:
    """Wrapper for robust execution of operations with retries and fallbacks."""
    
    def __init__(self, limits: ComputationLimits):
        self.limits = limits
        self.memory_manager = MemoryManager(limits)
        self.stabilizer = NumericalStabilizer(limits)
    
    def execute_with_retries(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation with retries and error handling.
        
        Args:
            operation: Function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
        """
        last_exception = None
        
        for attempt in range(self.limits.max_retries):
            try:
                with self.memory_manager.memory_monitor():
                    result = operation(*args, **kwargs)
                    
                    # Stabilize result if it's a tensor
                    if isinstance(result, torch.Tensor):
                        result = self.stabilizer.stabilize_tensor(result)
                    
                    return result
                    
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                last_exception = e
                error_msg = str(e).lower()
                
                if "out of memory" in error_msg or "oom" in error_msg:
                    warnings.warn(f"OOM error on attempt {attempt + 1}, cleaning up memory")
                    self.memory_manager.force_cleanup()
                    
                    # Reduce batch size if possible
                    if 'batch_size' in kwargs:
                        kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                    
                    if attempt < self.limits.max_retries - 1:
                        continue
                    # If this was the last attempt, fall through to final error handling
                else:
                    # Non-memory related RuntimeError, retry if attempts remain
                    if attempt < self.limits.max_retries - 1:
                        warnings.warn(f"Operation failed on attempt {attempt + 1}: {e}")
                        continue
                    # If this was the last attempt, fall through to final error handling
            
            except Exception as e:
                last_exception = e
                if attempt < self.limits.max_retries - 1:
                    warnings.warn(f"Operation failed on attempt {attempt + 1}: {e}")
                    continue
                # If this was the last attempt, fall through to final error handling
        
        # If we get here, all retries failed
        if last_exception:
            raise RuntimeError(f"Operation failed after {self.limits.max_retries} attempts. Last error: {last_exception}")
        else:
            raise RuntimeError(f"Operation failed after {self.limits.max_retries} attempts")


# Global instances with default configuration
default_limits = ComputationLimits()
memory_manager = MemoryManager(default_limits)
numerical_stabilizer = NumericalStabilizer(default_limits)
streaming_computation = StreamingComputation(default_limits)
robust_wrapper = RobustOperationWrapper(default_limits)


# Convenience functions
def safe_divide(numerator, denominator):
    """Safe division with epsilon handling."""
    return numerical_stabilizer.safe_divide(numerator, denominator)


def clip_gradients(parameters, max_norm=None):
    """Clip gradients to prevent exploding gradients."""
    return numerical_stabilizer.clip_gradients(parameters, max_norm)


def stabilize_tensor(tensor):
    """Stabilize tensor values to prevent numerical issues."""
    return numerical_stabilizer.stabilize_tensor(tensor)


def execute_with_retries(operation, *args, **kwargs):
    """Execute operation with retries and error handling."""
    return robust_wrapper.execute_with_retries(operation, *args, **kwargs)


def stream_batches(data, batch_size):
    """Stream data in batches with adaptive batch sizing."""
    return streaming_computation.stream_batches(data, batch_size)


@contextmanager
def memory_monitor():
    """Context manager that monitors memory usage."""
    with memory_manager.memory_monitor():
        yield