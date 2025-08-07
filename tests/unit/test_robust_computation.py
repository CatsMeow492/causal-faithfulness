"""
Unit tests for robust computation utilities.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import warnings

from src.robust_computation import (
    ComputationLimits, MemoryManager, NumericalStabilizer, 
    StreamingComputation, RobustOperationWrapper,
    safe_divide, clip_gradients, stabilize_tensor, 
    execute_with_retries, stream_batches
)


class TestComputationLimits:
    """Test ComputationLimits configuration."""
    
    def test_default_limits(self):
        """Test default computation limits."""
        limits = ComputationLimits()
        assert limits.max_memory_usage_gb == 8.0
        assert limits.numerical_epsilon == 1e-8
        assert limits.gradient_clip_value == 1.0
        assert limits.min_batch_size == 1
        assert limits.max_retries == 3
        assert limits.memory_check_interval == 100
    
    def test_custom_limits(self):
        """Test custom computation limits."""
        limits = ComputationLimits(
            max_memory_usage_gb=16.0,
            numerical_epsilon=1e-10,
            gradient_clip_value=0.5
        )
        assert limits.max_memory_usage_gb == 16.0
        assert limits.numerical_epsilon == 1e-10
        assert limits.gradient_clip_value == 0.5


class TestMemoryManager:
    """Test MemoryManager functionality."""
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        limits = ComputationLimits()
        manager = MemoryManager(limits)
        
        # Should return positive memory usage
        usage = manager.get_memory_usage_gb()
        assert usage > 0
        
        # Should return positive available memory
        available = manager.get_available_memory_gb()
        assert available > 0
    
    def test_memory_check(self):
        """Test memory usage checking."""
        limits = ComputationLimits(max_memory_usage_gb=1000.0)  # Very high limit
        manager = MemoryManager(limits)
        
        # Should pass with high limit
        assert manager.check_memory_usage() is True
        
        # Should fail with very low limit
        limits_low = ComputationLimits(max_memory_usage_gb=0.001)  # Very low limit
        manager_low = MemoryManager(limits_low)
        assert manager_low.check_memory_usage() is False
    
    def test_adaptive_batch_size(self):
        """Test adaptive batch size adjustment."""
        limits = ComputationLimits()
        manager = MemoryManager(limits)
        
        # Test with reasonable memory
        batch_size = manager.adaptive_batch_size(32)
        assert batch_size >= limits.min_batch_size
        assert isinstance(batch_size, int)
    
    def test_memory_monitor_context(self):
        """Test memory monitoring context manager."""
        limits = ComputationLimits()
        manager = MemoryManager(limits)
        
        with manager.memory_monitor():
            # Create some tensors
            x = torch.randn(100, 100)
            y = torch.randn(100, 100)
            z = x @ y
        
        # Should complete without error


class TestNumericalStabilizer:
    """Test NumericalStabilizer functionality."""
    
    def test_safe_divide_tensor(self):
        """Test safe division with tensors."""
        limits = ComputationLimits()
        stabilizer = NumericalStabilizer(limits)
        
        numerator = torch.tensor([1.0, 2.0, 3.0])
        denominator = torch.tensor([2.0, 0.0, 1.0])  # Contains zero
        
        result = stabilizer.safe_divide(numerator, denominator)
        
        assert torch.isfinite(result).all()
        assert result[0] == 0.5  # 1/2
        assert result[2] == 3.0  # 3/1
        # result[1] should be finite (not inf)
        assert torch.isfinite(result[1])
    
    def test_safe_divide_numpy(self):
        """Test safe division with numpy arrays."""
        limits = ComputationLimits()
        stabilizer = NumericalStabilizer(limits)
        
        numerator = np.array([1.0, 2.0, 3.0])
        denominator = np.array([2.0, 0.0, 1.0])  # Contains zero
        
        result = stabilizer.safe_divide(numerator, denominator)
        
        assert np.isfinite(result).all()
        assert result[0] == 0.5  # 1/2
        assert result[2] == 3.0  # 3/1
        # result[1] should be finite (not inf)
        assert np.isfinite(result[1])
    
    def test_safe_divide_scalar(self):
        """Test safe division with scalars."""
        limits = ComputationLimits()
        stabilizer = NumericalStabilizer(limits)
        
        # Normal division
        result = stabilizer.safe_divide(6.0, 2.0)
        assert result == 3.0
        
        # Division by zero
        result = stabilizer.safe_divide(1.0, 0.0)
        assert np.isfinite(result)
        
        # Division by very small number
        result = stabilizer.safe_divide(1.0, 1e-10)
        assert np.isfinite(result)
    
    def test_stabilize_tensor(self):
        """Test tensor stabilization."""
        limits = ComputationLimits()
        stabilizer = NumericalStabilizer(limits)
        
        # Create tensor with problematic values
        tensor = torch.tensor([1.0, float('inf'), float('-inf'), float('nan'), 1e8])
        
        stabilized = stabilizer.stabilize_tensor(tensor)
        
        # Should have no NaN or Inf values
        assert torch.isfinite(stabilized).all()
        assert not torch.isnan(stabilized).any()
        assert not torch.isinf(stabilized).any()
        
        # Normal values should be preserved
        assert stabilized[0] == 1.0
    
    def test_safe_log(self):
        """Test safe logarithm."""
        limits = ComputationLimits()
        stabilizer = NumericalStabilizer(limits)
        
        # Test with tensor
        tensor = torch.tensor([1.0, 0.0, -1.0, 2.0])
        result = stabilizer.safe_log(tensor)
        assert torch.isfinite(result).all()
        
        # Test with numpy
        array = np.array([1.0, 0.0, -1.0, 2.0])
        result = stabilizer.safe_log(array)
        assert np.isfinite(result).all()
    
    def test_safe_sqrt(self):
        """Test safe square root."""
        limits = ComputationLimits()
        stabilizer = NumericalStabilizer(limits)
        
        # Test with tensor
        tensor = torch.tensor([4.0, 0.0, -1.0, 9.0])
        result = stabilizer.safe_sqrt(tensor)
        assert torch.isfinite(result).all()
        assert result[0] == 2.0
        assert result[3] == 3.0
        
        # Test with numpy
        array = np.array([4.0, 0.0, -1.0, 9.0])
        result = stabilizer.safe_sqrt(array)
        assert np.isfinite(result).all()
        assert result[0] == 2.0
        assert result[3] == 3.0
    
    def test_clip_gradients(self):
        """Test gradient clipping."""
        limits = ComputationLimits(gradient_clip_value=1.0)
        stabilizer = NumericalStabilizer(limits)
        
        # Create tensor with gradient
        tensor = torch.randn(10, requires_grad=True)
        loss = tensor.sum()
        loss.backward()
        
        # Set large gradient
        tensor.grad.data.fill_(10.0)
        
        # Clip gradients
        total_norm = stabilizer.clip_gradients([tensor])
        
        # Gradient should be clipped
        assert tensor.grad.norm().item() <= limits.gradient_clip_value + 1e-6
        assert total_norm > limits.gradient_clip_value


class TestStreamingComputation:
    """Test StreamingComputation functionality."""
    
    def test_stream_batches_tensor(self):
        """Test streaming batches with tensor data."""
        limits = ComputationLimits()
        streaming = StreamingComputation(limits)
        
        data = torch.randn(100, 10)
        batch_size = 32
        
        batches = list(streaming.stream_batches(data, batch_size))
        
        # Should have correct number of batches
        expected_batches = (100 + batch_size - 1) // batch_size
        assert len(batches) == expected_batches
        
        # All batches should be tensors
        for batch in batches:
            assert isinstance(batch, torch.Tensor)
            assert batch.shape[1] == 10  # Feature dimension preserved
        
        # Total size should match
        total_size = sum(len(batch) for batch in batches)
        assert total_size == 100
    
    def test_stream_batches_numpy(self):
        """Test streaming batches with numpy data."""
        limits = ComputationLimits()
        streaming = StreamingComputation(limits)
        
        data = np.random.randn(100, 10)
        batch_size = 32
        
        batches = list(streaming.stream_batches(data, batch_size))
        
        # Should have correct number of batches
        expected_batches = (100 + batch_size - 1) // batch_size
        assert len(batches) == expected_batches
        
        # All batches should be numpy arrays
        for batch in batches:
            assert isinstance(batch, np.ndarray)
            assert batch.shape[1] == 10  # Feature dimension preserved
        
        # Total size should match
        total_size = sum(len(batch) for batch in batches)
        assert total_size == 100
    
    def test_stream_batches_list(self):
        """Test streaming batches with list data."""
        limits = ComputationLimits()
        streaming = StreamingComputation(limits)
        
        data = list(range(100))
        batch_size = 32
        
        batches = list(streaming.stream_batches(data, batch_size))
        
        # Should have correct number of batches
        expected_batches = (100 + batch_size - 1) // batch_size
        assert len(batches) == expected_batches
        
        # All batches should be lists
        for batch in batches:
            assert isinstance(batch, list)
        
        # Total size should match
        total_size = sum(len(batch) for batch in batches)
        assert total_size == 100


class TestRobustOperationWrapper:
    """Test RobustOperationWrapper functionality."""
    
    def test_successful_operation(self):
        """Test successful operation execution."""
        limits = ComputationLimits()
        wrapper = RobustOperationWrapper(limits)
        
        def simple_operation(x, y):
            return x + y
        
        result = wrapper.execute_with_retries(simple_operation, 2, 3)
        assert result == 5
    
    def test_operation_with_retries(self):
        """Test operation that fails then succeeds."""
        limits = ComputationLimits(max_retries=3)
        wrapper = RobustOperationWrapper(limits)
        
        call_count = 0
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = wrapper.execute_with_retries(failing_operation)
        assert result == "success"
        assert call_count == 3
    
    def test_operation_max_retries_exceeded(self):
        """Test operation that always fails."""
        limits = ComputationLimits(max_retries=2)
        wrapper = RobustOperationWrapper(limits)
        
        def always_failing_operation():
            raise RuntimeError("Always fails")
        
        with pytest.raises(RuntimeError, match="Operation failed after 2 attempts"):
            wrapper.execute_with_retries(always_failing_operation)
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_oom_handling(self, mock_cuda_available):
        """Test OOM error handling."""
        limits = ComputationLimits(max_retries=2)
        wrapper = RobustOperationWrapper(limits)
        
        call_count = 0
        def oom_operation(batch_size=32):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return f"success with batch_size={batch_size}"
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = wrapper.execute_with_retries(oom_operation, batch_size=32)
        
        assert "success" in result
        assert "batch_size=16" in result  # Should have reduced batch size
        assert call_count == 2


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_safe_divide_function(self):
        """Test safe_divide convenience function."""
        result = safe_divide(6.0, 2.0)
        assert result == 3.0
        
        result = safe_divide(1.0, 0.0)
        assert np.isfinite(result)
    
    def test_stabilize_tensor_function(self):
        """Test stabilize_tensor convenience function."""
        tensor = torch.tensor([1.0, float('inf'), float('nan')])
        result = stabilize_tensor(tensor)
        
        assert torch.isfinite(result).all()
        assert result[0] == 1.0
    
    def test_execute_with_retries_function(self):
        """Test execute_with_retries convenience function."""
        def simple_op(x):
            return x * 2
        
        result = execute_with_retries(simple_op, 5)
        assert result == 10
    
    def test_stream_batches_function(self):
        """Test stream_batches convenience function."""
        data = list(range(10))
        batches = list(stream_batches(data, 3))
        
        assert len(batches) == 4  # 10 items in batches of 3
        assert sum(len(batch) for batch in batches) == 10


if __name__ == "__main__":
    pytest.main([__file__])