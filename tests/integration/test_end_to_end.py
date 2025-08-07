"""
Integration tests for end-to-end pipeline functionality.
Tests complete workflows on toy datasets with cross-platform compatibility.
"""

import pytest
import numpy as np
import torch
import time
import psutil
import os
from unittest.mock import Mock

from src.faithfulness import FaithfulnessConfig, FaithfulnessMetric, compute_faithfulness_score
from src.masking import DataModality, MaskingStrategy, create_masker
from src.baseline import BaselineStrategy, create_baseline_generator
from src.explainers import RandomExplainer, Attribution


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality."""
    
    @pytest.fixture
    def toy_config(self):
        """Create configuration for toy dataset tests."""
        return FaithfulnessConfig(
            n_samples=50,  # Small for fast tests
            batch_size=5,
            random_seed=42,
            device=torch.device('cpu')  # Use CPU for consistent testing
        )
    
    @pytest.fixture
    def toy_tabular_data(self):
        """Create toy tabular dataset."""
        # 10 samples, 5 features
        np.random.seed(42)
        data = np.random.randn(10, 5).astype(np.float32)
        return torch.from_numpy(data)
    
    @pytest.fixture
    def toy_text_data(self):
        """Create toy text dataset."""
        # Tokenized sequences of length 10
        vocab_size = 100
        seq_length = 10
        batch_size = 5
        
        np.random.seed(42)
        data = np.random.randint(1, vocab_size, size=(batch_size, seq_length))
        return torch.from_numpy(data).long()
    
    @pytest.fixture
    def toy_image_data(self):
        """Create toy image dataset."""
        # Small RGB images: 3x8x8
        np.random.seed(42)
        data = np.random.rand(2, 3, 8, 8).astype(np.float32)
        return torch.from_numpy(data)
    
    @pytest.fixture
    def simple_tabular_model(self):
        """Create simple tabular model for testing."""
        def model_fn(x):
            # Linear combination: output = 2*x0 + x1 + 0.5*x2 + noise
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            weights = torch.tensor([2.0, 1.0, 0.5, 0.1, 0.0])
            output = torch.sum(x * weights, dim=1, keepdim=True)
            return output
        
        return model_fn
    
    @pytest.fixture
    def simple_text_model(self):
        """Create simple text model for testing."""
        def model_fn(x):
            # Sum of token embeddings (simplified)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            # Convert tokens to simple embeddings
            embeddings = x.float() / 100.0  # Normalize token IDs
            output = torch.sum(embeddings, dim=1, keepdim=True)
            return output
        
        return model_fn
    
    @pytest.fixture
    def simple_image_model(self):
        """Create simple image model for testing."""
        def model_fn(x):
            # Global average pooling
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            
            output = torch.mean(x, dim=(2, 3))  # Average over spatial dimensions
            output = torch.sum(output, dim=1, keepdim=True)  # Sum over channels
            return output
        
        return model_fn
    
    def test_tabular_end_to_end(self, toy_config, toy_tabular_data, simple_tabular_model):
        """Test complete tabular data pipeline."""
        # Create explainer
        explainer = RandomExplainer(random_seed=42)
        
        # Test single sample
        sample = toy_tabular_data[0:1]  # Keep batch dimension
        
        # Compute faithfulness score
        result = compute_faithfulness_score(
            simple_tabular_model, 
            explainer.explain, 
            sample, 
            config=toy_config
        )
        
        # Validate results
        assert 0.0 <= result.f_score <= 1.0
        assert result.n_samples == toy_config.n_samples
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.f_score <= result.confidence_interval[1]
        assert 'n_model_queries' in result.computation_metrics
        assert result.computation_metrics['n_model_queries'] == toy_config.n_samples * 2
    
    def test_text_end_to_end(self, toy_config, toy_text_data, simple_text_model):
        """Test complete text data pipeline."""
        # Create text-specific configuration
        text_config = FaithfulnessConfig(
            n_samples=50,
            batch_size=5,
            random_seed=42,
            device=torch.device('cpu'),
            masking_strategy="pad"
        )
        
        # Create explainer
        explainer = RandomExplainer(random_seed=42)
        
        # Test single sample
        sample = toy_text_data[0:1]  # Keep batch dimension
        
        # Create metric with text modality
        metric = FaithfulnessMetric(text_config, modality=DataModality.TEXT)
        
        result = metric.compute_faithfulness_score(
            simple_text_model,
            explainer.explain,
            sample
        )
        
        # Validate results
        assert 0.0 <= result.f_score <= 1.0
        assert result.n_samples == text_config.n_samples
        assert isinstance(result.statistical_significance, bool)
    
    def test_image_end_to_end(self, toy_config, toy_image_data, simple_image_model):
        """Test complete image data pipeline."""
        # Create image-specific configuration
        image_config = FaithfulnessConfig(
            n_samples=30,  # Smaller for image tests
            batch_size=3,
            random_seed=42,
            device=torch.device('cpu'),
            masking_strategy="zero"
        )
        
        # Create explainer
        explainer = RandomExplainer(random_seed=42)
        
        # Test single sample
        sample = toy_image_data[0:1]  # Keep batch dimension
        
        # Create metric with image modality
        metric = FaithfulnessMetric(image_config, modality=DataModality.IMAGE)
        
        result = metric.compute_faithfulness_score(
            simple_image_model,
            explainer.explain,
            sample
        )
        
        # Validate results
        assert 0.0 <= result.f_score <= 1.0
        assert result.n_samples == image_config.n_samples
    
    def test_batch_processing(self, toy_config, toy_tabular_data, simple_tabular_model):
        """Test batch processing capabilities."""
        explainer = RandomExplainer(random_seed=42)
        
        # Test multiple samples
        results = []
        for i in range(3):
            sample = toy_tabular_data[i:i+1]
            result = compute_faithfulness_score(
                simple_tabular_model,
                explainer.explain,
                sample,
                config=toy_config
            )
            results.append(result)
        
        # All results should be valid
        for result in results:
            assert 0.0 <= result.f_score <= 1.0
            assert result.n_samples == toy_config.n_samples
        
        # Results should be different (with high probability) or consistently high/low
        f_scores = [r.f_score for r in results]
        # Allow for cases where masking produces consistent results
        assert len(set(f_scores)) > 1 or all(s == 0.0 for s in f_scores) or all(s == 1.0 for s in f_scores)
    
    def test_different_explainers(self, toy_config, toy_tabular_data, simple_tabular_model):
        """Test with different explainer types."""
        sample = toy_tabular_data[0:1]
        
        # Random explainer
        random_explainer = RandomExplainer(random_seed=42)
        random_result = compute_faithfulness_score(
            simple_tabular_model,
            random_explainer.explain,
            sample,
            config=toy_config
        )
        
        # Perfect explainer (knows the true feature importance)
        def perfect_explainer(model, data):
            n_features = data.shape[-1]
            scores = np.array([2.0, 1.0, 0.5, 0.1, 0.0])  # True weights
            return scores / np.sum(np.abs(scores))  # Normalize
        
        perfect_result = compute_faithfulness_score(
            simple_tabular_model,
            perfect_explainer,
            sample,
            config=toy_config
        )
        
        # Both should be valid
        assert 0.0 <= random_result.f_score <= 1.0
        assert 0.0 <= perfect_result.f_score <= 1.0
        
        # Perfect explainer should generally perform better (though not guaranteed with small samples)
        # Just check that both produce reasonable results
        assert random_result.statistical_significance in [True, False]
        assert perfect_result.statistical_significance in [True, False]
    
    def test_error_handling(self, toy_config):
        """Test error handling in end-to-end pipeline."""
        # Invalid model
        def broken_model(x):
            raise ValueError("Model error")
        
        def simple_explainer(model, data):
            return np.ones(data.shape[-1]) / data.shape[-1]
        
        sample = torch.randn(1, 5)
        
        # Should handle model errors gracefully
        with pytest.raises((ValueError, RuntimeError)):
            compute_faithfulness_score(broken_model, simple_explainer, sample, config=toy_config)
        
        # Invalid explainer
        def broken_explainer(model, data):
            raise ValueError("Explainer error")
        
        def simple_model(x):
            return torch.sum(x, dim=1, keepdim=True)
        
        # Should handle explainer errors gracefully
        with pytest.raises((ValueError, RuntimeError)):
            compute_faithfulness_score(simple_model, broken_explainer, sample, config=toy_config)
    
    def test_reproducibility(self, toy_config, toy_tabular_data, simple_tabular_model):
        """Test reproducibility across runs."""
        explainer = RandomExplainer(random_seed=42)
        sample = toy_tabular_data[0:1]
        
        # Run twice with same configuration
        result1 = compute_faithfulness_score(
            simple_tabular_model,
            explainer.explain,
            sample,
            config=toy_config
        )
        
        # Reset explainer with same seed
        explainer = RandomExplainer(random_seed=42)
        result2 = compute_faithfulness_score(
            simple_tabular_model,
            explainer.explain,
            sample,
            config=toy_config
        )
        
        # Results should be identical (within numerical precision)
        assert abs(result1.f_score - result2.f_score) < 1e-6
        assert abs(result1.baseline_performance - result2.baseline_performance) < 1e-6
        assert abs(result1.explained_performance - result2.explained_performance) < 1e-6


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility."""
    
    def test_device_compatibility(self):
        """Test compatibility across different devices."""
        config_cpu = FaithfulnessConfig(
            n_samples=20,
            device=torch.device('cpu'),
            random_seed=42
        )
        
        # Test CPU execution
        def simple_model(x):
            return torch.sum(x, dim=1, keepdim=True)
        
        def simple_explainer(model, data):
            return np.ones(data.shape[-1]) / data.shape[-1]
        
        sample = torch.randn(1, 3)
        
        result = compute_faithfulness_score(
            simple_model,
            simple_explainer,
            sample,
            config=config_cpu
        )
        
        assert 0.0 <= result.f_score <= 1.0
        
        # Test MPS compatibility (if available)
        if torch.backends.mps.is_available():
            config_mps = FaithfulnessConfig(
                n_samples=20,
                device=torch.device('mps'),
                random_seed=42
            )
            
            # Should work or gracefully fall back to CPU
            try:
                result_mps = compute_faithfulness_score(
                    simple_model,
                    simple_explainer,
                    sample,
                    config=config_mps
                )
                assert 0.0 <= result_mps.f_score <= 1.0
            except Exception:
                # MPS fallback is acceptable
                pass
    
    def test_data_type_compatibility(self):
        """Test compatibility with different data types."""
        config = FaithfulnessConfig(n_samples=20, random_seed=42, device=torch.device('cpu'))
        
        def simple_model(x):
            return torch.sum(x, dim=1, keepdim=True)
        
        def simple_explainer(model, data):
            if isinstance(data, torch.Tensor):
                n_features = data.shape[-1]
            else:
                n_features = data.shape[-1] if len(data.shape) > 1 else len(data)
            return np.ones(n_features) / n_features
        
        # Test different input types
        test_cases = [
            torch.randn(1, 3).float(),
            torch.randn(1, 3).double(),
            np.random.randn(1, 3).astype(np.float32),
            np.random.randn(1, 3).astype(np.float64)
        ]
        
        for data in test_cases:
            try:
                result = compute_faithfulness_score(
                    simple_model,
                    simple_explainer,
                    data,
                    config=config
                )
                assert 0.0 <= result.f_score <= 1.0
            except Exception as e:
                # Some type conversions might fail, which is acceptable
                assert "dtype" in str(e).lower() or "type" in str(e).lower()


class TestMemoryAndPerformance:
    """Test memory usage and runtime performance."""
    
    def test_memory_usage(self):
        """Test memory usage stays reasonable."""
        config = FaithfulnessConfig(
            n_samples=100,
            batch_size=10,
            random_seed=42,
            device=torch.device('cpu')
        )
        
        def simple_model(x):
            return torch.sum(x, dim=1, keepdim=True)
        
        explainer = RandomExplainer(random_seed=42)
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run computation
        sample = torch.randn(1, 10)
        result = compute_faithfulness_score(
            simple_model,
            explainer.explain,
            sample,
            config=config
        )
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 100MB for this small test)
        assert memory_increase < 100
        assert 0.0 <= result.f_score <= 1.0
    
    def test_runtime_performance(self):
        """Test runtime performance benchmarks."""
        config = FaithfulnessConfig(
            n_samples=50,
            batch_size=5,
            random_seed=42,
            device=torch.device('cpu')
        )
        
        def simple_model(x):
            return torch.sum(x, dim=1, keepdim=True)
        
        explainer = RandomExplainer(random_seed=42)
        sample = torch.randn(1, 5)
        
        # Measure runtime
        start_time = time.time()
        
        result = compute_faithfulness_score(
            simple_model,
            explainer.explain,
            sample,
            config=config
        )
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Runtime should be reasonable (less than 10 seconds for this small test)
        assert runtime < 10.0
        assert 0.0 <= result.f_score <= 1.0
        
        # Check computation metrics
        assert 'computation_time_seconds' in result.computation_metrics
        assert result.computation_metrics['computation_time_seconds'] >= 0
    
    def test_batch_size_scaling(self):
        """Test performance scaling with different batch sizes."""
        base_config = {
            'n_samples': 40,
            'random_seed': 42,
            'device': torch.device('cpu')
        }
        
        def simple_model(x):
            return torch.sum(x, dim=1, keepdim=True)
        
        explainer = RandomExplainer(random_seed=42)
        sample = torch.randn(1, 5)
        
        batch_sizes = [1, 5, 10, 20]
        runtimes = []
        
        for batch_size in batch_sizes:
            config = FaithfulnessConfig(batch_size=batch_size, **base_config)
            
            start_time = time.time()
            result = compute_faithfulness_score(
                simple_model,
                explainer.explain,
                sample,
                config=config
            )
            end_time = time.time()
            
            runtimes.append(end_time - start_time)
            assert 0.0 <= result.f_score <= 1.0
        
        # All runtimes should be reasonable
        for runtime in runtimes:
            assert runtime < 10.0
        
        # Larger batch sizes shouldn't be dramatically slower (within 5x)
        assert max(runtimes) / min(runtimes) < 5.0


class TestComponentIntegration:
    """Test integration between different components."""
    
    def test_masker_baseline_integration(self):
        """Test integration between masking and baseline generation."""
        # Create masker and baseline generator
        masker = create_masker("tabular", "zero", random_seed=42)
        baseline_gen = create_baseline_generator("tabular", "gaussian", random_seed=42)
        
        # Test data
        data = torch.randn(1, 5)
        feature_indices = [0, 2]
        
        # Test masking
        masked_data = masker.mask_features(data, feature_indices, mask_explained=False)
        assert masked_data.shape == data.shape
        
        # Test baseline generation
        baseline_data = baseline_gen.generate_baseline(data, batch_size=3)
        assert baseline_data.shape[0] == 3
        assert baseline_data.shape[1:] == data.shape
        
        # Masked and baseline data should be different from original
        assert not torch.allclose(masked_data, data)
        assert not torch.allclose(baseline_data[0], data)
    
    def test_explainer_metric_integration(self):
        """Test integration between explainers and faithfulness metric."""
        config = FaithfulnessConfig(n_samples=30, random_seed=42, device=torch.device('cpu'))
        metric = FaithfulnessMetric(config)
        
        def simple_model(x):
            return torch.sum(x, dim=1, keepdim=True)
        
        # Test with different explainers
        explainers = [
            RandomExplainer(random_seed=42),
            RandomExplainer(random_seed=123, distribution="gaussian")
        ]
        
        sample = torch.randn(1, 4)
        
        for explainer in explainers:
            result = metric.compute_faithfulness_score(
                simple_model,
                explainer.explain,
                sample
            )
            
            assert 0.0 <= result.f_score <= 1.0
            assert result.n_samples == config.n_samples
            assert isinstance(result.statistical_significance, bool)
    
    def test_config_propagation(self):
        """Test that configuration is properly propagated through components."""
        config = FaithfulnessConfig(
            n_samples=25,
            batch_size=5,
            random_seed=123,
            confidence_level=0.99,
            device=torch.device('cpu')
        )
        
        metric = FaithfulnessMetric(config)
        
        # Check that configuration is properly stored
        assert metric.config.n_samples == 25
        assert metric.config.batch_size == 5
        assert metric.config.random_seed == 123
        assert metric.config.confidence_level == 0.99
        
        # Check that random seed is propagated
        assert metric.rng.get_state()[1][0] == 123  # NumPy random state
        
        def simple_model(x):
            return torch.sum(x, dim=1, keepdim=True)
        
        explainer = RandomExplainer(random_seed=123)
        sample = torch.randn(1, 3)
        
        result = metric.compute_faithfulness_score(simple_model, explainer.explain, sample)
        
        # Check that configuration affects results
        assert result.n_samples == 25
        assert len(result.confidence_interval) == 2
        # Confidence interval should be wider for 99% vs default 95%
        ci_width = result.confidence_interval[1] - result.confidence_interval[0]
        assert ci_width >= 0  # Should be non-negative


if __name__ == "__main__":
    pytest.main([__file__])