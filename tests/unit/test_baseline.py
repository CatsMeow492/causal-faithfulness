"""
Unit tests for baseline generation system.
Tests baseline generation strategies for different modalities.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock

from src.baseline import (
    BaselineStrategy,
    BaseBaselineGenerator,
    TextBaselineGenerator,
    TabularBaselineGenerator,
    ImageBaselineGenerator,
    BaselineGenerator,
    create_baseline_generator
)
from src.masking import DataModality


class TestBaselineStrategy:
    """Test BaselineStrategy enum."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        assert BaselineStrategy.RANDOM.value == "random"
        assert BaselineStrategy.MEAN.value == "mean"
        assert BaselineStrategy.ZERO.value == "zero"
        assert BaselineStrategy.GAUSSIAN.value == "gaussian"
        assert BaselineStrategy.UNIFORM.value == "uniform"
        assert BaselineStrategy.PERMUTE.value == "permute"


class TestTextBaselineGenerator:
    """Test TextBaselineGenerator for text data."""
    
    @pytest.fixture
    def text_generator(self):
        """Create text baseline generator for testing."""
        return TextBaselineGenerator(
            strategy=BaselineStrategy.RANDOM,
            vocab_size=1000,
            pad_token_id=0,
            special_tokens=[0, 1, 2],
            random_seed=42
        )
    
    def test_initialization(self, text_generator):
        """Test text generator initialization."""
        assert text_generator.strategy == BaselineStrategy.RANDOM
        assert text_generator.vocab_size == 1000
        assert text_generator.pad_token_id == 0
        assert text_generator.special_tokens == [0, 1, 2]
        assert len(text_generator.valid_tokens) == 997  # 1000 - 3 special tokens
    
    def test_random_baseline_tensor(self, text_generator):
        """Test random baseline generation with tensor input."""
        input_tensor = torch.tensor([10, 20, 30, 40, 50])
        
        baseline = text_generator.generate_baseline(input_tensor, batch_size=1)
        
        assert isinstance(baseline, torch.Tensor)
        assert baseline.shape == input_tensor.shape
        assert baseline.dtype == input_tensor.dtype
        
        # Should contain valid tokens (not special tokens)
        for token in baseline:
            assert token.item() not in text_generator.special_tokens
            assert 0 <= token.item() < text_generator.vocab_size
    
    def test_zero_baseline(self):
        """Test zero baseline generation."""
        generator = TextBaselineGenerator(
            strategy=BaselineStrategy.ZERO,
            pad_token_id=0,
            random_seed=42
        )
        
        input_tensor = torch.tensor([10, 20, 30])
        baseline = generator.generate_baseline(input_tensor, batch_size=1)
        
        assert torch.all(baseline == 0)
    
    def test_uniform_baseline(self, text_generator):
        """Test uniform baseline generation."""
        text_generator.strategy = BaselineStrategy.UNIFORM
        
        input_tensor = torch.tensor([10, 20, 30])
        baseline = text_generator.generate_baseline(input_tensor, batch_size=1)
        
        # Should contain tokens from full vocabulary range
        for token in baseline:
            assert 0 <= token.item() < text_generator.vocab_size
    
    def test_permute_baseline(self, text_generator):
        """Test permutation baseline generation."""
        text_generator.strategy = BaselineStrategy.PERMUTE
        
        input_tensor = torch.tensor([10, 20, 30, 40, 50])
        baseline = text_generator.generate_baseline(input_tensor, batch_size=1)
        
        # Should contain same tokens in different order
        original_sorted = torch.sort(input_tensor)[0]
        baseline_sorted = torch.sort(baseline)[0]
        assert torch.equal(original_sorted, baseline_sorted)
    
    def test_batch_generation(self, text_generator):
        """Test batch baseline generation."""
        input_tensor = torch.tensor([10, 20, 30])
        baselines = text_generator.generate_baseline(input_tensor, batch_size=3)
        
        assert isinstance(baselines, list)
        assert len(baselines) == 3
        
        for baseline in baselines:
            assert isinstance(baseline, torch.Tensor)
            assert baseline.shape == input_tensor.shape
    
    def test_dict_input(self, text_generator):
        """Test baseline generation with dictionary input."""
        input_dict = {
            'input_ids': torch.tensor([10, 20, 30]),
            'attention_mask': torch.tensor([1, 1, 1])
        }
        
        baseline = text_generator.generate_baseline(input_dict, batch_size=1)
        
        assert isinstance(baseline, dict)
        assert 'input_ids' in baseline
        assert 'attention_mask' in baseline
        
        # input_ids should be modified, attention_mask should be copied
        assert baseline['input_ids'].shape == input_dict['input_ids'].shape
        assert torch.equal(baseline['attention_mask'], input_dict['attention_mask'])
    
    def test_numpy_input(self, text_generator):
        """Test baseline generation with numpy input."""
        input_array = np.array([10, 20, 30])
        baseline = text_generator.generate_baseline(input_array, batch_size=1)
        
        assert isinstance(baseline, np.ndarray)
        assert baseline.shape == input_array.shape
        assert baseline.dtype == input_array.dtype
    
    def test_input_validation(self, text_generator):
        """Test input validation."""
        # Valid inputs
        assert text_generator.validate_input(torch.tensor([1, 2, 3]))
        assert text_generator.validate_input({'input_ids': [1, 2, 3]})
        assert text_generator.validate_input(np.array([1, 2, 3]))
        
        # Invalid inputs
        assert not text_generator.validate_input("invalid")
        assert not text_generator.validate_input({'no_input_ids': [1, 2, 3]})


class TestTabularBaselineGenerator:
    """Test TabularBaselineGenerator for tabular data."""
    
    @pytest.fixture
    def tabular_generator(self):
        """Create tabular baseline generator for testing."""
        feature_means = np.array([1.0, 2.0, 3.0, 4.0])
        feature_stds = np.array([0.1, 0.2, 0.3, 0.4])
        feature_ranges = (np.array([0.0, 1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0, 5.0]))
        
        return TabularBaselineGenerator(
            strategy=BaselineStrategy.GAUSSIAN,
            feature_means=feature_means,
            feature_stds=feature_stds,
            feature_ranges=feature_ranges,
            random_seed=42
        )
    
    def test_initialization(self, tabular_generator):
        """Test tabular generator initialization."""
        assert tabular_generator.strategy == BaselineStrategy.GAUSSIAN
        assert tabular_generator.feature_means is not None
        assert tabular_generator.feature_stds is not None
        assert tabular_generator.feature_ranges is not None
        assert len(tabular_generator.feature_means) == 4
    
    def test_zero_baseline(self):
        """Test zero baseline generation."""
        generator = TabularBaselineGenerator(
            strategy=BaselineStrategy.ZERO,
            random_seed=42
        )
        
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        baseline = generator.generate_baseline(input_tensor, batch_size=1)
        
        assert torch.all(baseline == 0.0)
        assert baseline.shape == input_tensor.shape
    
    def test_mean_baseline(self, tabular_generator):
        """Test mean baseline generation."""
        tabular_generator.strategy = BaselineStrategy.MEAN
        
        input_tensor = torch.tensor([[1.5, 2.5, 3.5, 4.5]])
        baseline = tabular_generator.generate_baseline(input_tensor, batch_size=1)
        
        # Should use provided feature means
        expected_means = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        assert torch.allclose(baseline, expected_means, atol=1e-6)
    
    def test_gaussian_baseline(self, tabular_generator):
        """Test Gaussian baseline generation."""
        input_tensor = torch.tensor([[1.5, 2.5, 3.5, 4.5]])
        baseline = tabular_generator.generate_baseline(input_tensor, batch_size=1)
        
        assert baseline.shape == input_tensor.shape
        # Values should be different from input (with high probability)
        assert not torch.allclose(baseline, input_tensor, atol=1e-3)
    
    def test_uniform_baseline(self, tabular_generator):
        """Test uniform baseline generation."""
        tabular_generator.strategy = BaselineStrategy.UNIFORM
        
        input_tensor = torch.tensor([[1.5, 2.5, 3.5, 4.5]])
        baseline = tabular_generator.generate_baseline(input_tensor, batch_size=1)
        
        # Should be within specified ranges
        min_vals, max_vals = tabular_generator.feature_ranges
        for i in range(len(min_vals)):
            assert min_vals[i] <= baseline[0, i] <= max_vals[i]
    
    def test_permute_baseline(self):
        """Test permutation baseline generation."""
        generator = TabularBaselineGenerator(
            strategy=BaselineStrategy.PERMUTE,
            random_seed=42
        )
        
        # Create batch data for permutation
        input_tensor = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        baseline = generator.generate_baseline(input_tensor, batch_size=1)
        
        # Each feature should contain same values in different order
        for feature_idx in range(input_tensor.shape[1]):
            original_feature = input_tensor[:, feature_idx].sort()[0]
            baseline_feature = baseline[:, feature_idx].sort()[0]
            assert torch.equal(original_feature, baseline_feature)
    
    def test_batch_generation(self, tabular_generator):
        """Test batch baseline generation."""
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        baselines = tabular_generator.generate_baseline(input_tensor, batch_size=3)
        
        assert isinstance(baselines, torch.Tensor)
        assert baselines.shape[0] == 3  # Batch dimension
        assert baselines.shape[1:] == input_tensor.shape  # Feature dimensions
    
    def test_numpy_input(self, tabular_generator):
        """Test baseline generation with numpy input."""
        input_array = np.array([[1.0, 2.0, 3.0, 4.0]])
        baseline = tabular_generator.generate_baseline(input_array, batch_size=1)
        
        assert isinstance(baseline, np.ndarray)
        assert baseline.shape == input_array.shape
    
    def test_input_validation(self, tabular_generator):
        """Test input validation."""
        # Valid inputs
        assert tabular_generator.validate_input(torch.randn(10, 5))
        assert tabular_generator.validate_input(np.random.randn(10, 5))
        assert tabular_generator.validate_input(torch.randn(5))  # 1D also valid
        
        # Invalid inputs
        assert not tabular_generator.validate_input("invalid")
        assert not tabular_generator.validate_input([1, 2, 3])


class TestImageBaselineGenerator:
    """Test ImageBaselineGenerator for image data."""
    
    @pytest.fixture
    def image_generator(self):
        """Create image baseline generator for testing."""
        return ImageBaselineGenerator(
            strategy=BaselineStrategy.GAUSSIAN,
            pixel_mean=0.5,
            pixel_std=0.1,
            random_seed=42
        )
    
    def test_initialization(self, image_generator):
        """Test image generator initialization."""
        assert image_generator.strategy == BaselineStrategy.GAUSSIAN
        assert image_generator.pixel_mean == 0.5
        assert image_generator.pixel_std == 0.1
    
    def test_zero_baseline(self):
        """Test zero baseline generation."""
        generator = ImageBaselineGenerator(
            strategy=BaselineStrategy.ZERO,
            random_seed=42
        )
        
        input_image = torch.randn(3, 32, 32)
        baseline = generator.generate_baseline(input_image, batch_size=1)
        
        assert torch.all(baseline == 0.0)
        assert baseline.shape == input_image.shape
    
    def test_gaussian_baseline(self, image_generator):
        """Test Gaussian baseline generation."""
        input_image = torch.randn(3, 32, 32)
        baseline = image_generator.generate_baseline(input_image, batch_size=1)
        
        assert baseline.shape == input_image.shape
        # Values should be clamped to [0, 1] range
        assert torch.all(baseline >= 0.0)
        assert torch.all(baseline <= 1.0)
        
        # Should be different from input
        assert not torch.allclose(baseline, input_image)
    
    def test_uniform_baseline(self, image_generator):
        """Test uniform baseline generation."""
        image_generator.strategy = BaselineStrategy.UNIFORM
        
        input_image = torch.randn(3, 32, 32)
        baseline = image_generator.generate_baseline(input_image, batch_size=1)
        
        # Should be in [0, 1] range
        assert torch.all(baseline >= 0.0)
        assert torch.all(baseline <= 1.0)
    
    def test_permute_baseline(self, image_generator):
        """Test permutation baseline generation."""
        image_generator.strategy = BaselineStrategy.PERMUTE
        
        input_image = torch.randn(3, 4, 4)  # Small image for testing
        baseline = image_generator.generate_baseline(input_image, batch_size=1)
        
        # Should contain same pixel values in different positions
        original_flat = input_image.flatten().sort()[0]
        baseline_flat = baseline.flatten().sort()[0]
        assert torch.allclose(original_flat, baseline_flat)
    
    def test_batch_generation(self, image_generator):
        """Test batch baseline generation."""
        input_image = torch.randn(3, 32, 32)
        baselines = image_generator.generate_baseline(input_image, batch_size=2)
        
        assert isinstance(baselines, torch.Tensor)
        assert baselines.shape == (2, 3, 32, 32)  # Batch of 2 images
    
    def test_numpy_input(self, image_generator):
        """Test baseline generation with numpy input."""
        input_array = np.random.randn(3, 32, 32)
        baseline = image_generator.generate_baseline(input_array, batch_size=1)
        
        assert isinstance(baseline, np.ndarray)
        assert baseline.shape == input_array.shape
        assert np.all(baseline >= 0.0)
        assert np.all(baseline <= 1.0)
    
    def test_input_validation(self, image_generator):
        """Test input validation."""
        # Valid inputs
        assert image_generator.validate_input(torch.randn(3, 32, 32))  # 3D
        assert image_generator.validate_input(torch.randn(1, 3, 32, 32))  # 4D
        assert image_generator.validate_input(np.random.randn(3, 32, 32))
        
        # Invalid inputs
        assert not image_generator.validate_input(torch.randn(32))  # 1D
        assert not image_generator.validate_input(torch.randn(32, 32))  # 2D
        assert not image_generator.validate_input("invalid")


class TestBaselineGenerator:
    """Test unified BaselineGenerator interface."""
    
    def test_text_modality(self):
        """Test text modality selection."""
        generator = BaselineGenerator(
            modality=DataModality.TEXT,
            strategy=BaselineStrategy.RANDOM,
            random_seed=42,
            vocab_size=1000
        )
        
        assert isinstance(generator.generator, TextBaselineGenerator)
        assert generator.modality == DataModality.TEXT
        assert generator.strategy == BaselineStrategy.RANDOM
    
    def test_tabular_modality(self):
        """Test tabular modality selection."""
        generator = BaselineGenerator(
            modality=DataModality.TABULAR,
            strategy=BaselineStrategy.GAUSSIAN,
            random_seed=42
        )
        
        assert isinstance(generator.generator, TabularBaselineGenerator)
        assert generator.modality == DataModality.TABULAR
        assert generator.strategy == BaselineStrategy.GAUSSIAN
    
    def test_image_modality(self):
        """Test image modality selection."""
        generator = BaselineGenerator(
            modality=DataModality.IMAGE,
            strategy=BaselineStrategy.GAUSSIAN,
            random_seed=42
        )
        
        assert isinstance(generator.generator, ImageBaselineGenerator)
        assert generator.modality == DataModality.IMAGE
        assert generator.strategy == BaselineStrategy.GAUSSIAN
    
    def test_unsupported_modality(self):
        """Test error handling for unsupported modality."""
        with pytest.raises(ValueError, match="Unsupported modality"):
            BaselineGenerator(
                modality="unsupported",
                strategy=BaselineStrategy.GAUSSIAN,
                random_seed=42
            )
    
    def test_generate_baseline_delegation(self):
        """Test that generate_baseline delegates to underlying generator."""
        generator = BaselineGenerator(
            modality=DataModality.TABULAR,
            strategy=BaselineStrategy.ZERO,
            random_seed=42
        )
        
        input_data = torch.tensor([[1.0, 2.0, 3.0]])
        baseline = generator.generate_baseline(input_data, batch_size=1)
        
        # Should delegate to TabularBaselineGenerator
        assert torch.all(baseline == 0.0)
        assert baseline.shape == input_data.shape
    
    def test_generate_batch(self):
        """Test batch generation for multiple inputs."""
        generator = BaselineGenerator(
            modality=DataModality.TABULAR,
            strategy=BaselineStrategy.ZERO,
            random_seed=42
        )
        
        data_batch = [
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[3.0, 4.0]]),
            torch.tensor([[5.0, 6.0]])
        ]
        
        baselines = generator.generate_batch(data_batch, samples_per_input=2)
        
        assert len(baselines) == 3
        for baseline in baselines:
            assert isinstance(baseline, torch.Tensor)
            assert baseline.shape[0] == 2  # 2 samples per input
    
    def test_input_validation_delegation(self):
        """Test that input validation delegates to underlying generator."""
        generator = BaselineGenerator(
            modality=DataModality.TABULAR,
            strategy=BaselineStrategy.ZERO,
            random_seed=42
        )
        
        # Should delegate to TabularBaselineGenerator validation
        assert generator.validate_input(torch.randn(10, 5))
        assert not generator.validate_input("invalid")
    
    def test_invalid_input_error(self):
        """Test error handling for invalid input."""
        generator = BaselineGenerator(
            modality=DataModality.TABULAR,
            strategy=BaselineStrategy.ZERO,
            random_seed=42
        )
        
        with pytest.raises(ValueError, match="Invalid input data"):
            generator.generate_baseline("invalid_input")


class TestCreateBaselineGenerator:
    """Test create_baseline_generator convenience function."""
    
    def test_string_modality(self):
        """Test creating generator with string modality."""
        generator = create_baseline_generator("text", "random", random_seed=42)
        
        assert isinstance(generator, BaselineGenerator)
        assert generator.modality == DataModality.TEXT
        assert generator.strategy == BaselineStrategy.RANDOM
    
    def test_enum_inputs(self):
        """Test creating generator with enum inputs."""
        generator = create_baseline_generator(
            DataModality.TABULAR,
            BaselineStrategy.GAUSSIAN,
            random_seed=42
        )
        
        assert isinstance(generator, BaselineGenerator)
        assert generator.modality == DataModality.TABULAR
        assert generator.strategy == BaselineStrategy.GAUSSIAN
    
    def test_default_strategies(self):
        """Test default strategy selection for each modality."""
        # Text default
        text_generator = create_baseline_generator("text", "default", random_seed=42)
        assert text_generator.strategy == BaselineStrategy.RANDOM
        
        # Tabular default
        tabular_generator = create_baseline_generator("tabular", "default", random_seed=42)
        assert tabular_generator.strategy == BaselineStrategy.GAUSSIAN
        
        # Image default
        image_generator = create_baseline_generator("image", "default", random_seed=42)
        assert image_generator.strategy == BaselineStrategy.GAUSSIAN
    
    def test_kwargs_passing(self):
        """Test that kwargs are passed to underlying generator."""
        generator = create_baseline_generator(
            "text",
            "random",
            random_seed=42,
            vocab_size=2000,
            pad_token_id=5
        )
        
        assert generator.generator.vocab_size == 2000
        assert generator.generator.pad_token_id == 5


if __name__ == "__main__":
    pytest.main([__file__])