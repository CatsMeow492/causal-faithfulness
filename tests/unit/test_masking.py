"""
Unit tests for feature masking strategies.
Tests masking strategies and baseline generation for different modalities.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock

from src.masking import (
    MaskingStrategy,
    DataModality,
    BaseMasker,
    TextMasker,
    TabularMasker,
    ImageMasker,
    FeatureMasker,
    create_masker
)


class TestMaskingStrategy:
    """Test MaskingStrategy enum."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        assert MaskingStrategy.PAD.value == "pad"
        assert MaskingStrategy.MASK.value == "mask"
        assert MaskingStrategy.UNK.value == "unk"
        assert MaskingStrategy.ZERO.value == "zero"
        assert MaskingStrategy.MEAN.value == "mean"
        assert MaskingStrategy.NOISE.value == "noise"
        assert MaskingStrategy.BLUR.value == "blur"
        assert MaskingStrategy.PERMUTE.value == "permute"


class TestDataModality:
    """Test DataModality enum."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        assert DataModality.TEXT.value == "text"
        assert DataModality.TABULAR.value == "tabular"
        assert DataModality.IMAGE.value == "image"
        assert DataModality.SEQUENCE.value == "sequence"


class TestTextMasker:
    """Test TextMasker for text data masking."""
    
    @pytest.fixture
    def text_masker(self):
        """Create text masker for testing."""
        return TextMasker(
            strategy=MaskingStrategy.PAD,
            pad_token_id=0,
            mask_token_id=103,
            unk_token_id=100,
            vocab_size=1000,
            random_seed=42
        )
    
    def test_initialization(self, text_masker):
        """Test text masker initialization."""
        assert text_masker.strategy == MaskingStrategy.PAD
        assert text_masker.pad_token_id == 0
        assert text_masker.mask_token_id == 103
        assert text_masker.unk_token_id == 100
        assert text_masker.vocab_size == 1000
    
    def test_pad_masking_tensor(self, text_masker):
        """Test PAD token masking with tensor input."""
        input_tensor = torch.tensor([1, 2, 3, 4, 5])
        feature_indices = [0, 2]  # Keep positions 0 and 2
        
        # Mask non-explained features (positions 1, 3, 4)
        masked = text_masker.mask_features(input_tensor, feature_indices, mask_explained=False)
        
        assert masked[0] == 1  # Kept
        assert masked[1] == 0  # Masked (PAD)
        assert masked[2] == 3  # Kept
        assert masked[3] == 0  # Masked (PAD)
        assert masked[4] == 0  # Masked (PAD)
    
    def test_mask_token_masking(self):
        """Test MASK token masking."""
        masker = TextMasker(
            strategy=MaskingStrategy.MASK,
            mask_token_id=103,
            random_seed=42
        )
        
        input_tensor = torch.tensor([1, 2, 3, 4, 5])
        feature_indices = [1, 3]  # Mask positions 1 and 3
        
        masked = masker.mask_features(input_tensor, feature_indices, mask_explained=True)
        
        assert masked[0] == 1  # Not masked
        assert masked[1] == 103  # Masked with [MASK]
        assert masked[2] == 3  # Not masked
        assert masked[3] == 103  # Masked with [MASK]
        assert masked[4] == 5  # Not masked
    
    def test_noise_masking(self, text_masker):
        """Test noise masking with random tokens."""
        text_masker.strategy = MaskingStrategy.NOISE
        
        input_tensor = torch.tensor([1, 2, 3, 4, 5])
        feature_indices = [0, 1]
        
        masked = text_masker.mask_features(input_tensor, feature_indices, mask_explained=True)
        
        # Masked positions should be different from original (with high probability)
        assert masked[2] == 3  # Unchanged
        assert masked[3] == 4  # Unchanged
        assert masked[4] == 5  # Unchanged
        
        # Masked positions should be valid token IDs
        assert 0 <= masked[0] < text_masker.vocab_size
        assert 0 <= masked[1] < text_masker.vocab_size
    
    def test_dict_input(self, text_masker):
        """Test masking with dictionary input (tokenizer format)."""
        input_dict = {
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1])
        }
        feature_indices = [0, 2]
        
        masked = text_masker.mask_features(input_dict, feature_indices, mask_explained=False)
        
        assert isinstance(masked, dict)
        assert 'input_ids' in masked
        assert 'attention_mask' in masked
        
        # Check masking applied to input_ids
        assert masked['input_ids'][0] == 1  # Kept
        assert masked['input_ids'][1] == 0  # Masked
        assert masked['input_ids'][2] == 3  # Kept
        
        # Attention mask should be unchanged
        assert torch.equal(masked['attention_mask'], input_dict['attention_mask'])
    
    def test_numpy_array_input(self, text_masker):
        """Test masking with numpy array input."""
        input_array = np.array([1, 2, 3, 4, 5])
        feature_indices = [1, 3]
        
        masked = text_masker.mask_features(input_array, feature_indices, mask_explained=True)
        
        assert isinstance(masked, np.ndarray)
        assert masked[0] == 1  # Not masked
        assert masked[1] == 0  # Masked
        assert masked[2] == 3  # Not masked
        assert masked[3] == 0  # Masked
        assert masked[4] == 5  # Not masked
    
    def test_input_validation(self, text_masker):
        """Test input validation."""
        # Valid inputs
        assert text_masker.validate_input(torch.tensor([1, 2, 3]))
        assert text_masker.validate_input(np.array([1, 2, 3]))
        assert text_masker.validate_input({'input_ids': [1, 2, 3]})
        
        # Invalid inputs
        assert not text_masker.validate_input("invalid")
        assert not text_masker.validate_input({'no_input_ids': [1, 2, 3]})
        assert not text_masker.validate_input(123)


class TestTabularMasker:
    """Test TabularMasker for tabular data masking."""
    
    @pytest.fixture
    def tabular_masker(self):
        """Create tabular masker for testing."""
        feature_means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        feature_stds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        return TabularMasker(
            strategy=MaskingStrategy.MEAN,
            feature_means=feature_means,
            feature_stds=feature_stds,
            random_seed=42
        )
    
    def test_initialization(self, tabular_masker):
        """Test tabular masker initialization."""
        assert tabular_masker.strategy == MaskingStrategy.MEAN
        assert tabular_masker.feature_means is not None
        assert tabular_masker.feature_stds is not None
        assert len(tabular_masker.feature_means) == 5
    
    def test_zero_masking(self):
        """Test zero masking strategy."""
        masker = TabularMasker(strategy=MaskingStrategy.ZERO, random_seed=42)
        
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        feature_indices = [0, 2]  # Keep features 0 and 2
        
        masked = masker.mask_features(input_tensor, feature_indices, mask_explained=False)
        
        assert masked[0, 0] == 1.0  # Kept
        assert masked[0, 1] == 0.0  # Masked
        assert masked[0, 2] == 3.0  # Kept
        assert masked[0, 3] == 0.0  # Masked
        assert masked[0, 4] == 0.0  # Masked
    
    def test_mean_masking(self, tabular_masker):
        """Test mean masking strategy."""
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        feature_indices = [1, 3]  # Mask features 1 and 3
        
        masked = tabular_masker.mask_features(input_tensor, feature_indices, mask_explained=True)
        
        assert masked[0, 0] == 1.0  # Not masked
        assert masked[0, 1] == 2.0  # Masked with mean
        assert masked[0, 2] == 3.0  # Not masked
        assert masked[0, 3] == 4.0  # Masked with mean
        assert masked[0, 4] == 5.0  # Not masked
    
    def test_noise_masking(self, tabular_masker):
        """Test noise masking strategy."""
        tabular_masker.strategy = MaskingStrategy.NOISE
        
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        feature_indices = [0, 2]
        
        masked = tabular_masker.mask_features(input_tensor, feature_indices, mask_explained=True)
        
        # Non-masked features should be unchanged
        assert masked[0, 1] == 2.0
        assert masked[0, 3] == 4.0
        assert masked[0, 4] == 5.0
        
        # Masked features should be different (noise added)
        assert masked[0, 0] != 1.0  # Should have noise
        assert masked[0, 2] != 3.0  # Should have noise
    
    def test_permute_masking(self):
        """Test permutation masking strategy."""
        masker = TabularMasker(strategy=MaskingStrategy.PERMUTE, random_seed=42)
        
        # Create batch data for permutation
        input_tensor = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        feature_indices = [0]  # Permute feature 0
        
        masked = masker.mask_features(input_tensor, feature_indices, mask_explained=True)
        
        # Features 1 and 2 should be unchanged
        assert torch.equal(masked[:, 1], input_tensor[:, 1])
        assert torch.equal(masked[:, 2], input_tensor[:, 2])
        
        # Feature 0 should be permuted (different order)
        original_feature_0 = input_tensor[:, 0].sort()[0]
        masked_feature_0 = masked[:, 0].sort()[0]
        assert torch.equal(original_feature_0, masked_feature_0)  # Same values, different order
    
    def test_numpy_input(self, tabular_masker):
        """Test masking with numpy array input."""
        input_array = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        feature_indices = [0, 2]
        
        masked = tabular_masker.mask_features(input_array, feature_indices, mask_explained=False)
        
        assert isinstance(masked, np.ndarray)
        assert masked[0, 0] == 1.0  # Kept
        assert masked[0, 1] == 2.0  # Masked with mean
        assert masked[0, 2] == 3.0  # Kept
    
    def test_input_validation(self, tabular_masker):
        """Test input validation."""
        # Valid inputs
        assert tabular_masker.validate_input(torch.randn(10, 5))
        assert tabular_masker.validate_input(np.random.randn(10, 5))
        
        # Invalid inputs
        assert not tabular_masker.validate_input(torch.randn(5))  # 1D
        assert not tabular_masker.validate_input("invalid")
        assert not tabular_masker.validate_input([1, 2, 3])


class TestImageMasker:
    """Test ImageMasker for image data masking."""
    
    @pytest.fixture
    def image_masker(self):
        """Create image masker for testing."""
        return ImageMasker(
            strategy=MaskingStrategy.ZERO,
            blur_kernel_size=15,
            noise_std=0.1,
            random_seed=42
        )
    
    def test_initialization(self, image_masker):
        """Test image masker initialization."""
        assert image_masker.strategy == MaskingStrategy.ZERO
        assert image_masker.blur_kernel_size == 15
        assert image_masker.noise_std == 0.1
    
    def test_zero_masking_3d(self, image_masker):
        """Test zero masking with 3D image (C x H x W)."""
        input_image = torch.randn(3, 4, 4)  # 3 channels, 4x4 image
        feature_indices = [0, 1, 2]  # Mask first few pixels
        
        masked = image_masker.mask_features(input_image, feature_indices, mask_explained=True)
        
        # Check that specified pixels are zeroed
        assert masked[0, 0, 0] == 0.0  # First pixel
        assert masked[1, 0, 0] == 0.0
        assert masked[2, 0, 0] == 0.0
        
        # Other pixels should be unchanged (approximately)
        assert masked[0, 3, 3] == input_image[0, 3, 3]
    
    def test_noise_masking(self):
        """Test noise masking strategy."""
        masker = ImageMasker(strategy=MaskingStrategy.NOISE, noise_std=0.1, random_seed=42)
        
        input_image = torch.ones(3, 4, 4)  # All ones
        feature_indices = [0, 1]  # Mask first two pixels
        
        masked = masker.mask_features(input_image, feature_indices, mask_explained=True)
        
        # Masked pixels should have noise (not equal to 1.0)
        assert masked[0, 0, 0] != 1.0
        assert masked[1, 0, 0] != 1.0
        
        # Non-masked pixels should be unchanged
        assert masked[0, 3, 3] == 1.0
    
    def test_numpy_input(self, image_masker):
        """Test masking with numpy array input."""
        input_array = np.random.randn(3, 4, 4)
        feature_indices = [0, 1]
        
        masked = image_masker.mask_features(input_array, feature_indices, mask_explained=True)
        
        assert isinstance(masked, np.ndarray)
        assert masked.shape == input_array.shape
        
        # Masked pixels should be zero
        assert masked[0, 0, 0] == 0.0
    
    def test_input_validation(self, image_masker):
        """Test input validation."""
        # Valid inputs
        assert image_masker.validate_input(torch.randn(3, 32, 32))  # 3D
        assert image_masker.validate_input(torch.randn(1, 3, 32, 32))  # 4D
        assert image_masker.validate_input(np.random.randn(3, 32, 32))
        
        # Invalid inputs
        assert not image_masker.validate_input(torch.randn(32))  # 1D
        assert not image_masker.validate_input(torch.randn(32, 32))  # 2D
        assert not image_masker.validate_input("invalid")


class TestFeatureMasker:
    """Test unified FeatureMasker interface."""
    
    def test_text_modality(self):
        """Test text modality selection."""
        masker = FeatureMasker(
            modality=DataModality.TEXT,
            strategy=MaskingStrategy.PAD,
            random_seed=42,
            pad_token_id=0,
            vocab_size=1000
        )
        
        assert isinstance(masker.masker, TextMasker)
        assert masker.modality == DataModality.TEXT
        assert masker.strategy == MaskingStrategy.PAD
    
    def test_tabular_modality(self):
        """Test tabular modality selection."""
        masker = FeatureMasker(
            modality=DataModality.TABULAR,
            strategy=MaskingStrategy.MEAN,
            random_seed=42
        )
        
        assert isinstance(masker.masker, TabularMasker)
        assert masker.modality == DataModality.TABULAR
        assert masker.strategy == MaskingStrategy.MEAN
    
    def test_image_modality(self):
        """Test image modality selection."""
        masker = FeatureMasker(
            modality=DataModality.IMAGE,
            strategy=MaskingStrategy.ZERO,
            random_seed=42
        )
        
        assert isinstance(masker.masker, ImageMasker)
        assert masker.modality == DataModality.IMAGE
        assert masker.strategy == MaskingStrategy.ZERO
    
    def test_unsupported_modality(self):
        """Test error handling for unsupported modality."""
        with pytest.raises(ValueError, match="Unsupported modality"):
            FeatureMasker(
                modality="unsupported",
                strategy=MaskingStrategy.ZERO,
                random_seed=42
            )
    
    def test_mask_features_delegation(self):
        """Test that mask_features delegates to underlying masker."""
        masker = FeatureMasker(
            modality=DataModality.TABULAR,
            strategy=MaskingStrategy.ZERO,
            random_seed=42
        )
        
        input_data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        feature_indices = [0, 2]
        
        masked = masker.mask_features(input_data, feature_indices, mask_explained=False)
        
        # Should delegate to TabularMasker
        assert masked[0, 0] == 1.0  # Kept
        assert masked[0, 1] == 0.0  # Masked
        assert masked[0, 2] == 3.0  # Kept
        assert masked[0, 3] == 0.0  # Masked
        assert masked[0, 4] == 0.0  # Masked
    
    def test_input_validation_delegation(self):
        """Test that input validation delegates to underlying masker."""
        masker = FeatureMasker(
            modality=DataModality.TABULAR,
            strategy=MaskingStrategy.ZERO,
            random_seed=42
        )
        
        # Should delegate to TabularMasker validation
        assert masker.validate_input(torch.randn(10, 5))
        assert not masker.validate_input(torch.randn(5))  # 1D not valid for tabular
    
    def test_invalid_input_error(self):
        """Test error handling for invalid input."""
        masker = FeatureMasker(
            modality=DataModality.TABULAR,
            strategy=MaskingStrategy.ZERO,
            random_seed=42
        )
        
        with pytest.raises(ValueError, match="Invalid input data"):
            masker.mask_features("invalid_input", [0, 1])


class TestCreateMasker:
    """Test create_masker convenience function."""
    
    def test_string_modality(self):
        """Test creating masker with string modality."""
        masker = create_masker("text", "pad", random_seed=42)
        
        assert isinstance(masker, FeatureMasker)
        assert masker.modality == DataModality.TEXT
        assert masker.strategy == MaskingStrategy.PAD
    
    def test_enum_inputs(self):
        """Test creating masker with enum inputs."""
        masker = create_masker(
            DataModality.TABULAR,
            MaskingStrategy.MEAN,
            random_seed=42
        )
        
        assert isinstance(masker, FeatureMasker)
        assert masker.modality == DataModality.TABULAR
        assert masker.strategy == MaskingStrategy.MEAN
    
    def test_default_strategies(self):
        """Test default strategy selection for each modality."""
        # Text default
        text_masker = create_masker("text", "default", random_seed=42)
        assert text_masker.strategy == MaskingStrategy.PAD
        
        # Tabular default
        tabular_masker = create_masker("tabular", "default", random_seed=42)
        assert tabular_masker.strategy == MaskingStrategy.MEAN
        
        # Image default
        image_masker = create_masker("image", "default", random_seed=42)
        assert image_masker.strategy == MaskingStrategy.ZERO
    
    def test_kwargs_passing(self):
        """Test that kwargs are passed to underlying masker."""
        masker = create_masker(
            "text",
            "pad",
            random_seed=42,
            pad_token_id=5,
            vocab_size=2000
        )
        
        assert masker.masker.pad_token_id == 5
        assert masker.masker.vocab_size == 2000


if __name__ == "__main__":
    pytest.main([__file__])