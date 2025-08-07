"""
Feature masking engine for different data modalities.
Implements modality-specific masking strategies for faithfulness metric computation.
"""

import numpy as np
import torch
from typing import Union, List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import warnings


class MaskingStrategy(Enum):
    """Enumeration of available masking strategies."""
    PAD = "pad"  # For text: use [PAD] tokens
    MASK = "mask"  # For text: use [MASK] tokens  
    UNK = "unk"  # For text: use [UNK] tokens
    ZERO = "zero"  # Set features to zero
    MEAN = "mean"  # Replace with feature means
    NOISE = "noise"  # Add Gaussian noise
    BLUR = "blur"  # For images: blur regions
    PERMUTE = "permute"  # Permute feature values


class DataModality(Enum):
    """Enumeration of supported data modalities."""
    TEXT = "text"
    TABULAR = "tabular"
    IMAGE = "image"
    SEQUENCE = "sequence"


class BaseMasker(ABC):
    """Abstract base class for feature maskers."""
    
    def __init__(self, strategy: MaskingStrategy, random_seed: int = 42):
        """
        Initialize the masker.
        
        Args:
            strategy: Masking strategy to use
            random_seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.rng = np.random.RandomState(random_seed)
        torch.manual_seed(random_seed)
    
    @abstractmethod
    def mask_features(
        self,
        data: Union[torch.Tensor, np.ndarray, Dict],
        feature_indices: List[int],
        mask_explained: bool = False
    ) -> Union[torch.Tensor, np.ndarray, Dict]:
        """
        Mask specified features in the data.
        
        Args:
            data: Input data to mask
            feature_indices: Indices of features to mask/keep
            mask_explained: If True, mask the specified features; if False, mask all others
            
        Returns:
            Masked data with same structure as input
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: Union[torch.Tensor, np.ndarray, Dict]) -> bool:
        """Validate that input data is compatible with this masker."""
        pass


class TextMasker(BaseMasker):
    """Masker for text data using tokenized inputs."""
    
    def __init__(
        self, 
        strategy: MaskingStrategy = MaskingStrategy.PAD,
        pad_token_id: int = 0,
        mask_token_id: Optional[int] = None,
        unk_token_id: Optional[int] = None,
        vocab_size: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize text masker.
        
        Args:
            strategy: Masking strategy (PAD, MASK, UNK, or NOISE)
            pad_token_id: ID of padding token
            mask_token_id: ID of mask token (for BERT-style models)
            unk_token_id: ID of unknown token
            vocab_size: Vocabulary size for random token sampling
            random_seed: Random seed for reproducibility
        """
        super().__init__(strategy, random_seed)
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.unk_token_id = unk_token_id
        self.vocab_size = vocab_size
        
        # Validate strategy compatibility
        if strategy == MaskingStrategy.MASK and mask_token_id is None:
            raise ValueError("mask_token_id required for MASK strategy")
        if strategy == MaskingStrategy.UNK and unk_token_id is None:
            raise ValueError("unk_token_id required for UNK strategy")
    
    def mask_features(
        self,
        data: Union[torch.Tensor, Dict],
        feature_indices: List[int],
        mask_explained: bool = False
    ) -> Union[torch.Tensor, Dict]:
        """
        Mask text features (tokens).
        
        Args:
            data: Tokenized text data (tensor or dict with 'input_ids')
            feature_indices: Token positions to mask/keep
            mask_explained: If True, mask explained tokens; if False, mask others
            
        Returns:
            Masked text data
        """
        if isinstance(data, dict):
            # Handle dictionary input (e.g., from transformers tokenizer)
            masked_data = data.copy()
            input_ids = data['input_ids']
            
            if isinstance(input_ids, torch.Tensor):
                masked_input_ids = self._mask_tensor(input_ids, feature_indices, mask_explained)
            else:
                masked_input_ids = self._mask_array(
                    np.array(input_ids), feature_indices, mask_explained
                )
            
            masked_data['input_ids'] = masked_input_ids
            return masked_data
        
        elif isinstance(data, torch.Tensor):
            return self._mask_tensor(data, feature_indices, mask_explained)
        
        else:
            # Handle numpy array or list
            data_array = np.array(data)
            return self._mask_array(data_array, feature_indices, mask_explained)
    
    def _mask_tensor(
        self, 
        tensor: torch.Tensor, 
        feature_indices: List[int], 
        mask_explained: bool
    ) -> torch.Tensor:
        """Mask a PyTorch tensor."""
        masked_tensor = tensor.clone()
        
        # Determine which positions to mask
        if mask_explained:
            positions_to_mask = feature_indices
        else:
            # Mask all positions except the explained ones
            all_positions = list(range(tensor.shape[-1]))
            positions_to_mask = [pos for pos in all_positions if pos not in feature_indices]
        
        # Apply masking strategy
        if self.strategy == MaskingStrategy.PAD:
            masked_tensor[..., positions_to_mask] = self.pad_token_id
        elif self.strategy == MaskingStrategy.MASK:
            masked_tensor[..., positions_to_mask] = self.mask_token_id
        elif self.strategy == MaskingStrategy.UNK:
            masked_tensor[..., positions_to_mask] = self.unk_token_id
        elif self.strategy == MaskingStrategy.NOISE:
            if self.vocab_size:
                # Random token sampling
                random_tokens = torch.randint(
                    0, self.vocab_size, 
                    (len(positions_to_mask),), 
                    device=tensor.device
                )
                masked_tensor[..., positions_to_mask] = random_tokens
            else:
                warnings.warn("vocab_size not provided for NOISE strategy, using PAD")
                masked_tensor[..., positions_to_mask] = self.pad_token_id
        
        return masked_tensor
    
    def _mask_array(
        self, 
        array: np.ndarray, 
        feature_indices: List[int], 
        mask_explained: bool
    ) -> np.ndarray:
        """Mask a numpy array."""
        masked_array = array.copy()
        
        # Determine which positions to mask
        if mask_explained:
            positions_to_mask = feature_indices
        else:
            all_positions = list(range(array.shape[-1]))
            positions_to_mask = [pos for pos in all_positions if pos not in feature_indices]
        
        # Apply masking strategy
        if self.strategy == MaskingStrategy.PAD:
            masked_array[..., positions_to_mask] = self.pad_token_id
        elif self.strategy == MaskingStrategy.MASK:
            masked_array[..., positions_to_mask] = self.mask_token_id
        elif self.strategy == MaskingStrategy.UNK:
            masked_array[..., positions_to_mask] = self.unk_token_id
        elif self.strategy == MaskingStrategy.NOISE:
            if self.vocab_size:
                random_tokens = self.rng.randint(0, self.vocab_size, len(positions_to_mask))
                masked_array[..., positions_to_mask] = random_tokens
            else:
                warnings.warn("vocab_size not provided for NOISE strategy, using PAD")
                masked_array[..., positions_to_mask] = self.pad_token_id
        
        return masked_array
    
    def validate_input(self, data: Union[torch.Tensor, np.ndarray, Dict]) -> bool:
        """Validate text input data."""
        if isinstance(data, dict):
            return 'input_ids' in data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            return len(data.shape) >= 1  # At least 1D for token sequences
        else:
            return False


class TabularMasker(BaseMasker):
    """Masker for tabular/structured data."""
    
    def __init__(
        self,
        strategy: MaskingStrategy = MaskingStrategy.MEAN,
        feature_means: Optional[np.ndarray] = None,
        feature_stds: Optional[np.ndarray] = None,
        random_seed: int = 42
    ):
        """
        Initialize tabular masker.
        
        Args:
            strategy: Masking strategy (ZERO, MEAN, NOISE, PERMUTE)
            feature_means: Mean values for each feature (for MEAN strategy)
            feature_stds: Standard deviations for each feature (for NOISE strategy)
            random_seed: Random seed for reproducibility
        """
        super().__init__(strategy, random_seed)
        self.feature_means = feature_means
        self.feature_stds = feature_stds
    
    def mask_features(
        self,
        data: Union[torch.Tensor, np.ndarray],
        feature_indices: List[int],
        mask_explained: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Mask tabular features.
        
        Args:
            data: Tabular data (samples x features)
            feature_indices: Feature indices to mask/keep
            mask_explained: If True, mask explained features; if False, mask others
            
        Returns:
            Masked tabular data
        """
        if isinstance(data, torch.Tensor):
            return self._mask_tensor(data, feature_indices, mask_explained)
        else:
            return self._mask_array(np.array(data), feature_indices, mask_explained)
    
    def _mask_tensor(
        self, 
        tensor: torch.Tensor, 
        feature_indices: List[int], 
        mask_explained: bool
    ) -> torch.Tensor:
        """Mask a PyTorch tensor for tabular data."""
        masked_tensor = tensor.clone()
        
        # Determine which features to mask
        if mask_explained:
            features_to_mask = feature_indices
        else:
            all_features = list(range(tensor.shape[-1]))
            features_to_mask = [f for f in all_features if f not in feature_indices]
        
        # Apply masking strategy
        if self.strategy == MaskingStrategy.ZERO:
            masked_tensor[..., features_to_mask] = 0.0
        
        elif self.strategy == MaskingStrategy.MEAN:
            if self.feature_means is not None:
                means_tensor = torch.from_numpy(self.feature_means).to(tensor.device, dtype=tensor.dtype)
                masked_tensor[..., features_to_mask] = means_tensor[features_to_mask]
            else:
                # Compute means on the fly
                feature_means = torch.mean(tensor, dim=0)
                masked_tensor[..., features_to_mask] = feature_means[features_to_mask]
        
        elif self.strategy == MaskingStrategy.NOISE:
            if self.feature_stds is not None:
                stds_tensor = torch.from_numpy(self.feature_stds).to(tensor.device, dtype=tensor.dtype)
                noise = torch.randn_like(masked_tensor[..., features_to_mask]) * stds_tensor[features_to_mask]
            else:
                # Use unit Gaussian noise
                noise = torch.randn_like(masked_tensor[..., features_to_mask])
            
            if self.feature_means is not None:
                means_tensor = torch.from_numpy(self.feature_means).to(tensor.device, dtype=tensor.dtype)
                masked_tensor[..., features_to_mask] = means_tensor[features_to_mask] + noise
            else:
                masked_tensor[..., features_to_mask] = noise
        
        elif self.strategy == MaskingStrategy.PERMUTE:
            # Permute values within each feature
            for feature_idx in features_to_mask:
                perm_indices = torch.randperm(tensor.shape[0])
                masked_tensor[:, feature_idx] = tensor[perm_indices, feature_idx]
        
        return masked_tensor
    
    def _mask_array(
        self, 
        array: np.ndarray, 
        feature_indices: List[int], 
        mask_explained: bool
    ) -> np.ndarray:
        """Mask a numpy array for tabular data."""
        masked_array = array.copy()
        
        # Determine which features to mask
        if mask_explained:
            features_to_mask = feature_indices
        else:
            all_features = list(range(array.shape[-1]))
            features_to_mask = [f for f in all_features if f not in feature_indices]
        
        # Apply masking strategy
        if self.strategy == MaskingStrategy.ZERO:
            masked_array[..., features_to_mask] = 0.0
        
        elif self.strategy == MaskingStrategy.MEAN:
            if self.feature_means is not None:
                masked_array[..., features_to_mask] = self.feature_means[features_to_mask]
            else:
                # Compute means on the fly
                feature_means = np.mean(array, axis=0)
                masked_array[..., features_to_mask] = feature_means[features_to_mask]
        
        elif self.strategy == MaskingStrategy.NOISE:
            if self.feature_stds is not None:
                noise = self.rng.randn(*masked_array[..., features_to_mask].shape) * self.feature_stds[features_to_mask]
            else:
                noise = self.rng.randn(*masked_array[..., features_to_mask].shape)
            
            if self.feature_means is not None:
                masked_array[..., features_to_mask] = self.feature_means[features_to_mask] + noise
            else:
                masked_array[..., features_to_mask] = noise
        
        elif self.strategy == MaskingStrategy.PERMUTE:
            # Permute values within each feature
            for feature_idx in features_to_mask:
                perm_indices = self.rng.permutation(array.shape[0])
                masked_array[:, feature_idx] = array[perm_indices, feature_idx]
        
        return masked_array
    
    def validate_input(self, data: Union[torch.Tensor, np.ndarray]) -> bool:
        """Validate tabular input data."""
        if isinstance(data, (torch.Tensor, np.ndarray)):
            return len(data.shape) == 2  # Should be 2D: samples x features
        return False


class ImageMasker(BaseMasker):
    """Masker for image data."""
    
    def __init__(
        self,
        strategy: MaskingStrategy = MaskingStrategy.ZERO,
        blur_kernel_size: int = 15,
        noise_std: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize image masker.
        
        Args:
            strategy: Masking strategy (ZERO, NOISE, BLUR)
            blur_kernel_size: Kernel size for blur masking
            noise_std: Standard deviation for noise masking
            random_seed: Random seed for reproducibility
        """
        super().__init__(strategy, random_seed)
        self.blur_kernel_size = blur_kernel_size
        self.noise_std = noise_std
    
    def mask_features(
        self,
        data: Union[torch.Tensor, np.ndarray],
        feature_indices: List[int],
        mask_explained: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Mask image features (pixels or regions).
        
        Args:
            data: Image data (C x H x W or H x W x C)
            feature_indices: Pixel/region indices to mask/keep
            mask_explained: If True, mask explained regions; if False, mask others
            
        Returns:
            Masked image data
        """
        if isinstance(data, torch.Tensor):
            return self._mask_tensor(data, feature_indices, mask_explained)
        else:
            return self._mask_array(np.array(data), feature_indices, mask_explained)
    
    def _mask_tensor(
        self, 
        tensor: torch.Tensor, 
        feature_indices: List[int], 
        mask_explained: bool
    ) -> torch.Tensor:
        """Mask a PyTorch tensor for image data."""
        masked_tensor = tensor.clone()
        
        # Convert feature indices to spatial coordinates
        # This is a simplified implementation - in practice, you'd need
        # more sophisticated region-to-pixel mapping
        if len(tensor.shape) == 3:  # C x H x W
            h, w = tensor.shape[1], tensor.shape[2]
        elif len(tensor.shape) == 4:  # B x C x H x W
            h, w = tensor.shape[2], tensor.shape[3]
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
        
        # Create mask for pixels to modify
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        
        # Simple implementation: treat feature_indices as flattened pixel indices
        for idx in feature_indices:
            if idx < h * w:
                row, col = idx // w, idx % w
                if mask_explained:
                    mask[..., row, col] = True
                else:
                    mask[..., row, col] = False
        
        if not mask_explained:
            mask = ~mask  # Invert mask to mask non-explained pixels
        
        # Apply masking strategy
        if self.strategy == MaskingStrategy.ZERO:
            masked_tensor[mask] = 0.0
        
        elif self.strategy == MaskingStrategy.NOISE:
            noise = torch.randn_like(masked_tensor) * self.noise_std
            masked_tensor[mask] = noise[mask]
        
        elif self.strategy == MaskingStrategy.BLUR:
            # Simplified blur implementation
            # In practice, you'd use proper convolution with blur kernel
            if mask.any():
                masked_tensor[mask] = torch.mean(masked_tensor[mask])
        
        return masked_tensor
    
    def _mask_array(
        self, 
        array: np.ndarray, 
        feature_indices: List[int], 
        mask_explained: bool
    ) -> np.ndarray:
        """Mask a numpy array for image data."""
        masked_array = array.copy()
        
        # Similar implementation as tensor version
        if len(array.shape) == 3:  # H x W x C or C x H x W
            if array.shape[0] <= 3:  # C x H x W
                h, w = array.shape[1], array.shape[2]
            else:  # H x W x C
                h, w = array.shape[0], array.shape[1]
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")
        
        # Create mask for pixels to modify
        mask = np.zeros_like(array, dtype=bool)
        
        for idx in feature_indices:
            if idx < h * w:
                row, col = idx // w, idx % w
                if mask_explained:
                    mask[..., row, col] = True
                else:
                    mask[..., row, col] = False
        
        if not mask_explained:
            mask = ~mask
        
        # Apply masking strategy
        if self.strategy == MaskingStrategy.ZERO:
            masked_array[mask] = 0.0
        
        elif self.strategy == MaskingStrategy.NOISE:
            noise = self.rng.randn(*masked_array.shape) * self.noise_std
            masked_array[mask] = noise[mask]
        
        elif self.strategy == MaskingStrategy.BLUR:
            if mask.any():
                masked_array[mask] = np.mean(masked_array[mask])
        
        return masked_array
    
    def validate_input(self, data: Union[torch.Tensor, np.ndarray]) -> bool:
        """Validate image input data."""
        if isinstance(data, (torch.Tensor, np.ndarray)):
            return len(data.shape) in [3, 4]  # 3D or 4D for images
        return False


class FeatureMasker:
    """
    Main feature masking interface that automatically selects appropriate masker
    based on data modality and provides unified API.
    """
    
    def __init__(
        self,
        modality: DataModality,
        strategy: MaskingStrategy,
        random_seed: int = 42,
        **kwargs
    ):
        """
        Initialize feature masker.
        
        Args:
            modality: Data modality (TEXT, TABULAR, IMAGE)
            strategy: Masking strategy to use
            random_seed: Random seed for reproducibility
            **kwargs: Additional arguments for specific maskers
        """
        self.modality = modality
        self.strategy = strategy
        self.random_seed = random_seed
        
        # Create appropriate masker based on modality
        if modality == DataModality.TEXT:
            self.masker = TextMasker(strategy, random_seed=random_seed, **kwargs)
        elif modality == DataModality.TABULAR:
            self.masker = TabularMasker(strategy, random_seed=random_seed, **kwargs)
        elif modality == DataModality.IMAGE:
            self.masker = ImageMasker(strategy, random_seed=random_seed, **kwargs)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def mask_features(
        self,
        data: Union[torch.Tensor, np.ndarray, Dict],
        feature_indices: List[int],
        mask_explained: bool = False
    ) -> Union[torch.Tensor, np.ndarray, Dict]:
        """
        Mask features in the data.
        
        Args:
            data: Input data to mask
            feature_indices: Indices of features to mask/keep
            mask_explained: If True, mask explained features; if False, mask others
            
        Returns:
            Masked data with same structure as input
        """
        # Validate input
        if not self.masker.validate_input(data):
            raise ValueError(f"Invalid input data for {self.modality} modality")
        
        return self.masker.mask_features(data, feature_indices, mask_explained)
    
    def validate_input(self, data: Union[torch.Tensor, np.ndarray, Dict]) -> bool:
        """Validate input data compatibility."""
        return self.masker.validate_input(data)


def create_masker(
    modality: Union[DataModality, str],
    strategy: Union[MaskingStrategy, str] = "default",
    random_seed: int = 42,
    **kwargs
) -> FeatureMasker:
    """
    Convenience function to create a feature masker.
    
    Args:
        modality: Data modality ("text", "tabular", "image")
        strategy: Masking strategy ("pad", "zero", "mean", "noise", etc.)
        random_seed: Random seed for reproducibility
        **kwargs: Additional arguments for specific maskers
        
    Returns:
        Configured FeatureMasker instance
    """
    # Convert string inputs to enums
    if isinstance(modality, str):
        modality = DataModality(modality.lower())
    
    if isinstance(strategy, str):
        if strategy == "default":
            # Use default strategies for each modality
            if modality == DataModality.TEXT:
                strategy = MaskingStrategy.PAD
            elif modality == DataModality.TABULAR:
                strategy = MaskingStrategy.MEAN
            elif modality == DataModality.IMAGE:
                strategy = MaskingStrategy.ZERO
        else:
            strategy = MaskingStrategy(strategy.lower())
    
    return FeatureMasker(modality, strategy, random_seed, **kwargs)