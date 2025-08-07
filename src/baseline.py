"""
Baseline generation system for x_rand baseline creation.
Implements random, mean, and zero baseline strategies per modality with batch processing.
"""

import numpy as np
import torch
from typing import Union, Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from enum import Enum
import warnings

from .masking import DataModality


class BaselineStrategy(Enum):
    """Enumeration of baseline generation strategies."""
    RANDOM = "random"
    MEAN = "mean"
    ZERO = "zero"
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    PERMUTE = "permute"


class BaseBaselineGenerator(ABC):
    """Abstract base class for baseline generators."""
    
    def __init__(self, strategy: BaselineStrategy, random_seed: int = 42):
        """
        Initialize the baseline generator.
        
        Args:
            strategy: Baseline generation strategy
            random_seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.rng = np.random.RandomState(random_seed)
        torch.manual_seed(random_seed)
    
    @abstractmethod
    def generate_baseline(
        self,
        data: Union[torch.Tensor, np.ndarray, Dict],
        batch_size: int = 1
    ) -> Union[torch.Tensor, np.ndarray, Dict, List]:
        """
        Generate baseline data.
        
        Args:
            data: Original data to generate baseline from
            batch_size: Number of baseline samples to generate
            
        Returns:
            Generated baseline data (single sample or batch)
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: Union[torch.Tensor, np.ndarray, Dict]) -> bool:
        """Validate that input data is compatible with this generator."""
        pass


class TextBaselineGenerator(BaseBaselineGenerator):
    """Baseline generator for text data."""
    
    def __init__(
        self,
        strategy: BaselineStrategy = BaselineStrategy.RANDOM,
        vocab_size: Optional[int] = None,
        pad_token_id: int = 0,
        special_tokens: Optional[List[int]] = None,
        random_seed: int = 42
    ):
        """
        Initialize text baseline generator.
        
        Args:
            strategy: Baseline generation strategy
            vocab_size: Vocabulary size for random token generation
            pad_token_id: Padding token ID
            special_tokens: List of special token IDs to avoid
            random_seed: Random seed for reproducibility
        """
        super().__init__(strategy, random_seed)
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.special_tokens = special_tokens or []
        
        # Create valid token range (excluding special tokens)
        if vocab_size:
            self.valid_tokens = [
                i for i in range(vocab_size) 
                if i not in self.special_tokens
            ]
        else:
            self.valid_tokens = None
    
    def generate_baseline(
        self,
        data: Union[torch.Tensor, Dict, np.ndarray],
        batch_size: int = 1
    ) -> Union[torch.Tensor, Dict, List]:
        """Generate text baseline data."""
        if isinstance(data, dict):
            return self._generate_dict_baseline(data, batch_size)
        elif isinstance(data, torch.Tensor):
            return self._generate_tensor_baseline(data, batch_size)
        elif isinstance(data, np.ndarray):
            return self._generate_array_baseline(data, batch_size)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _generate_dict_baseline(self, data: Dict, batch_size: int) -> Union[Dict, List[Dict]]:
        """Generate baseline for dictionary input (e.g., tokenizer output)."""
        if 'input_ids' not in data:
            raise ValueError("Dictionary must contain 'input_ids' key")
        
        input_ids = data['input_ids']
        
        if isinstance(input_ids, torch.Tensor):
            baseline_input_ids = self._generate_tensor_baseline(input_ids, batch_size)
        else:
            baseline_input_ids = self._generate_array_baseline(
                np.array(input_ids), batch_size
            )
        
        if batch_size == 1:
            baseline_data = data.copy()
            baseline_data['input_ids'] = baseline_input_ids
            return baseline_data
        else:
            # Return list of baseline dictionaries
            baselines = []
            for i in range(batch_size):
                baseline_data = data.copy()
                if isinstance(baseline_input_ids, list):
                    baseline_data['input_ids'] = baseline_input_ids[i]
                else:
                    baseline_data['input_ids'] = baseline_input_ids[i]
                baselines.append(baseline_data)
            return baselines
    
    def _generate_tensor_baseline(
        self, 
        tensor: torch.Tensor, 
        batch_size: int
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate baseline for PyTorch tensor."""
        if batch_size == 1:
            return self._generate_single_tensor_baseline(tensor)
        else:
            baselines = []
            for _ in range(batch_size):
                baselines.append(self._generate_single_tensor_baseline(tensor))
            return baselines
    
    def _generate_single_tensor_baseline(self, tensor: torch.Tensor) -> torch.Tensor:
        """Generate a single baseline tensor."""
        if self.strategy == BaselineStrategy.RANDOM:
            if self.valid_tokens:
                # Sample from valid tokens
                random_indices = self.rng.choice(
                    len(self.valid_tokens), 
                    size=tensor.shape, 
                    replace=True
                )
                baseline_tokens = [self.valid_tokens[i] for i in random_indices.flatten()]
                baseline_tensor = torch.tensor(baseline_tokens, dtype=tensor.dtype, device=tensor.device)
                baseline_tensor = baseline_tensor.reshape(tensor.shape)
            else:
                warnings.warn("vocab_size not provided, using random integers")
                baseline_tensor = torch.randint_like(tensor, 0, 1000)
        
        elif self.strategy == BaselineStrategy.ZERO:
            baseline_tensor = torch.zeros_like(tensor)
        
        elif self.strategy == BaselineStrategy.UNIFORM:
            if self.vocab_size:
                baseline_tensor = torch.randint_like(tensor, 0, self.vocab_size)
            else:
                baseline_tensor = torch.randint_like(tensor, 0, 1000)
        
        elif self.strategy == BaselineStrategy.PERMUTE:
            # Permute the original tokens
            flat_tensor = tensor.flatten()
            perm_indices = torch.randperm(len(flat_tensor))
            baseline_tensor = flat_tensor[perm_indices].reshape(tensor.shape)
        
        else:
            # Default to padding tokens
            baseline_tensor = torch.full_like(tensor, self.pad_token_id)
        
        return baseline_tensor
    
    def _generate_array_baseline(
        self, 
        array: np.ndarray, 
        batch_size: int
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate baseline for numpy array."""
        if batch_size == 1:
            return self._generate_single_array_baseline(array)
        else:
            baselines = []
            for _ in range(batch_size):
                baselines.append(self._generate_single_array_baseline(array))
            return baselines
    
    def _generate_single_array_baseline(self, array: np.ndarray) -> np.ndarray:
        """Generate a single baseline array."""
        if self.strategy == BaselineStrategy.RANDOM:
            if self.valid_tokens:
                baseline_tokens = self.rng.choice(self.valid_tokens, size=array.shape)
                baseline_array = baseline_tokens.astype(array.dtype)
            else:
                baseline_array = self.rng.randint(0, 1000, size=array.shape).astype(array.dtype)
        
        elif self.strategy == BaselineStrategy.ZERO:
            baseline_array = np.zeros_like(array)
        
        elif self.strategy == BaselineStrategy.UNIFORM:
            if self.vocab_size:
                baseline_array = self.rng.randint(0, self.vocab_size, size=array.shape).astype(array.dtype)
            else:
                baseline_array = self.rng.randint(0, 1000, size=array.shape).astype(array.dtype)
        
        elif self.strategy == BaselineStrategy.PERMUTE:
            baseline_array = self.rng.permutation(array.flatten()).reshape(array.shape)
        
        else:
            baseline_array = np.full_like(array, self.pad_token_id)
        
        return baseline_array
    
    def validate_input(self, data: Union[torch.Tensor, np.ndarray, Dict]) -> bool:
        """Validate text input data."""
        if isinstance(data, dict):
            return 'input_ids' in data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            return len(data.shape) >= 1
        return False


class TabularBaselineGenerator(BaseBaselineGenerator):
    """Baseline generator for tabular/structured data."""
    
    def __init__(
        self,
        strategy: BaselineStrategy = BaselineStrategy.GAUSSIAN,
        feature_means: Optional[np.ndarray] = None,
        feature_stds: Optional[np.ndarray] = None,
        feature_ranges: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        random_seed: int = 42
    ):
        """
        Initialize tabular baseline generator.
        
        Args:
            strategy: Baseline generation strategy
            feature_means: Mean values for each feature
            feature_stds: Standard deviations for each feature
            feature_ranges: (min, max) values for each feature
            random_seed: Random seed for reproducibility
        """
        super().__init__(strategy, random_seed)
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.feature_ranges = feature_ranges
    
    def generate_baseline(
        self,
        data: Union[torch.Tensor, np.ndarray],
        batch_size: int = 1
    ) -> Union[torch.Tensor, np.ndarray, List]:
        """Generate tabular baseline data."""
        if isinstance(data, torch.Tensor):
            return self._generate_tensor_baseline(data, batch_size)
        else:
            return self._generate_array_baseline(np.array(data), batch_size)
    
    def _generate_tensor_baseline(
        self, 
        tensor: torch.Tensor, 
        batch_size: int
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate baseline for PyTorch tensor."""
        if batch_size == 1:
            return self._generate_single_tensor_baseline(tensor)
        else:
            # Generate batch of baselines
            single_baseline = self._generate_single_tensor_baseline(tensor)
            if len(tensor.shape) == 1:
                # Single sample input
                baselines = torch.stack([
                    self._generate_single_tensor_baseline(tensor) 
                    for _ in range(batch_size)
                ])
            else:
                # Batch input - generate baseline for each sample
                baselines = []
                for i in range(tensor.shape[0]):
                    sample_baselines = torch.stack([
                        self._generate_single_tensor_baseline(tensor[i:i+1])
                        for _ in range(batch_size)
                    ])
                    baselines.append(sample_baselines)
                baselines = torch.cat(baselines, dim=1)
            
            return baselines
    
    def _generate_single_tensor_baseline(self, tensor: torch.Tensor) -> torch.Tensor:
        """Generate a single baseline tensor."""
        if self.strategy == BaselineStrategy.ZERO:
            baseline_tensor = torch.zeros_like(tensor)
        
        elif self.strategy == BaselineStrategy.MEAN:
            if self.feature_means is not None:
                means_tensor = torch.from_numpy(self.feature_means).to(tensor.device, dtype=tensor.dtype)
                if len(tensor.shape) == 1:
                    baseline_tensor = means_tensor
                else:
                    baseline_tensor = means_tensor.expand_as(tensor)
            else:
                # Compute means from the data
                if len(tensor.shape) > 1:
                    feature_means = torch.mean(tensor, dim=0, keepdim=True)
                    baseline_tensor = feature_means.expand_as(tensor)
                else:
                    baseline_tensor = torch.mean(tensor).expand_as(tensor)
        
        elif self.strategy == BaselineStrategy.GAUSSIAN:
            if self.feature_means is not None and self.feature_stds is not None:
                means_tensor = torch.from_numpy(self.feature_means).to(tensor.device, dtype=tensor.dtype)
                stds_tensor = torch.from_numpy(self.feature_stds).to(tensor.device, dtype=tensor.dtype)
                
                noise = torch.randn_like(tensor)
                baseline_tensor = means_tensor + noise * stds_tensor
            else:
                # Use unit Gaussian
                baseline_tensor = torch.randn_like(tensor)
        
        elif self.strategy == BaselineStrategy.UNIFORM:
            if self.feature_ranges is not None:
                min_vals, max_vals = self.feature_ranges
                min_tensor = torch.from_numpy(min_vals).to(tensor.device, dtype=tensor.dtype)
                max_tensor = torch.from_numpy(max_vals).to(tensor.device, dtype=tensor.dtype)
                
                uniform_noise = torch.rand_like(tensor)
                baseline_tensor = min_tensor + uniform_noise * (max_tensor - min_tensor)
            else:
                # Use [0, 1] uniform
                baseline_tensor = torch.rand_like(tensor)
        
        elif self.strategy == BaselineStrategy.PERMUTE:
            if len(tensor.shape) > 1:
                # Permute each feature independently
                baseline_tensor = tensor.clone()
                for feature_idx in range(tensor.shape[-1]):
                    perm_indices = torch.randperm(tensor.shape[0])
                    baseline_tensor[:, feature_idx] = tensor[perm_indices, feature_idx]
            else:
                # Permute the single sample
                baseline_tensor = tensor[torch.randperm(len(tensor))]
        
        else:
            # Default to Gaussian noise
            baseline_tensor = torch.randn_like(tensor)
        
        return baseline_tensor
    
    def _generate_array_baseline(
        self, 
        array: np.ndarray, 
        batch_size: int
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate baseline for numpy array."""
        if batch_size == 1:
            return self._generate_single_array_baseline(array)
        else:
            baselines = []
            for _ in range(batch_size):
                baselines.append(self._generate_single_array_baseline(array))
            return np.array(baselines) if len(baselines) > 1 else baselines
    
    def _generate_single_array_baseline(self, array: np.ndarray) -> np.ndarray:
        """Generate a single baseline array."""
        if self.strategy == BaselineStrategy.ZERO:
            baseline_array = np.zeros_like(array)
        
        elif self.strategy == BaselineStrategy.MEAN:
            if self.feature_means is not None:
                if len(array.shape) == 1:
                    baseline_array = self.feature_means.copy()
                else:
                    baseline_array = np.tile(self.feature_means, (array.shape[0], 1))
            else:
                if len(array.shape) > 1:
                    feature_means = np.mean(array, axis=0, keepdims=True)
                    baseline_array = np.tile(feature_means, (array.shape[0], 1))
                else:
                    baseline_array = np.full_like(array, np.mean(array))
        
        elif self.strategy == BaselineStrategy.GAUSSIAN:
            if self.feature_means is not None and self.feature_stds is not None:
                noise = self.rng.randn(*array.shape)
                baseline_array = self.feature_means + noise * self.feature_stds
            else:
                baseline_array = self.rng.randn(*array.shape)
        
        elif self.strategy == BaselineStrategy.UNIFORM:
            if self.feature_ranges is not None:
                min_vals, max_vals = self.feature_ranges
                uniform_noise = self.rng.rand(*array.shape)
                baseline_array = min_vals + uniform_noise * (max_vals - min_vals)
            else:
                baseline_array = self.rng.rand(*array.shape)
        
        elif self.strategy == BaselineStrategy.PERMUTE:
            if len(array.shape) > 1:
                baseline_array = array.copy()
                for feature_idx in range(array.shape[-1]):
                    perm_indices = self.rng.permutation(array.shape[0])
                    baseline_array[:, feature_idx] = array[perm_indices, feature_idx]
            else:
                baseline_array = self.rng.permutation(array)
        
        else:
            baseline_array = self.rng.randn(*array.shape)
        
        return baseline_array
    
    def validate_input(self, data: Union[torch.Tensor, np.ndarray]) -> bool:
        """Validate tabular input data."""
        return isinstance(data, (torch.Tensor, np.ndarray)) and len(data.shape) >= 1


class ImageBaselineGenerator(BaseBaselineGenerator):
    """Baseline generator for image data."""
    
    def __init__(
        self,
        strategy: BaselineStrategy = BaselineStrategy.GAUSSIAN,
        pixel_mean: float = 0.5,
        pixel_std: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize image baseline generator.
        
        Args:
            strategy: Baseline generation strategy
            pixel_mean: Mean pixel value for Gaussian noise
            pixel_std: Standard deviation for Gaussian noise
            random_seed: Random seed for reproducibility
        """
        super().__init__(strategy, random_seed)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
    
    def generate_baseline(
        self,
        data: Union[torch.Tensor, np.ndarray],
        batch_size: int = 1
    ) -> Union[torch.Tensor, np.ndarray, List]:
        """Generate image baseline data."""
        if isinstance(data, torch.Tensor):
            return self._generate_tensor_baseline(data, batch_size)
        else:
            return self._generate_array_baseline(np.array(data), batch_size)
    
    def _generate_tensor_baseline(
        self, 
        tensor: torch.Tensor, 
        batch_size: int
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate baseline for PyTorch tensor."""
        if batch_size == 1:
            return self._generate_single_tensor_baseline(tensor)
        else:
            baselines = []
            for _ in range(batch_size):
                baselines.append(self._generate_single_tensor_baseline(tensor))
            return torch.stack(baselines) if len(baselines) > 1 else baselines
    
    def _generate_single_tensor_baseline(self, tensor: torch.Tensor) -> torch.Tensor:
        """Generate a single baseline tensor."""
        if self.strategy == BaselineStrategy.ZERO:
            baseline_tensor = torch.zeros_like(tensor)
        
        elif self.strategy == BaselineStrategy.GAUSSIAN:
            noise = torch.randn_like(tensor) * self.pixel_std + self.pixel_mean
            baseline_tensor = torch.clamp(noise, 0.0, 1.0)  # Clamp to valid pixel range
        
        elif self.strategy == BaselineStrategy.UNIFORM:
            baseline_tensor = torch.rand_like(tensor)
        
        elif self.strategy == BaselineStrategy.PERMUTE:
            # Permute pixels
            flat_tensor = tensor.flatten()
            perm_indices = torch.randperm(len(flat_tensor))
            baseline_tensor = flat_tensor[perm_indices].reshape(tensor.shape)
        
        else:
            # Default to Gaussian noise
            baseline_tensor = torch.randn_like(tensor) * self.pixel_std + self.pixel_mean
            baseline_tensor = torch.clamp(baseline_tensor, 0.0, 1.0)
        
        return baseline_tensor
    
    def _generate_array_baseline(
        self, 
        array: np.ndarray, 
        batch_size: int
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate baseline for numpy array."""
        if batch_size == 1:
            return self._generate_single_array_baseline(array)
        else:
            baselines = []
            for _ in range(batch_size):
                baselines.append(self._generate_single_array_baseline(array))
            return np.array(baselines) if len(baselines) > 1 else baselines
    
    def _generate_single_array_baseline(self, array: np.ndarray) -> np.ndarray:
        """Generate a single baseline array."""
        if self.strategy == BaselineStrategy.ZERO:
            baseline_array = np.zeros_like(array)
        
        elif self.strategy == BaselineStrategy.GAUSSIAN:
            noise = self.rng.randn(*array.shape) * self.pixel_std + self.pixel_mean
            baseline_array = np.clip(noise, 0.0, 1.0)
        
        elif self.strategy == BaselineStrategy.UNIFORM:
            baseline_array = self.rng.rand(*array.shape)
        
        elif self.strategy == BaselineStrategy.PERMUTE:
            baseline_array = self.rng.permutation(array.flatten()).reshape(array.shape)
        
        else:
            baseline_array = self.rng.randn(*array.shape) * self.pixel_std + self.pixel_mean
            baseline_array = np.clip(baseline_array, 0.0, 1.0)
        
        return baseline_array
    
    def validate_input(self, data: Union[torch.Tensor, np.ndarray]) -> bool:
        """Validate image input data."""
        return isinstance(data, (torch.Tensor, np.ndarray)) and len(data.shape) in [3, 4]


class BaselineGenerator:
    """
    Main baseline generation interface that automatically selects appropriate generator
    based on data modality and provides unified API with batch processing.
    """
    
    def __init__(
        self,
        modality: DataModality,
        strategy: BaselineStrategy,
        random_seed: int = 42,
        **kwargs
    ):
        """
        Initialize baseline generator.
        
        Args:
            modality: Data modality (TEXT, TABULAR, IMAGE)
            strategy: Baseline generation strategy
            random_seed: Random seed for reproducibility
            **kwargs: Additional arguments for specific generators
        """
        self.modality = modality
        self.strategy = strategy
        self.random_seed = random_seed
        
        # Create appropriate generator based on modality
        if modality == DataModality.TEXT:
            self.generator = TextBaselineGenerator(strategy, random_seed=random_seed, **kwargs)
        elif modality == DataModality.TABULAR:
            self.generator = TabularBaselineGenerator(strategy, random_seed=random_seed, **kwargs)
        elif modality == DataModality.IMAGE:
            self.generator = ImageBaselineGenerator(strategy, random_seed=random_seed, **kwargs)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def generate_baseline(
        self,
        data: Union[torch.Tensor, np.ndarray, Dict],
        batch_size: int = 1
    ) -> Union[torch.Tensor, np.ndarray, Dict, List]:
        """
        Generate baseline data with batch processing support.
        
        Args:
            data: Original data to generate baseline from
            batch_size: Number of baseline samples to generate
            
        Returns:
            Generated baseline data (single sample or batch)
        """
        # Validate input
        if not self.generator.validate_input(data):
            raise ValueError(f"Invalid input data for {self.modality} modality")
        
        return self.generator.generate_baseline(data, batch_size)
    
    def generate_batch(
        self,
        data_batch: List[Union[torch.Tensor, np.ndarray, Dict]],
        samples_per_input: int = 1
    ) -> List[Union[torch.Tensor, np.ndarray, Dict, List]]:
        """
        Generate baselines for a batch of inputs efficiently.
        
        Args:
            data_batch: List of input data samples
            samples_per_input: Number of baseline samples per input
            
        Returns:
            List of baseline data for each input
        """
        baselines = []
        for data in data_batch:
            baseline = self.generate_baseline(data, samples_per_input)
            baselines.append(baseline)
        
        return baselines
    
    def validate_input(self, data: Union[torch.Tensor, np.ndarray, Dict]) -> bool:
        """Validate input data compatibility."""
        return self.generator.validate_input(data)


def create_baseline_generator(
    modality: Union[DataModality, str],
    strategy: Union[BaselineStrategy, str] = "default",
    random_seed: int = 42,
    **kwargs
) -> BaselineGenerator:
    """
    Convenience function to create a baseline generator.
    
    Args:
        modality: Data modality ("text", "tabular", "image")
        strategy: Baseline strategy ("random", "gaussian", "zero", etc.)
        random_seed: Random seed for reproducibility
        **kwargs: Additional arguments for specific generators
        
    Returns:
        Configured BaselineGenerator instance
    """
    # Convert string inputs to enums
    if isinstance(modality, str):
        modality = DataModality(modality.lower())
    
    if isinstance(strategy, str):
        if strategy == "default":
            # Use default strategies for each modality
            if modality == DataModality.TEXT:
                strategy = BaselineStrategy.RANDOM
            elif modality == DataModality.TABULAR:
                strategy = BaselineStrategy.GAUSSIAN
            elif modality == DataModality.IMAGE:
                strategy = BaselineStrategy.GAUSSIAN
        else:
            strategy = BaselineStrategy(strategy.lower())
    
    return BaselineGenerator(modality, strategy, random_seed, **kwargs)