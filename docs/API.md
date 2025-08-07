# API Documentation

## Overview

The Causal-Faithfulness Metric provides a unified, model-agnostic framework for evaluating explanation methods across different model architectures and data modalities. This document describes the core API components and their usage.

## Core Components

### FaithfulnessMetric

The main class for computing causal-faithfulness scores.

```python
from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
from src.masking import DataModality

# Initialize metric
config = FaithfulnessConfig(
    n_samples=1000,
    baseline_strategy="random",
    confidence_level=0.95,
    random_seed=42
)

metric = FaithfulnessMetric(config, modality=DataModality.TABULAR)

# Compute faithfulness score
result = metric.compute_faithfulness_score(
    model=your_model,
    explainer=your_explainer,
    data=input_data,
    target_class=None  # Auto-detect if None
)

print(f"F-score: {result.f_score:.4f}")
print(f"95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
```

### Configuration Classes

#### FaithfulnessConfig

Configuration for faithfulness metric computation.

**Parameters:**
- `n_samples` (int, default=1000): Number of Monte-Carlo samples for expectation computation
- `baseline_strategy` (str, default="random"): Strategy for generating x_rand baselines
  - `"random"`: Random sampling from appropriate distribution
  - `"mean"`: Use feature means as baseline
  - `"zero"`: Use zero values as baseline
- `masking_strategy` (str, default="pad"): Strategy for masking features
  - `"pad"`: Use padding tokens (text)
  - `"zero"`: Set features to zero
  - `"mean"`: Replace with feature means
  - `"noise"`: Add Gaussian noise
- `confidence_level` (float, default=0.95): Confidence level for statistical analysis
- `batch_size` (int, default=32): Batch size for computation (auto-adjusted for hardware)
- `random_seed` (int, default=42): Random seed for reproducibility
- `device` (torch.device, optional): Computation device (auto-detected if None)
- `numerical_epsilon` (float, default=1e-8): Epsilon for numerical stability
- `enable_streaming` (bool, default=False): Enable streaming computation for large datasets

#### FaithfulnessResult

Result container for faithfulness metric computation.

**Attributes:**
- `f_score` (float): The computed F(E) score ∈ [0,1]
- `confidence_interval` (Tuple[float, float]): Bootstrap confidence interval
- `n_samples` (int): Number of Monte-Carlo samples used
- `baseline_performance` (float): Mean baseline difference E[|f(x) - f(x_rand)|]
- `explained_performance` (float): Mean explained difference E[|f(x) - f(x_∖E)|]
- `statistical_significance` (bool): Whether difference is statistically significant
- `p_value` (float): P-value from paired t-test
- `computation_metrics` (Dict[str, float]): Performance metrics (time, memory, etc.)

### Explanation Wrappers

#### SHAPWrapper

Wrapper for SHAP explanation methods.

```python
from src.explainers import SHAPWrapper

# Initialize SHAP explainer
shap_explainer = SHAPWrapper(
    explainer_type="kernel",  # "kernel", "tree", "deep"
    n_samples=1000,
    random_seed=42
)

# Generate explanation
attribution = shap_explainer.explain(
    model=your_model,
    input_data=input_data,
    target_class=None,  # Auto-detect if None
    background_data=background_samples  # Required for KernelSHAP
)
```

**Parameters:**
- `explainer_type` (str): Type of SHAP explainer
  - `"kernel"`: KernelSHAP (model-agnostic, requires background data)
  - `"tree"`: TreeSHAP (for tree-based models)
  - `"deep"`: DeepSHAP (for neural networks, requires background data)
- `n_samples` (int, default=1000): Number of samples for KernelSHAP
- `random_seed` (int, default=42): Random seed for reproducibility

#### IntegratedGradientsWrapper

Wrapper for Integrated Gradients explanation method.

```python
from src.explainers import IntegratedGradientsWrapper

# Initialize IG explainer
ig_explainer = IntegratedGradientsWrapper(
    n_steps=50,
    baseline_strategy="zero",  # "zero", "random", "mean"
    random_seed=42
)

# Generate explanation
attribution = ig_explainer.explain(
    model=your_model,
    input_data=input_data,
    target_class=None,
    baseline_data=None  # Optional custom baseline
)
```

**Parameters:**
- `n_steps` (int, default=50): Number of integration steps
- `baseline_strategy` (str, default="zero"): Strategy for baseline generation
- `random_seed` (int, default=42): Random seed for reproducibility
- `internal_batch_size` (int, default=32): Batch size for gradient computation

#### LIMEWrapper

Wrapper for LIME explanation method.

```python
from src.explainers import LIMEWrapper

# Initialize LIME explainer
lime_explainer = LIMEWrapper(
    n_samples=500,
    modality="tabular",  # "tabular", "text", "image"
    random_seed=42
)

# Generate explanation
attribution = lime_explainer.explain(
    model=your_model,
    input_data=input_data,
    target_class=None,
    training_data=training_samples,  # Required for tabular data
    feature_names=feature_names  # Optional
)
```

**Parameters:**
- `n_samples` (int, default=500): Number of perturbation samples
- `modality` (str, default="tabular"): Data modality
- `random_seed` (int, default=42): Random seed for reproducibility

#### RandomExplainer

Random baseline explainer for sanity checking.

```python
from src.explainers import RandomExplainer

# Initialize random explainer
random_explainer = RandomExplainer(
    random_seed=42,
    distribution="uniform",  # "uniform", "gaussian"
    scale=1.0
)

# Generate random explanation
attribution = random_explainer.explain(
    model=your_model,
    input_data=input_data
)
```

### Feature Masking

#### FeatureMasker

Unified interface for feature masking across modalities.

```python
from src.masking import FeatureMasker, DataModality, MaskingStrategy

# Create masker
masker = FeatureMasker(
    modality=DataModality.TEXT,
    strategy=MaskingStrategy.PAD,
    random_seed=42,
    pad_token_id=0,  # Text-specific parameter
    vocab_size=30000
)

# Mask features
masked_data = masker.mask_features(
    data=input_data,
    feature_indices=[1, 3, 5],  # Features to keep/mask
    mask_explained=False  # False: mask non-explained, True: mask explained
)
```

**Modalities:**
- `DataModality.TEXT`: Text/token data
- `DataModality.TABULAR`: Structured/tabular data  
- `DataModality.IMAGE`: Image data

**Masking Strategies:**
- `MaskingStrategy.PAD`: Use padding tokens (text)
- `MaskingStrategy.MASK`: Use mask tokens (text)
- `MaskingStrategy.ZERO`: Set to zero
- `MaskingStrategy.MEAN`: Replace with means
- `MaskingStrategy.NOISE`: Add noise
- `MaskingStrategy.BLUR`: Blur regions (images)

### Baseline Generation

#### BaselineGenerator

Unified interface for generating x_rand baselines.

```python
from src.baseline import BaselineGenerator, BaselineStrategy

# Create generator
generator = BaselineGenerator(
    modality=DataModality.TABULAR,
    strategy=BaselineStrategy.GAUSSIAN,
    random_seed=42,
    feature_means=feature_means,  # Optional statistics
    feature_stds=feature_stds
)

# Generate baseline
baseline_data = generator.generate_baseline(
    data=input_data,
    batch_size=10  # Number of baseline samples
)
```

**Baseline Strategies:**
- `BaselineStrategy.RANDOM`: Random sampling
- `BaselineStrategy.GAUSSIAN`: Gaussian noise
- `BaselineStrategy.UNIFORM`: Uniform distribution
- `BaselineStrategy.ZERO`: Zero values
- `BaselineStrategy.MEAN`: Feature means
- `BaselineStrategy.PERMUTE`: Permute values

### Hardware Configuration

#### HardwareConfig

Automatic hardware detection and optimization.

```python
from src.config import HardwareConfig, get_device, get_batch_size

# Get system information
config = HardwareConfig()
print(f"Optimal device: {config.device}")
print(f"MPS support: {config.supports_mps}")

# Get optimal settings
device = get_device()
batch_size = get_batch_size(default=32, memory_factor=0.8)

# Memory-efficient computation
from src.config import memory_efficient_context

with memory_efficient_context():
    # Your computation here
    result = model(data)
```

## Convenience Functions

### compute_faithfulness_score

High-level function for quick faithfulness computation.

```python
from src.faithfulness import compute_faithfulness_score, FaithfulnessConfig

# Quick computation with defaults
result = compute_faithfulness_score(
    model=your_model,
    explainer=your_explainer,
    data=input_data,
    config=None,  # Uses default config
    target_class=None
)
```

### create_masker

Factory function for creating maskers.

```python
from src.masking import create_masker

# Create with string parameters
masker = create_masker(
    modality="text",
    strategy="pad",  # or "default" for modality-specific default
    random_seed=42,
    pad_token_id=0
)
```

### create_baseline_generator

Factory function for creating baseline generators.

```python
from src.baseline import create_baseline_generator

# Create with string parameters
generator = create_baseline_generator(
    modality="tabular",
    strategy="gaussian",  # or "default"
    random_seed=42
)
```

## Error Handling

The API includes comprehensive error handling with graceful degradation:

- **Memory Management**: Automatic batch size reduction on OOM errors
- **Hardware Fallbacks**: CPU fallback when GPU operations fail
- **Numerical Stability**: Epsilon handling and gradient clipping
- **Input Validation**: Type checking and format validation
- **Computation Retries**: Automatic retry with different parameters

## Performance Optimization

### Hardware-Aware Computation

- **Apple Silicon**: Optimized MPS support with CPU fallbacks
- **CUDA**: GPU acceleration when available
- **Memory Management**: Automatic memory cleanup and monitoring
- **Batch Processing**: Adaptive batch sizing based on available memory

### Streaming Computation

For large datasets, enable streaming computation:

```python
config = FaithfulnessConfig(
    enable_streaming=True,
    batch_size=16  # Smaller batches for streaming
)
```

### Computation Limits

Configure computational limits for robustness:

```python
from src.robust_computation import ComputationLimits

limits = ComputationLimits(
    max_memory_gb=8.0,
    max_computation_time=3600,  # 1 hour
    numerical_epsilon=1e-8
)

config = FaithfulnessConfig(computation_limits=limits)
```

## Examples

See the `examples/` directory for complete usage examples:

- `examples/basic_usage.py`: Basic faithfulness computation
- `examples/multi_explainer_comparison.py`: Comparing multiple explainers
- `examples/text_classification.py`: Text classification with BERT
- `examples/tabular_data.py`: Tabular data with various explainers
- `examples/hardware_optimization.py`: Hardware-specific optimizations

## Type Hints

All public functions include comprehensive type hints for better IDE support and type checking:

```python
from typing import Union, Optional, Dict, List, Tuple, Callable
import torch
import numpy as np

def compute_faithfulness_score(
    model: Callable[[Union[torch.Tensor, np.ndarray]], torch.Tensor],
    explainer: Callable,
    data: Union[torch.Tensor, np.ndarray, Dict],
    config: Optional[FaithfulnessConfig] = None,
    target_class: Optional[int] = None
) -> FaithfulnessResult:
    ...
```