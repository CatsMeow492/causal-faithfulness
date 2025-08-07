# Configuration Guide

## Overview

This document provides comprehensive documentation for all configuration options and hyperparameters in the Causal-Faithfulness Metric framework.

## Core Configuration

### FaithfulnessConfig

The main configuration class for faithfulness metric computation.

```python
from src.faithfulness import FaithfulnessConfig
from src.robust_computation import ComputationLimits

config = FaithfulnessConfig(
    # Monte-Carlo sampling
    n_samples=1000,
    
    # Baseline generation
    baseline_strategy="random",
    
    # Feature masking
    masking_strategy="pad",
    
    # Statistical analysis
    confidence_level=0.95,
    
    # Computation settings
    batch_size=32,
    random_seed=42,
    device=None,  # Auto-detect
    
    # Numerical stability
    numerical_epsilon=1e-8,
    
    # Advanced options
    computation_limits=None,
    enable_streaming=False
)
```

#### Monte-Carlo Sampling Parameters

**n_samples** (int, default=1000)
- Number of Monte-Carlo samples for computing expectations E[|f(x) - f(x_∖E)|] and E[|f(x) - f(x_rand)|]
- Higher values provide more accurate estimates but increase computation time
- Recommended values:
  - Quick testing: 100-500
  - Standard evaluation: 1000-2000
  - High precision: 5000+

**confidence_level** (float, default=0.95)
- Confidence level for bootstrap confidence intervals
- Must be between 0 and 1
- Common values: 0.90, 0.95, 0.99

#### Baseline Generation Parameters

**baseline_strategy** (str, default="random")
- Strategy for generating x_rand baseline samples
- Available options:
  - `"random"`: Random sampling from appropriate distribution per modality
  - `"mean"`: Use feature means as baseline
  - `"zero"`: Use zero values as baseline
  - `"gaussian"`: Gaussian noise (for tabular data)
  - `"uniform"`: Uniform distribution
  - `"permute"`: Permute feature values

**Modality-specific defaults:**
- Text: `"random"` (random token sampling)
- Tabular: `"gaussian"` (Gaussian noise around means)
- Image: `"gaussian"` (Gaussian pixel noise)

#### Feature Masking Parameters

**masking_strategy** (str, default="pad")
- Strategy for creating x_∖E (masked explained features)
- Available options:
  - `"pad"`: Use padding tokens (text)
  - `"mask"`: Use mask tokens (text, e.g., [MASK])
  - `"unk"`: Use unknown tokens (text)
  - `"zero"`: Set features to zero
  - `"mean"`: Replace with feature means
  - `"noise"`: Add Gaussian noise
  - `"blur"`: Blur regions (images)
  - `"permute"`: Permute feature values

**Modality-specific defaults:**
- Text: `"pad"` (padding tokens)
- Tabular: `"mean"` (feature means)
- Image: `"zero"` (zero pixels)

#### Computation Parameters

**batch_size** (int, default=32)
- Batch size for model queries and computation
- Automatically adjusted based on available memory
- Set to `None` for automatic detection
- Larger values are more efficient but use more memory

**random_seed** (int, default=42)
- Random seed for reproducibility
- Affects all random operations (sampling, noise generation, etc.)
- Set to `None` for non-deterministic behavior

**device** (torch.device, optional)
- Computation device (CPU, CUDA, MPS)
- If `None`, automatically detects optimal device
- Override for specific device requirements

**numerical_epsilon** (float, default=1e-8)
- Epsilon value for numerical stability
- Used in division operations to prevent divide-by-zero
- Smaller values provide higher precision but may cause instability

#### Advanced Configuration

**computation_limits** (ComputationLimits, optional)
- Limits for computational resources
- See [Computation Limits](#computation-limits) section

**enable_streaming** (bool, default=False)
- Enable streaming computation for large datasets
- Reduces memory usage at the cost of some performance
- Recommended for datasets that don't fit in memory

## Explainer Configuration

### SHAP Configuration

```python
from src.explainers import SHAPWrapper

shap_explainer = SHAPWrapper(
    explainer_type="kernel",  # "kernel", "tree", "deep"
    n_samples=1000,          # For KernelSHAP
    random_seed=42,
    # SHAP-specific parameters
    link="identity",         # Link function
    l1_reg="auto"           # L1 regularization
)
```

**explainer_type** (str, default="kernel")
- Type of SHAP explainer to use
- Options:
  - `"kernel"`: KernelSHAP (model-agnostic, requires background data)
  - `"tree"`: TreeSHAP (for tree-based models like XGBoost, Random Forest)
  - `"deep"`: DeepSHAP (for neural networks, requires background data)

**n_samples** (int, default=1000)
- Number of samples for KernelSHAP approximation
- Only used with `explainer_type="kernel"`
- Higher values provide better approximation but slower computation

**Background Data Requirements:**
- KernelSHAP and DeepSHAP require background/reference data
- Should be representative of the training distribution
- Typical size: 50-200 samples

### Integrated Gradients Configuration

```python
from src.explainers import IntegratedGradientsWrapper

ig_explainer = IntegratedGradientsWrapper(
    n_steps=50,                    # Integration steps
    baseline_strategy="zero",      # Baseline generation
    random_seed=42,
    internal_batch_size=32         # Gradient computation batch size
)
```

**n_steps** (int, default=50)
- Number of steps for path integration
- More steps provide better approximation but slower computation
- Recommended values: 20-100

**baseline_strategy** (str, default="zero")
- Strategy for generating integration baseline
- Options:
  - `"zero"`: Zero baseline (most common)
  - `"random"`: Random noise baseline
  - `"mean"`: Mean values baseline

**internal_batch_size** (int, default=32)
- Batch size for gradient computation
- Affects memory usage during gradient calculation
- Reduce if encountering memory issues

### LIME Configuration

```python
from src.explainers import LIMEWrapper

lime_explainer = LIMEWrapper(
    n_samples=500,              # Perturbation samples
    modality="tabular",         # Data modality
    random_seed=42,
    # LIME-specific parameters
    kernel_width=0.25,          # Kernel width for weighting
    feature_selection="auto"    # Feature selection method
)
```

**n_samples** (int, default=500)
- Number of perturbation samples for local approximation
- Higher values provide better local approximation
- Recommended values: 500-5000

**modality** (str, default="tabular")
- Data modality for LIME explainer
- Options: `"tabular"`, `"text"`, `"image"`
- Determines perturbation strategy

### Random Explainer Configuration

```python
from src.explainers import RandomExplainer

random_explainer = RandomExplainer(
    random_seed=42,
    distribution="uniform",     # "uniform", "gaussian"
    scale=1.0                  # Scale factor for random values
)
```

**distribution** (str, default="uniform")
- Distribution for random attribution scores
- Options:
  - `"uniform"`: Uniform distribution in [-scale, scale]
  - `"gaussian"`: Gaussian distribution with std=scale

**scale** (float, default=1.0)
- Scale factor for random values
- Controls the magnitude of random attributions

## Hardware Configuration

### Automatic Hardware Detection

```python
from src.config import HardwareConfig, get_device, get_batch_size

# Get optimal settings
device = get_device()
batch_size = get_batch_size(default=32, memory_factor=0.8)

# Print system information
from src.config import print_system_info
print_system_info()
```

### Manual Hardware Configuration

```python
import torch
from src.faithfulness import FaithfulnessConfig

# Force specific device
config = FaithfulnessConfig(
    device=torch.device('cpu'),     # Force CPU
    batch_size=16                   # Smaller batch for CPU
)

# Or force GPU
config = FaithfulnessConfig(
    device=torch.device('cuda:0'),  # Specific GPU
    batch_size=64                   # Larger batch for GPU
)
```

### Memory Management

```python
from src.config import memory_efficient_context

# Use memory-efficient context
with memory_efficient_context():
    result = metric.compute_faithfulness_score(model, explainer, data)
```

**Memory Optimization Parameters:**

**memory_factor** (float, default=0.8)
- Fraction of available memory to use for batch size calculation
- Lower values are more conservative
- Range: 0.1-0.9

**oom_reduction_factor** (float, default=0.5)
- Factor to reduce batch size when OOM occurs
- Smaller values provide more aggressive reduction
- Range: 0.1-0.8

## Computation Limits

### ComputationLimits Configuration

```python
from src.robust_computation import ComputationLimits

limits = ComputationLimits(
    max_memory_gb=8.0,              # Maximum memory usage
    max_computation_time=3600,      # Maximum time in seconds
    numerical_epsilon=1e-8,         # Numerical stability
    max_retries=3,                  # Maximum retry attempts
    retry_delay=1.0,                # Delay between retries
    enable_memory_monitoring=True,   # Monitor memory usage
    enable_gradient_clipping=True,   # Clip gradients for stability
    gradient_clip_value=1.0         # Gradient clipping threshold
)

config = FaithfulnessConfig(computation_limits=limits)
```

**max_memory_gb** (float, optional)
- Maximum memory usage in GB
- Computation will be limited to stay under this threshold
- `None` for no limit

**max_computation_time** (int, optional)
- Maximum computation time in seconds
- Computation will be terminated if exceeded
- `None` for no time limit

**max_retries** (int, default=3)
- Maximum number of retry attempts for failed operations
- Helps handle transient failures

**enable_memory_monitoring** (bool, default=True)
- Enable real-time memory monitoring
- Automatically adjusts computation to prevent OOM

**enable_gradient_clipping** (bool, default=True)
- Enable gradient clipping for numerical stability
- Prevents exploding gradients in neural networks

## Modality-Specific Configuration

### Text Data Configuration

```python
from src.masking import create_masker
from src.baseline import create_baseline_generator

# Text masker configuration
text_masker = create_masker(
    modality="text",
    strategy="pad",
    pad_token_id=0,           # Padding token ID
    mask_token_id=103,        # [MASK] token ID for BERT
    unk_token_id=100,         # [UNK] token ID
    vocab_size=30522,         # Vocabulary size
    random_seed=42
)

# Text baseline generator
text_generator = create_baseline_generator(
    modality="text",
    strategy="random",
    vocab_size=30522,
    pad_token_id=0,
    special_tokens=[0, 101, 102, 103],  # [PAD], [CLS], [SEP], [MASK]
    random_seed=42
)
```

### Tabular Data Configuration

```python
import numpy as np

# Compute feature statistics
feature_means = np.mean(training_data, axis=0)
feature_stds = np.std(training_data, axis=0)
feature_mins = np.min(training_data, axis=0)
feature_maxs = np.max(training_data, axis=0)

# Tabular masker configuration
tabular_masker = create_masker(
    modality="tabular",
    strategy="mean",
    feature_means=feature_means,
    feature_stds=feature_stds,
    random_seed=42
)

# Tabular baseline generator
tabular_generator = create_baseline_generator(
    modality="tabular",
    strategy="gaussian",
    feature_means=feature_means,
    feature_stds=feature_stds,
    feature_ranges=(feature_mins, feature_maxs),
    random_seed=42
)
```

### Image Data Configuration

```python
# Image masker configuration
image_masker = create_masker(
    modality="image",
    strategy="zero",
    blur_kernel_size=15,      # Kernel size for blur masking
    noise_std=0.1,           # Standard deviation for noise
    random_seed=42
)

# Image baseline generator
image_generator = create_baseline_generator(
    modality="image",
    strategy="gaussian",
    pixel_mean=0.5,          # Mean pixel value
    pixel_std=0.1,           # Pixel noise standard deviation
    random_seed=42
)
```

## Performance Tuning

### Computation Speed vs. Accuracy Trade-offs

**Fast Configuration (for development/testing):**
```python
fast_config = FaithfulnessConfig(
    n_samples=100,           # Fewer samples
    batch_size=64,           # Larger batches
    confidence_level=0.90,   # Lower confidence
    enable_streaming=False   # Batch processing
)
```

**Balanced Configuration (recommended):**
```python
balanced_config = FaithfulnessConfig(
    n_samples=1000,          # Standard samples
    batch_size=32,           # Moderate batches
    confidence_level=0.95,   # Standard confidence
    enable_streaming=False   # Batch processing
)
```

**High Precision Configuration (for final results):**
```python
precision_config = FaithfulnessConfig(
    n_samples=5000,          # Many samples
    batch_size=16,           # Smaller batches for stability
    confidence_level=0.99,   # High confidence
    enable_streaming=True    # Memory efficient
)
```

### Memory Optimization

**Low Memory Configuration:**
```python
low_memory_config = FaithfulnessConfig(
    batch_size=8,            # Small batches
    enable_streaming=True,   # Stream computation
    computation_limits=ComputationLimits(
        max_memory_gb=4.0,
        enable_memory_monitoring=True
    )
)
```

**High Memory Configuration:**
```python
high_memory_config = FaithfulnessConfig(
    batch_size=128,          # Large batches
    enable_streaming=False,  # Batch processing
    computation_limits=ComputationLimits(
        max_memory_gb=32.0
    )
)
```

## Environment Variables

Set environment variables to override default configurations:

```bash
# Hardware settings
export FAITHFULNESS_DEVICE="cuda"
export FAITHFULNESS_BATCH_SIZE="64"

# Computation settings
export FAITHFULNESS_N_SAMPLES="2000"
export FAITHFULNESS_RANDOM_SEED="123"

# Memory settings
export FAITHFULNESS_MAX_MEMORY_GB="16.0"
export FAITHFULNESS_MEMORY_FACTOR="0.7"
```

## Configuration Validation

The framework includes automatic configuration validation:

```python
from src.faithfulness import FaithfulnessConfig

# Invalid configuration will raise ValueError
try:
    config = FaithfulnessConfig(
        n_samples=-100,          # Invalid: negative samples
        confidence_level=1.5,    # Invalid: > 1.0
        batch_size=0            # Invalid: zero batch size
    )
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Best Practices

### Reproducibility
- Always set `random_seed` for reproducible results
- Use the same configuration across experiments
- Document all hyperparameter choices

### Performance
- Start with default configurations
- Adjust `n_samples` based on accuracy requirements
- Use `enable_streaming=True` for large datasets
- Monitor memory usage and adjust `batch_size` accordingly

### Accuracy
- Use higher `n_samples` for final evaluations
- Choose appropriate baseline and masking strategies for your modality
- Validate results with multiple random seeds

### Hardware Optimization
- Let the framework auto-detect optimal settings
- Override only when necessary for specific requirements
- Use memory-efficient contexts for large computations