# Tutorial: Getting Started with Causal-Faithfulness Metric

## Overview

This tutorial provides a step-by-step guide to using the Causal-Faithfulness Metric framework for evaluating explanation methods. We'll cover basic concepts, setup, and practical examples.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Basic Concepts](#basic-concepts)
3. [Quick Start](#quick-start)
4. [Working with Different Data Types](#working-with-different-data-types)
5. [Comparing Explanation Methods](#comparing-explanation-methods)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Installation and Setup

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.1.0
- NumPy, SciPy, scikit-learn
- Optional: SHAP, LIME, Captum for explanation methods

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.faithfulness import FaithfulnessMetric; print('Installation successful!')"
```

### Hardware Compatibility

The framework automatically detects and optimizes for your hardware:

```python
from src.config import print_system_info

# Check your system configuration
print_system_info()
```

## Basic Concepts

### What is Causal-Faithfulness?

The causal-faithfulness metric F(E) quantifies how well an explanation method E captures a model's true decision logic through causal intervention:

```
F(E) = 1 - E[|f(x) - f(x_∖E)|] / E[|f(x) - f(x_rand)|]
```

Where:
- `x` is the original input
- `x_∖E` is the input with non-explained features masked
- `x_rand` is a random baseline
- `f` is the model prediction function

### Key Properties

- **Range**: F(E) ∈ [0, 1]
- **Interpretation**: Higher scores indicate better faithfulness
- **Model-agnostic**: Works with any model providing predictions
- **Multi-modal**: Supports text, tabular, and image data

## Quick Start

### Minimal Example

```python
import torch
import numpy as np
from src.faithfulness import compute_faithfulness_score
from src.explainers import RandomExplainer

# Create dummy model and data
def dummy_model(x):
    return torch.randn(x.shape[0], 2)  # Binary classification

dummy_data = torch.randn(1, 10)  # Single sample, 10 features
explainer = RandomExplainer()

# Compute faithfulness score
result = compute_faithfulness_score(
    model=dummy_model,
    explainer=explainer.explain,
    data=dummy_data
)

print(f"F-score: {result.f_score:.4f}")
print(f"Confidence interval: {result.confidence_interval}")
```

### Step-by-Step Example

```python
from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
from src.explainers import SHAPWrapper
from src.masking import DataModality

# Step 1: Configure the metric
config = FaithfulnessConfig(
    n_samples=1000,
    baseline_strategy="gaussian",
    confidence_level=0.95,
    random_seed=42
)

# Step 2: Initialize the metric
metric = FaithfulnessMetric(config, modality=DataModality.TABULAR)

# Step 3: Set up your explainer
explainer = SHAPWrapper(explainer_type="kernel", n_samples=500)

# Step 4: Compute faithfulness
result = metric.compute_faithfulness_score(
    model=your_model,
    explainer=explainer.explain,
    data=your_data,
    target_class=None  # Auto-detect
)

# Step 5: Interpret results
print(f"Faithfulness score: {result.f_score:.4f}")
print(f"Statistical significance: {result.statistical_significance}")
```

## Working with Different Data Types

### Tabular Data

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.masking import DataModality
from src.explainers import SHAPWrapper, LIMEWrapper

# Load your tabular data
df = pd.read_csv('your_data.csv')
X, y = df.drop('target', axis=1), df['target']

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Create model wrapper for PyTorch compatibility
def model_wrapper(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    proba = model.predict_proba(x)
    return torch.from_numpy(proba).float()

# Configure for tabular data
config = FaithfulnessConfig(
    baseline_strategy="gaussian",
    masking_strategy="mean"
)
metric = FaithfulnessMetric(config, modality=DataModality.TABULAR)

# Use SHAP for tree models
explainer = SHAPWrapper(explainer_type="tree")
result = metric.compute_faithfulness_score(
    model=model_wrapper,
    explainer=explainer.explain,
    data=torch.from_numpy(X.iloc[0:1].values).float()
)
```

### Text Data

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.masking import DataModality

# Load pre-trained model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare text
text = "This movie is fantastic!"
tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Create model wrapper
def text_model_wrapper(tokenized_input):
    with torch.no_grad():
        outputs = model(**tokenized_input)
        return outputs.logits

# Configure for text data
config = FaithfulnessConfig(
    baseline_strategy="random",
    masking_strategy="pad"
)
metric = FaithfulnessMetric(config, modality=DataModality.TEXT)

# Use Integrated Gradients for neural networks
from src.explainers import IntegratedGradientsWrapper
explainer = IntegratedGradientsWrapper(n_steps=50)

result = metric.compute_faithfulness_score(
    model=text_model_wrapper,
    explainer=explainer.explain,
    data=tokenized
)
```

### Image Data

```python
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

image = Image.open('your_image.jpg')
image_tensor = transform(image).unsqueeze(0)

# Configure for image data
config = FaithfulnessConfig(
    baseline_strategy="gaussian",
    masking_strategy="zero"
)
metric = FaithfulnessMetric(config, modality=DataModality.IMAGE)

# Use SHAP for images (requires background images)
background_images = torch.randn(10, 3, 224, 224)  # Random background
explainer = SHAPWrapper(explainer_type="deep")

result = metric.compute_faithfulness_score(
    model=model,
    explainer=lambda m, d: explainer.explain(m, d, background_data=background_images),
    data=image_tensor
)
```

## Comparing Explanation Methods

### Multi-Explainer Evaluation

```python
from src.explainers import SHAPWrapper, IntegratedGradientsWrapper, LIMEWrapper, RandomExplainer

# Initialize multiple explainers
explainers = {
    "SHAP_Kernel": SHAPWrapper(explainer_type="kernel", n_samples=500),
    "IntegratedGradients": IntegratedGradientsWrapper(n_steps=50),
    "LIME": LIMEWrapper(n_samples=1000, modality="tabular"),
    "Random_Baseline": RandomExplainer()
}

# Evaluate all explainers
results = {}
for name, explainer in explainers.items():
    print(f"Evaluating {name}...")
    
    # Handle explainer-specific requirements
    if name == "SHAP_Kernel":
        result = metric.compute_faithfulness_score(
            model=model,
            explainer=lambda m, d: explainer.explain(m, d, background_data=background_data),
            data=test_data
        )
    elif name == "LIME":
        result = metric.compute_faithfulness_score(
            model=model,
            explainer=lambda m, d: explainer.explain(m, d, training_data=training_data),
            data=test_data
        )
    else:
        result = metric.compute_faithfulness_score(
            model=model,
            explainer=explainer.explain,
            data=test_data
        )
    
    results[name] = result

# Compare results
print("\nComparison Results:")
print("="*60)
print(f"{'Method':<20} {'F-score':<10} {'95% CI':<20} {'Significant'}")
print("="*60)

for name, result in results.items():
    ci_str = f"[{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]"
    sig_str = "Yes" if result.statistical_significance else "No"
    print(f"{name:<20} {result.f_score:<10.4f} {ci_str:<20} {sig_str}")
```

### Statistical Analysis

```python
import scipy.stats as stats

# Perform pairwise comparisons
explainer_names = list(results.keys())
for i in range(len(explainer_names)):
    for j in range(i+1, len(explainer_names)):
        name1, name2 = explainer_names[i], explainer_names[j]
        score1, score2 = results[name1].f_score, results[name2].f_score
        
        # Simple comparison (for more rigorous analysis, use bootstrap)
        if abs(score1 - score2) > 0.05:  # Practical significance threshold
            better = name1 if score1 > score2 else name2
            print(f"{better} significantly outperforms the other (Δ = {abs(score1-score2):.3f})")
```

## Advanced Configuration

### Custom Masking Strategies

```python
from src.masking import create_masker, MaskingStrategy

# Create custom masker with specific parameters
masker = create_masker(
    modality="text",
    strategy="mask",
    pad_token_id=0,
    mask_token_id=103,  # [MASK] token for BERT
    vocab_size=30522,
    random_seed=42
)

# Use in faithfulness computation
config = FaithfulnessConfig(masking_strategy="mask")
```

### Custom Baseline Generation

```python
from src.baseline import create_baseline_generator, BaselineStrategy

# Create custom baseline generator
generator = create_baseline_generator(
    modality="tabular",
    strategy="gaussian",
    feature_means=np.mean(training_data, axis=0),
    feature_stds=np.std(training_data, axis=0),
    random_seed=42
)

config = FaithfulnessConfig(baseline_strategy="gaussian")
```

### Hardware Optimization

```python
from src.config import HardwareConfig, memory_efficient_context

# Check hardware capabilities
hw_config = HardwareConfig()
print(f"Device: {hw_config.device}")
print(f"MPS support: {hw_config.supports_mps}")

# Use memory-efficient computation
config = FaithfulnessConfig(
    batch_size=hw_config.get_batch_size(default=32, memory_factor=0.8),
    enable_streaming=True  # For large datasets
)

# Compute with memory management
with memory_efficient_context():
    result = metric.compute_faithfulness_score(model, explainer, data)
```

### Computation Limits

```python
from src.robust_computation import ComputationLimits

# Set computational limits
limits = ComputationLimits(
    max_memory_gb=8.0,
    max_computation_time=1800,  # 30 minutes
    numerical_epsilon=1e-8,
    max_retries=3
)

config = FaithfulnessConfig(computation_limits=limits)
```

## Troubleshooting

### Common Issues and Solutions

#### Out of Memory Errors

```python
# Solution 1: Reduce batch size
config = FaithfulnessConfig(batch_size=8)

# Solution 2: Enable streaming
config = FaithfulnessConfig(enable_streaming=True)

# Solution 3: Use memory-efficient context
with memory_efficient_context():
    result = metric.compute_faithfulness_score(model, explainer, data)
```

#### Numerical Instability

```python
# Solution: Increase numerical epsilon
config = FaithfulnessConfig(numerical_epsilon=1e-6)

# Or use computation limits with gradient clipping
limits = ComputationLimits(
    enable_gradient_clipping=True,
    gradient_clip_value=1.0
)
config = FaithfulnessConfig(computation_limits=limits)
```

#### Slow Computation

```python
# Solution 1: Reduce samples for testing
config = FaithfulnessConfig(n_samples=100)

# Solution 2: Use appropriate explainer settings
explainer = SHAPWrapper(n_samples=100)  # Reduce SHAP samples
explainer = IntegratedGradientsWrapper(n_steps=20)  # Reduce IG steps

# Solution 3: Enable hardware acceleration
config = FaithfulnessConfig(device=torch.device('cuda'))  # Force GPU
```

#### Explainer Compatibility Issues

```python
# Solution: Create wrapper functions
def safe_explainer_wrapper(model, data, **kwargs):
    try:
        return explainer.explain(model, data, **kwargs)
    except Exception as e:
        print(f"Explainer failed: {e}")
        # Return zero attribution as fallback
        if isinstance(data, torch.Tensor):
            n_features = data.shape[-1]
        else:
            n_features = len(data)
        
        from src.explainers import Attribution
        return Attribution(
            feature_scores=np.zeros(n_features),
            feature_indices=list(range(n_features)),
            method_name="FALLBACK",
            computation_time=0.0
        )

# Use the wrapper
result = metric.compute_faithfulness_score(
    model=model,
    explainer=safe_explainer_wrapper,
    data=data
)
```

### Debugging Tips

#### Enable Verbose Output

```python
import logging
logging.basicConfig(level=logging.INFO)

# The framework will provide detailed progress information
```

#### Check Intermediate Results

```python
# Access detailed computation metrics
result = metric.compute_faithfulness_score(model, explainer, data)

print("Computation Metrics:")
for key, value in result.computation_metrics.items():
    print(f"  {key}: {value}")

print(f"Baseline performance: {result.baseline_performance}")
print(f"Explained performance: {result.explained_performance}")
```

#### Validate Input Data

```python
from src.masking import create_masker

# Test masking functionality
masker = create_masker(modality="tabular", strategy="mean")
if masker.validate_input(your_data):
    print("Data is compatible with masker")
else:
    print("Data format issue detected")
```

## Best Practices

### Experimental Design

1. **Use Multiple Random Seeds**: Run experiments with different random seeds to assess stability
2. **Appropriate Sample Sizes**: Use at least 1000 Monte-Carlo samples for reliable estimates
3. **Include Baselines**: Always include random explainer as a sanity check
4. **Statistical Testing**: Report confidence intervals and significance tests

### Performance Optimization

1. **Start Small**: Begin with reduced sample sizes for development
2. **Hardware Awareness**: Let the framework auto-detect optimal settings
3. **Memory Management**: Use streaming for large datasets
4. **Batch Processing**: Use appropriate batch sizes for your hardware

### Reproducibility

1. **Fix Random Seeds**: Always set random seeds for reproducible results
2. **Document Configuration**: Save all hyperparameters and settings
3. **Version Control**: Track framework and dependency versions
4. **Data Consistency**: Use the same preprocessing across experiments

### Interpretation Guidelines

1. **F-score Ranges**:
   - 0.8-1.0: Excellent faithfulness
   - 0.6-0.8: Good faithfulness
   - 0.4-0.6: Moderate faithfulness
   - 0.0-0.4: Poor faithfulness

2. **Statistical Significance**: Consider both statistical and practical significance
3. **Confidence Intervals**: Report uncertainty estimates
4. **Sanity Checks**: Verify random explainers have low scores

### Common Pitfalls to Avoid

1. **Insufficient Samples**: Using too few Monte-Carlo samples
2. **Inappropriate Baselines**: Using baselines that don't match data distribution
3. **Ignoring Hardware Limits**: Not accounting for memory constraints
4. **Over-interpretation**: Drawing conclusions from small effect sizes
5. **Missing Validation**: Not including negative controls

## Next Steps

After completing this tutorial, you should be able to:

- Set up and configure the causal-faithfulness metric
- Evaluate explanation methods across different data modalities
- Compare multiple explainers systematically
- Handle common issues and optimize performance
- Interpret results appropriately

For more advanced usage, see:
- [API Documentation](API.md) for detailed function references
- [Configuration Guide](CONFIGURATION.md) for comprehensive parameter documentation
- [Examples](../examples/) for complete working examples
- [Hardware Optimization Guide](HARDWARE_OPTIMIZATION_SUMMARY.md) for performance tuning