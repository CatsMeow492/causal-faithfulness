# Design Document: Causal-Faithfulness Score for Post-hoc Explanations

## Overview

The causal-faithfulness metric addresses a critical gap in explanation evaluation by providing a unified, model-agnostic score that quantifies how well feature-based explanations reflect a model's true decision logic. Unlike existing methods that focus on specific aspects (e.g., ROAR's accuracy drop), our metric incorporates causal intervention semantics through a principled mathematical framework.

The core innovation lies in measuring the relative impact of removing explained features compared to random baseline perturbations, normalized to provide interpretable scores between 0 and 1. This design enables fair comparison across different explanation methods, model architectures, and data modalities.

## Architecture

### System Components

The system follows a modular architecture with clear separation of concerns:

1. **Metric Core**: Implements the F(E) formula with statistical analysis
2. **Explainer Wrappers**: Unified interfaces for SHAP, IG, LIME, and baselines  
3. **Modality Handlers**: Text, tabular, and image-specific processing
4. **Hardware Adapters**: GPU/CPU optimization and fallback strategies
5. **Experimental Framework**: Dataset management and result analysis

### Core Architecture Principles

1. **Model Agnosticism**: The system interfaces with models through prediction functions, supporting any architecture that provides probabilistic outputs
2. **Modality Flexibility**: Separate masking strategies for text, tabular, and image data
3. **Computational Efficiency**: Batch processing and hardware-aware optimizations
4. **Statistical Rigor**: Built-in confidence intervals and significance testing## Comp
onents and Interfaces

### 1. Faithfulness Metric Core (`src/faithfulness.py`)

**Primary Interface:**
```python
def compute_faithfulness_score(
    model: Callable,
    explainer: Callable, 
    data: Union[torch.Tensor, Dict],
    config: FaithfulnessConfig
) -> FaithfulnessResult
```

**Key Components:**

- **MetricComputer**: Implements the core F(E) formula with Monte-Carlo sampling
- **BaselineGenerator**: Creates x_rand baselines using modality-specific strategies
- **FeatureMasker**: Generates x_∖E by masking non-explained features
- **StatisticalAnalyzer**: Computes confidence intervals and significance tests

### 2. Explanation Wrappers (`src/explainers/`)

**Supported Methods:**
- **SHAPWrapper**: Interfaces with KernelSHAP, TreeSHAP, and DeepSHAP
- **IntegratedGradientsWrapper**: Handles gradient-based attributions
- **LIMEWrapper**: Manages local perturbation-based explanations
- **RandomExplainer**: Negative control baseline

**Common Interface:**
```python
class ExplainerWrapper:
    def explain(self, model, input_data) -> Attribution
    def get_top_features(self, attribution, k) -> List[int]
```

### 3. Modality Handlers (`src/modalities/`)

**Text Handler:**
- Masking: Replace tokens with [PAD], [MASK], or [UNK] based on model type
- Baseline: Random token sampling from vocabulary or uniform noise

**Tabular Handler:**
- Masking: Replace with feature means, zeros, or sampled values
- Baseline: Gaussian noise or permutation-based scrambling

**Image Handler:**
- Masking: Zero-out pixels, blur regions, or noise injection
- Baseline: Random pixel values or structured noise patterns## Da
ta Models

### Core Data Structures

```python
@dataclass
class FaithfulnessConfig:
    n_samples: int = 1000  # Monte-Carlo samples
    baseline_strategy: str = "random"  # or "mean", "zero"
    masking_strategy: str = "pad"  # modality-specific
    confidence_level: float = 0.95
    batch_size: int = 32
    random_seed: int = 42

@dataclass
class Attribution:
    feature_scores: np.ndarray
    feature_indices: List[int]
    method_name: str
    computation_time: float

@dataclass
class FaithfulnessResult:
    f_score: float
    confidence_interval: Tuple[float, float]
    n_samples: int
    baseline_performance: float
    explained_performance: float
    statistical_significance: bool
    computation_metrics: Dict[str, float]
```

### Experimental Data Schema

```python
@dataclass
class ExperimentResult:
    dataset_name: str
    model_name: str
    explainer_name: str
    f_scores: List[float]
    roar_correlation: float
    p_value: float
    runtime_seconds: float
    memory_usage_mb: float
```## Error 
Handling

### Computational Robustness

1. **Memory Management:**
   - Automatic batch size reduction on OOM errors
   - Streaming computation for large datasets
   - Garbage collection between batches

2. **Numerical Stability:**
   - Epsilon handling for division by zero
   - Gradient clipping for unstable attributions
   - Fallback to double precision when needed

3. **Hardware Compatibility:**
   - Graceful degradation from GPU to CPU
   - Alternative implementations for unsupported operations
   - Clear error messages with suggested fixes

### Input Validation

```python
def validate_inputs(model, data, explainer):
    # Check model callable and output format
    # Validate data shapes and types
    # Verify explainer compatibility
    # Ensure reproducibility settings
```

## Testing Strategy

### Unit Tests (`tests/unit/`)

1. **Metric Properties:**
   - Axiom satisfaction (monotonicity, normalization)
   - Boundary conditions (F ∈ [0,1])
   - Linear model equivalence proofs

2. **Component Testing:**
   - Masking strategies per modality
   - Baseline generation consistency
   - Statistical computation accuracy

### Integration Tests (`tests/integration/`)

1. **End-to-End Workflows:**
   - Complete pipeline on toy datasets
   - Cross-platform compatibility
   - Memory and runtime benchmarks

2. **Explainer Compatibility:**
   - SHAP, IG, LIME integration
   - Different model architectures
   - Various input formats

### Validation Tests (`tests/validation/`)

1. **Sanity Checks:**
   - Random explainer scores lower than informed methods
   - Perfect explainer achieves F ≈ 1
   - Correlation with ROAR on known cases

2. **Reproducibility:**
   - Identical results across runs with same seed
   - Platform-independent outputs
   - Version compatibility