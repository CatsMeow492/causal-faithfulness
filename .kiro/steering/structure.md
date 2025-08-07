# Project Structure

## Directory Organization

```
├── src/                    # Core implementation
│   ├── faithfulness.py     # Main metric computation (F-score)
│   ├── masking.py          # Feature masking for different modalities
│   ├── baseline.py         # Baseline generation (x_rand)
│   ├── explainers.py       # Explanation method wrappers
│   └── config.py           # Hardware compatibility & settings
├── data/                   # Datasets and preprocessed data
├── results/                # Experimental results and analysis
├── paper/                  # Research paper and figures
├── requirements.txt        # Python dependencies
└── references.bib          # Bibliography
```

## Core Modules

### faithfulness.py
- `FaithfulnessMetric`: Main computation class
- `FaithfulnessConfig`: Configuration dataclass
- `FaithfulnessResult`: Result container with statistics
- `compute_faithfulness_score()`: Convenience function

### masking.py
- `FeatureMasker`: Unified masking interface
- `TextMasker`, `TabularMasker`, `ImageMasker`: Modality-specific implementations
- `MaskingStrategy`, `DataModality`: Enums for configuration

### baseline.py
- `BaselineGenerator`: Unified baseline generation
- `TextBaselineGenerator`, `TabularBaselineGenerator`, `ImageBaselineGenerator`: Modality-specific
- `BaselineStrategy`: Enum for different baseline types

### explainers.py
- `ExplainerWrapper`: Abstract base class
- `SHAPWrapper`, `IntegratedGradientsWrapper`, `LIMEWrapper`: Method implementations
- `Attribution`: Unified result format
- `RandomExplainer`: Negative control baseline

### config.py
- `HardwareConfig`: Mac M-series optimization
- `get_device()`, `get_batch_size()`: Hardware detection utilities
- `DEFAULT_CONFIG`: System-wide defaults

## Coding Conventions

- **Imports**: Absolute imports from `src/` modules
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings for classes and methods
- **Error handling**: Graceful degradation with warnings, not exceptions
- **Device handling**: Automatic GPU/CPU detection with fallbacks
- **Reproducibility**: Random seeds throughout for consistent results

## Data Flow

1. **Input**: Model + explainer + data
2. **Explanation**: Generate feature attributions
3. **Masking**: Create x_∖E (explained features masked)
4. **Baseline**: Generate x_rand samples
5. **Evaluation**: Compute F(E) = 1 - E[|f(x) - f(x_∖E)|] / E[|f(x) - f(x_rand)|]
6. **Statistics**: Bootstrap confidence intervals and significance testing