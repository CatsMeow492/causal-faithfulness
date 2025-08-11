# Causal Faithfulness Metric

This repository accompanies the paper "Measuring Causal Faithfulness of Post-hoc Explanations" and contains code to reproduce experiments, analysis, and figures.

## Environment

- Python 3.13
- PyTorch 2.7.1

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing main experiments

See Appendix of the paper for full commands. Key runs:

```
./venv/bin/python scripts/run_sst2_experiments.py --num-samples 200 --max-eval-samples 200 --device cpu --faithfulness-samples 64 --explainers IntegratedGradients,Random --random-seed 42 --output-dir results/sst2_200_ig_random_fs64

./venv/bin/python scripts/run_wikitext2_experiments.py --num-samples 200 --max-eval-samples 200 --device cpu --faithfulness-samples 64 --max-length 256 --explainers IntegratedGradients,Random --random-seed 42 --output-dir results/wikitext2_200_ig_random_fs64_len256_full
```

## Figures and Paper

```
./venv/bin/python scripts/make_figures.py --runs <...> --figdir figures
./scripts/build_paper.sh
```

## License

MIT. See `LICENSE` and dataset attributions in `data/LICENSE_COMPLIANCE.md`.
# A Causal-Faithfulness Score for Post-hoc Explanations Across Model Architectures

[![CI Status](https://github.com/[username]/[repo]/workflows/Continuous%20Integration/badge.svg)](https://github.com/[username]/[repo]/actions)
[![Release](https://img.shields.io/github/v/release/[username]/[repo])](https://github.com/[username]/[repo]/releases)
[![DOI](https://zenodo.org/badge/DOI/[DOI].svg)](https://doi.org/[DOI])
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository implements a novel model-agnostic metric that quantifies how faithfully feature-based explanations reflect a model's true decision logic through causal intervention semantics.

## Overview

The causal-faithfulness metric addresses critical gaps in explanation evaluation by providing a unified score that:
- Satisfies key theoretical axioms (Causal Influence, Sufficiency, Monotonicity, Normalization)
- Works across different model architectures and data modalities
- Incorporates causal intervention through principled feature masking
- Provides interpretable scores between 0 and 1

## Project Structure

```
├── paper/          # Research paper and figures
├── src/            # Source code implementation
├── data/           # Datasets and preprocessed data
├── results/        # Experimental results and analysis
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Installation

### Prerequisites
- Python ≥ 3.10
- macOS with Apple Silicon (M1/M2) or Intel-based systems
- Linux (Ubuntu 20.04+) or Windows with WSL2

### Quick Install (Recommended)

Download the latest release:

1. Go to [Releases](https://github.com/[username]/[repo]/releases)
2. Download the latest `causal-faithfulness-metric-v*.zip`
3. Extract and follow the setup instructions in the archive

### Development Install

1. Clone the repository:
```bash
git clone https://github.com/[username]/[repo].git
cd causal-faithfulness-metric
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or on Windows:
# venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python scripts/ci_toy_model_test.py
```

### Mac M-Series Compatibility

This project is optimized for Apple Silicon with:
- MPS (Metal Performance Shaders) acceleration when available
- Automatic CPU fallbacks for unsupported operations
- Memory-efficient batch processing

## Quick Start

```python
from src.faithfulness import compute_faithfulness_score, FaithfulnessConfig
from src.masking import DataModality

# Load your model and data
model = load_model()
data = load_data()

# Define a simple explainer function
def simple_explainer(model, data):
    # Return random attribution for demonstration
    import numpy as np
    return np.random.randn(len(data))

# Compute faithfulness score
config = FaithfulnessConfig(
    n_samples=1000,
    baseline_strategy="random",
    masking_strategy="zero",
    confidence_level=0.95
)

result = compute_faithfulness_score(
    model=model,
    explainer=simple_explainer,
    data=data,
    config=config
)

print(f"Faithfulness Score: {result.f_score:.3f}")
print(f"95% CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
print(f"Statistical significance: {result.statistical_significance} (p={result.p_value:.4f})")
```

## Core Faithfulness Metric

### Overview

The causal-faithfulness metric F(E) quantifies how well an explanation method E captures the true causal relationships in a model's decision process. The metric is defined as:

**F(E) = 1 - E[|f(x) - f(x_∖E)|] / E[|f(x) - f(x_rand)|]**

Where:
- `x` is the original input
- `x_∖E` is the input with non-explained features masked
- `x_rand` is a random baseline input
- `f` is the model prediction function
- `E[·]` denotes expectation computed via Monte-Carlo sampling

### Key Components

#### FaithfulnessConfig

Configuration class for metric computation with hardware-optimized defaults:

```python
from src.faithfulness import FaithfulnessConfig
from src.masking import DataModality

config = FaithfulnessConfig(
    n_samples=1000,              # Monte-Carlo samples for expectation
    baseline_strategy="random",   # "random", "mean", "zero", "gaussian"
    masking_strategy="pad",      # "pad", "zero", "mean", "noise"
    confidence_level=0.95,       # Statistical confidence level
    batch_size=32,              # Processing batch size
    random_seed=42,             # Reproducibility seed
    device=None,                # Auto-detected (MPS/CUDA/CPU)
    numerical_epsilon=1e-8      # Numerical stability threshold
)
```

## Datasets and Models

### Text Classification
- **Dataset**: SST-2 (Stanford Sentiment Treebank)
- **Model**: bert-base-uncased fine-tuned on SST-2

### Language Modeling
- **Dataset**: WikiText-2
- **Model**: gpt2-small

## Key Features

- **Model Agnostic**: Works with any model providing probabilistic outputs
- **Multi-Modal**: Supports text, tabular, and image data
- **Statistical Rigor**: Built-in confidence intervals and significance testing
- **ROAR Benchmark**: Comprehensive comparison with RemOve And Retrain methodology
- **Validation Suite**: Automated sanity checks and statistical validation
- **Hardware Optimized**: Efficient computation on both GPU and CPU
- **Reproducible**: Fixed seeds and documented hyperparameters

## Evaluation Pipeline

The evaluation framework provides comprehensive experiment management:

```python
from src.evaluation import EvaluationPipeline, ExperimentConfig
from src.explainers import create_shap_explainer, create_integrated_gradients_explainer

# Create experiment configuration
config = ExperimentConfig(
    experiment_name="sst2_bert_faithfulness",
    dataset_name="sst2",
    model_name="textattack/bert-base-uncased-SST-2",
    explainer_names=["shap", "integrated_gradients"],
    num_samples=200
)

# Initialize pipeline
pipeline = EvaluationPipeline(config)

# Create explainers
explainers = {
    "shap": create_shap_explainer("kernel", n_samples=500),
    "integrated_gradients": create_integrated_gradients_explainer()
}

# Run evaluation
results = pipeline.run_evaluation(explainers)

# View results
print(f"Experiment completed in {results.total_runtime:.2f}s")
for explainer_name, summary in results.summary_statistics.items():
    print(f"{explainer_name}: F-score = {summary['mean_f_score']:.3f}")
```

## ROAR Benchmark Comparison

The framework includes a comprehensive ROAR (RemOve And Retrain) benchmark for validation:

```python
from src.roar import ROARBenchmark, ROARConfig, run_roar_evaluation
from src.statistical_analysis import compute_roar_faithfulness_correlations

# Configure ROAR benchmark
roar_config = ROARConfig(
    removal_percentages=[0.1, 0.2, 0.3, 0.4, 0.5],  # Feature removal levels
    n_samples=200,
    batch_size=32,
    random_seed=42
)

# Run ROAR evaluation
roar_results = run_roar_evaluation(
    model=model,
    explainers=explainers,
    samples=dataset_samples,
    config=roar_config
)

# Compute correlations with faithfulness scores
correlations = compute_roar_faithfulness_correlations(
    roar_results=roar_results,
    faithfulness_results=faithfulness_results
)

# View correlation results
for correlation in correlations:
    print(f"{correlation.explainer_name}:")
    print(f"  Pearson r = {correlation.pearson_r:.4f} (p = {correlation.p_value:.4f})")
    print(f"  Spearman ρ = {correlation.spearman_rho:.4f} (p = {correlation.spearman_p:.4f})")
    print(f"  Significant: {'Yes' if correlation.is_significant else 'No'}")
```

## Statistical Analysis and Validation

Comprehensive statistical validation with automated sanity checks:

```python
from src.statistical_analysis import ValidationSuite, run_statistical_analysis, run_sanity_checks

# Initialize validation suite
validation_suite = ValidationSuite(alpha=0.05, random_seed=42)

# Run comprehensive validation
validation_report = validation_suite.run_full_validation(
    faithfulness_results=faithfulness_results,
    roar_results=roar_results,
    random_explainer_results=random_baseline_results
)

# Generate human-readable report
text_report = validation_suite.generate_validation_report_text(validation_report)
print(text_report)

# Run individual statistical tests
statistical_tests = run_statistical_analysis(faithfulness_results, alpha=0.05)
sanity_checks = run_sanity_checks(faithfulness_results, random_baseline_results)

# View test results
for test_name, result in statistical_tests.items():
    print(f"{test_name}: {result}")

for check_name, result in sanity_checks.items():
    print(f"{check_name}: {result}")
```

### Validation Features

- **Paired t-tests**: Statistical significance testing between explainers
- **Bootstrap confidence intervals**: Robust uncertainty quantification
- **Wilcoxon signed-rank tests**: Non-parametric alternatives
- **Multiple comparisons correction**: Bonferroni, Holm, and FDR methods
- **Random explainer validation**: Ensures informed explainers outperform random baselines
- **Metric bounds validation**: Verifies scores remain in [0,1] range
- **Reproducibility checks**: Validates consistent results across runs
- **Monotonicity validation**: Tests expected behavior with feature removal

## Dataset Loading

The framework includes optimized dataset loaders:

```python
from src.datasets import DatasetManager

# Initialize dataset manager
manager = DatasetManager()

# Load SST-2 for sentiment analysis
sst2_samples = manager.load_sst2(split="validation", num_samples=200)

# Load WikiText-2 for language modeling
wikitext_samples = manager.load_wikitext2(split="validation", num_samples=200)

# Print dataset information
manager.print_dataset_summary()
```

## Model Integration

Hardware-aware model wrappers with automatic fallbacks:

```python
from src.models import ModelManager

# Initialize model manager
manager = ModelManager()

# Load BERT for sentiment analysis
bert_model = manager.load_bert_sst2()

# Load GPT-2 for language modeling
gpt2_model = manager.load_gpt2_small()

# Make predictions
prediction = bert_model.predict("This is a great movie!")
print(f"Sentiment: {prediction.predicted_class}, Confidence: {prediction.confidence:.3f}")
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'torch'/'scipy'**
```bash
# Install core dependencies
pip install torch scipy numpy
```

**Memory errors during computation**
```python
# Reduce computational load
config = FaithfulnessConfig(
    n_samples=100,      # Fewer Monte-Carlo samples
    batch_size=4,       # Smaller batches
    device=torch.device('cpu')  # Use CPU
)
```

**MPS/GPU memory errors on Mac**
```python
# Use CPU fallback for problematic operations
from src.config import FaithfulnessConfig
config = FaithfulnessConfig(device=torch.device('cpu'))
```

### Performance Optimization

**Reduce computation time:**
```python
# Use fewer samples for faster computation
config = FaithfulnessConfig(n_samples=500)  # Default: 1000

# Reduce ROAR evaluation scope
roar_config = ROARConfig(
    removal_percentages=[0.2, 0.5],  # Fewer removal levels
    n_samples=100                    # Fewer samples
)
```

**Hardware-specific settings:**
```python
from src.config import get_device, get_batch_size

# Get optimal settings for your hardware
device = get_device(fallback_cpu=True)
batch_size = get_batch_size(default=32)

config = FaithfulnessConfig(device=device, batch_size=batch_size)
```

### Validation and Quality Assurance

**Run comprehensive validation:**
```python
from src.statistical_analysis import ValidationSuite

# Quick validation check
suite = ValidationSuite()
report = suite.run_full_validation(faithfulness_results)

# Check overall assessment
assessment = report['summary']['overall_assessment']
if assessment == 'POOR':
    print("Warning: Validation failed. Check metric implementation.")
elif assessment == 'MODERATE':
    print("Caution: Some validation issues detected.")
else:
    print("Validation passed: Metric is working correctly.")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{causal_faithfulness_metric_v1_0_0,
  title={A Causal-Faithfulness Score for Post-hoc Explanations Across Model Architectures},
  author={[Authors]},
  version={1.0.0},
  year={2025},
  url={https://github.com/[username]/[repo]},
  doi={[Zenodo DOI]}
}
```

For the accompanying research paper:
```bibtex
@article{causal_faithfulness_2025,
  title={A Causal-Faithfulness Score for Post-hoc Explanations Across Model Architectures},
  author={[Authors]},
  journal={[Journal]},
  year={2025},
  doi={[Paper DOI]}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## Contact

For questions or issues, please open a GitHub issue or contact [contact information].