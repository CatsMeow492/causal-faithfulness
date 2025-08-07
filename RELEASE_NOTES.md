# Release Notes - Version 1.0.0

**Release Date:** 2025-08-06

## Overview

This release provides a complete implementation of the causal-faithfulness metric for evaluating post-hoc explanations across different model architectures and data modalities.

## Key Features

- **Model-agnostic faithfulness metric**: Quantifies explanation quality through causal intervention semantics
- **Multi-modal support**: Works with text, tabular, and image data
- **Theoretical foundation**: Satisfies key axioms (Causal Influence, Sufficiency, Monotonicity, Normalization)
- **Statistical rigor**: Built-in confidence intervals and significance testing
- **Hardware optimization**: Efficient computation on Mac M-series, CUDA, and CPU
- **Comprehensive testing**: Unit, integration, and validation test suites
- **Reproducibility framework**: Ensures consistent results across runs and platforms

## Supported Explanation Methods

- SHAP (KernelSHAP, TreeSHAP, DeepSHAP)
- Integrated Gradients
- LIME
- Random baseline (for sanity checking)

## Changes in This Release

- Initial release

## Contributors



## Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

## Quick Start

```python
from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
from src.explainers import SHAPWrapper

# Configure metric
config = FaithfulnessConfig(n_samples=1000, random_seed=42)
metric = FaithfulnessMetric(config)

# Evaluate explainer
result = metric.compute_faithfulness_score(model, explainer, data)
print(f"Faithfulness score: {result.f_score:.4f}")
```

## Documentation

- API Documentation: `docs/API.md`
- Tutorial: `docs/TUTORIAL.md`
- Configuration Guide: `docs/CONFIGURATION.md`
- Reproducibility Guide: `docs/REPRODUCIBILITY.md`

## Testing

Run the test suite:
```bash
pytest tests/ -v
python scripts/ci_toy_model_test.py
```

## Citation

If you use this work in your research, please cite:

```bibtex
@software{causal_faithfulness_metric_1_0_0,
  title={A Causal-Faithfulness Score for Post-hoc Explanations Across Model Architectures},
  author={[Authors]},
  version={1.0.0},
  year={2025},
  url={[Repository URL]},
  doi={[Zenodo DOI]}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the examples in the `examples/` directory
