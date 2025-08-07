# Technology Stack

## Core Dependencies

- **Python**: ≥3.10 required
- **PyTorch**: ≥2.1.0 for tensor operations and model compatibility
- **NumPy/SciPy**: Core numerical computing and statistical analysis
- **Transformers**: HuggingFace library for NLP model support

## ML/XAI Libraries

- **SHAP**: ≥0.42.0 for Shapley value explanations
- **Captum**: ≥0.6.0 for Integrated Gradients
- **LIME**: ≥0.2.0 for local explanations
- **scikit-learn**: ≥1.3.0 for ML utilities

## Hardware Compatibility

- **Mac M-series**: Optimized with MPS (Metal Performance Shaders) support
- **CUDA**: GPU acceleration when available
- **CPU fallback**: Automatic detection and graceful degradation

## Development Tools

- **pytest**: Testing framework with coverage reporting
- **black**: Code formatting (23.0+)
- **flake8**: Linting and style checking
- **Jupyter**: Interactive development and analysis

## Common Commands

```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Run tests
pytest tests/ --cov=src

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Run example
python -m src.faithfulness --help
```

## Build System

- Standard Python package structure
- No complex build system required
- Direct module imports from `src/`