# Product Overview

This repository implements a novel causal-faithfulness metric for evaluating post-hoc explanations across different model architectures and data modalities.

## Core Purpose

The causal-faithfulness metric F(E) quantifies how well explanation methods capture the true causal relationships in a model's decision process through principled feature masking and causal intervention semantics.

## Key Features

- **Model-agnostic**: Works with any model providing probabilistic outputs
- **Multi-modal**: Supports text, tabular, and image data
- **Theoretically grounded**: Satisfies key axioms (Causal Influence, Sufficiency, Monotonicity, Normalization)
- **Statistical rigor**: Built-in confidence intervals and significance testing
- **Hardware optimized**: Efficient computation on Mac M-series, CUDA, and CPU

## Target Users

- ML researchers evaluating explanation methods
- Practitioners needing reliable explanation quality metrics
- Academic researchers in explainable AI