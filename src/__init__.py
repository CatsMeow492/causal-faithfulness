"""
Causal-Faithfulness Score for Post-hoc Explanations
A model-agnostic metric for evaluating explanation faithfulness across architectures.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Core faithfulness metric
from .faithfulness import (
    FaithfulnessConfig,
    FaithfulnessResult,
    FaithfulnessMetric,
    compute_faithfulness_score
)

# Configuration utilities
from .config import (
    HardwareConfig,
    get_device,
    get_batch_size,
    print_system_info,
    DEFAULT_CONFIG
)

# Feature masking utilities
from .masking import (
    FeatureMasker,
    DataModality,
    MaskingStrategy,
    TextMasker,
    TabularMasker,
    ImageMasker,
    create_masker
)

# Baseline generation utilities
from .baseline import (
    BaselineGenerator,
    BaselineStrategy,
    TextBaselineGenerator,
    TabularBaselineGenerator,
    ImageBaselineGenerator,
    create_baseline_generator
)