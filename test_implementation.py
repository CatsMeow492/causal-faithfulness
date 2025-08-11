#!/usr/bin/env python3
"""
Test script to validate the fixed implementation and resolve import issues.
This script tests all core components of the causal-faithfulness framework.
"""

import sys
import torch
import numpy as np
import warnings
from typing import Dict, List, Any

# Test imports
def test_imports():
    """Test that all core modules can be imported without errors."""
    print("=== Testing Imports ===")
    
    try:
        # Core faithfulness metric
        from src.faithfulness import (
            FaithfulnessConfig,
            FaithfulnessResult,
            FaithfulnessMetric,
            compute_faithfulness_score
        )
        print("✓ Faithfulness module imported successfully")
        
        # Configuration utilities
        from src.config import (
            HardwareConfig,
            get_device,
            get_batch_size,
            print_system_info,
            DEFAULT_CONFIG
        )
        print("✓ Config module imported successfully")
        
        # Feature masking utilities
        from src.masking import (
            FeatureMasker,
            DataModality,
            MaskingStrategy,
            TextMasker,
            TabularMasker,
            ImageMasker,
            create_masker
        )
        print("✓ Masking module imported successfully")
        
        # Baseline generation utilities
        from src.baseline import (
            BaselineGenerator,
            BaselineStrategy,
            TextBaselineGenerator,
            TabularBaselineGenerator,
            ImageBaselineGenerator,
            create_baseline_generator
        )
        print("✓ Baseline module imported successfully")
        
        # Explainer wrappers
        from src.explainers import (
            ExplainerWrapper,
            Attribution,
            SHAPWrapper,
            IntegratedGradientsWrapper,
            LIMEWrapper,
            RandomExplainer
        )
        print("✓ Explainers module imported successfully")
        
        # Dataset loaders
        from src.datasets import (
            DatasetSample,
            DatasetInfo,
            DatasetManager,
            SST2DatasetLoader,
            WikiText2DatasetLoader
        )
        print("✓ Datasets module imported successfully")
        
        # Model wrappers
        from src.models import (
            ModelPrediction,
            ModelConfig,
            BaseModelWrapper,
            BERTSentimentWrapper,
            GPT2LanguageModelWrapper,
            ModelManager
        )
        print("✓ Models module imported successfully")
        
        # Additional modules
        from src.evaluation import EvaluationPipeline
        print("✓ Evaluation module imported successfully")
        
        from src.roar import ROARBenchmark
        print("✓ ROAR module imported successfully")
        
        from src.statistical_analysis import StatisticalAnalyzer
        print("✓ Statistical analysis module imported successfully")
        
        from src.visualization import ResultVisualizer
        print("✓ Visualization module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import: {e}")
        return False


def test_hardware_configuration():
    """Test hardware configuration and device detection."""
    print("\n=== Testing Hardware Configuration ===")
    
    try:
        from src.config import get_device, get_batch_size, print_system_info
        
        # Test device detection
        device = get_device()
        print(f"✓ Device detected: {device}")
        
        # Test batch size calculation
        batch_size = get_batch_size()
        print(f"✓ Batch size calculated: {batch_size}")
        
        # Print system info
        print_system_info()
        
        return True
        
    except Exception as e:
        print(f"✗ Hardware configuration test failed: {e}")
        return False


def test_faithfulness_metric():
    """Test the core faithfulness metric with synthetic data."""
    print("\n=== Testing Faithfulness Metric ===")
    
    try:
        from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
        from src.masking import DataModality
        
        # Create configuration
        config = FaithfulnessConfig(
            n_samples=100,  # Reduced for testing
            batch_size=16,
            random_seed=42
        )
        
        # Initialize metric
        metric = FaithfulnessMetric(config, modality=DataModality.TABULAR)
        print("✓ FaithfulnessMetric initialized successfully")
        
        # Create synthetic model and data
        def synthetic_model(x):
            """Simple synthetic model for testing."""
            if isinstance(x, torch.Tensor):
                # Simple linear model: sum of features
                return torch.softmax(torch.sum(x, dim=-1, keepdim=True).repeat(1, 2), dim=-1)
            return torch.tensor([[0.5, 0.5]])
        
        def synthetic_explainer(model, data):
            """Simple synthetic explainer for testing."""
            if isinstance(data, torch.Tensor):
                # Return feature importance as absolute values
                importance = torch.abs(data).cpu().numpy().flatten()
            else:
                importance = np.random.random(10)
            
            from src.explainers import Attribution
            return Attribution(
                feature_scores=importance,
                feature_indices=list(range(len(importance))),
                method_name="SyntheticExplainer",
                computation_time=0.1
            )
        
        # Create synthetic data
        synthetic_data = torch.randn(1, 10)
        
        # Test faithfulness computation
        result = metric.compute_faithfulness_score(
            model=synthetic_model,
            explainer=synthetic_explainer,
            data=synthetic_data
        )
        
        print(f"✓ Faithfulness score computed: {result.f_score:.4f}")
        print(f"✓ Confidence interval: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        print(f"✓ Statistical significance: {result.statistical_significance}")
        
        return True
        
    except Exception as e:
        print(f"✗ Faithfulness metric test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_explainer_wrappers():
    """Test explainer wrapper functionality."""
    print("\n=== Testing Explainer Wrappers ===")
    
    try:
        from src.explainers import RandomExplainer, Attribution
        
        # Test random explainer (should always work)
        explainer = RandomExplainer(random_seed=42)
        
        # Create synthetic model and data
        def synthetic_model(x):
            return torch.softmax(torch.randn(1, 2), dim=-1)
        
        synthetic_data = torch.randn(1, 10)
        
        # Test explanation
        attribution = explainer.explain(synthetic_model, synthetic_data)
        
        print(f"✓ Random explainer generated attribution with {len(attribution.feature_scores)} features")
        print(f"✓ Attribution method: {attribution.method_name}")
        print(f"✓ Computation time: {attribution.computation_time:.4f}s")
        
        # Test top features extraction
        top_features = attribution.get_top_features(k=5)
        print(f"✓ Top 5 features: {top_features}")
        
        return True
        
    except Exception as e:
        print(f"✗ Explainer wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_masking_and_baseline():
    """Test feature masking and baseline generation."""
    print("\n=== Testing Masking and Baseline Generation ===")
    
    try:
        from src.masking import FeatureMasker, DataModality, MaskingStrategy
        from src.baseline import BaselineGenerator, BaselineStrategy
        
        # Test tabular masking
        masker = FeatureMasker(
            modality=DataModality.TABULAR,
            strategy=MaskingStrategy.ZERO,
            random_seed=42
        )
        
        # Test baseline generation
        baseline_gen = BaselineGenerator(
            modality=DataModality.TABULAR,
            strategy=BaselineStrategy.RANDOM,
            random_seed=42
        )
        
        # Create synthetic data
        data = torch.randn(1, 10)
        features_to_mask = [0, 2, 4]
        
        # Test masking
        masked_data = masker.mask_features(data, features_to_mask, mask_explained=True)
        print(f"✓ Feature masking successful, shape: {masked_data.shape}")
        
        # Test baseline generation
        baseline_data = baseline_gen.generate_baseline(data, batch_size=1)
        print(f"✓ Baseline generation successful, shape: {baseline_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Masking and baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test dataset loading functionality (without actually downloading)."""
    print("\n=== Testing Dataset Loading ===")
    
    try:
        from src.datasets import DatasetManager, DatasetSample
        
        # Initialize dataset manager
        manager = DatasetManager()
        print("✓ DatasetManager initialized successfully")
        
        # Test dataset sample creation
        sample = DatasetSample(
            text="This is a test sentence.",
            tokens=torch.tensor([101, 2023, 2003, 1037, 3231, 6251, 1012, 102]),
            attention_mask=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]),
            label=1,
            metadata={"test": True}
        )
        print("✓ DatasetSample created successfully")
        
        # Test validation
        from src.datasets import BaseDatasetLoader
        loader = BaseDatasetLoader()
        is_valid = loader._validate_sample(sample)
        print(f"✓ Sample validation: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_wrappers():
    """Test model wrapper functionality (without loading actual models)."""
    print("\n=== Testing Model Wrappers ===")
    
    try:
        from src.models import ModelManager, ModelConfig, ModelPrediction
        
        # Initialize model manager
        manager = ModelManager()
        print("✓ ModelManager initialized successfully")
        
        # Test model configuration
        config = ModelConfig(
            model_name="test-model",
            device=torch.device('cpu'),
            batch_size=16,
            max_length=512
        )
        print("✓ ModelConfig created successfully")
        
        # Test model prediction structure
        prediction = ModelPrediction(
            logits=torch.tensor([0.1, 0.9]),
            probabilities=torch.tensor([0.25, 0.75]),
            predicted_class=1,
            confidence=0.75,
            metadata={"test": True}
        )
        print("✓ ModelPrediction created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_integration():
    """Test end-to-end integration with all components."""
    print("\n=== Testing End-to-End Integration ===")
    
    try:
        from src.faithfulness import compute_faithfulness_score, FaithfulnessConfig
        from src.explainers import RandomExplainer
        
        # Create simple synthetic setup
        def simple_model(x):
            """Simple model that prefers positive features."""
            if isinstance(x, torch.Tensor):
                score = torch.sum(torch.clamp(x, min=0), dim=-1, keepdim=True)
                return torch.softmax(torch.cat([score, -score], dim=-1), dim=-1)
            return torch.tensor([[0.5, 0.5]])
        
        # Create explainer
        explainer = RandomExplainer(random_seed=42)
        
        # Create synthetic data
        data = torch.randn(1, 5)
        
        # Create configuration
        config = FaithfulnessConfig(
            n_samples=50,  # Small for testing
            batch_size=8,
            random_seed=42
        )
        
        # Compute faithfulness score
        result = compute_faithfulness_score(
            model=simple_model,
            explainer=lambda m, d: explainer.explain(m, d),
            data=data,
            config=config
        )
        
        print(f"✓ End-to-end faithfulness computation successful")
        print(f"  F-score: {result.f_score:.4f}")
        print(f"  Confidence interval: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        print(f"  Samples: {result.n_samples}")
        print(f"  Significant: {result.statistical_significance}")
        
        return True
        
    except Exception as e:
        print(f"✗ End-to-end integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests to validate the implementation."""
    print("Testing Causal-Faithfulness Implementation")
    print("=" * 50)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    tests = [
        ("Import Tests", test_imports),
        ("Hardware Configuration", test_hardware_configuration),
        ("Faithfulness Metric", test_faithfulness_metric),
        ("Explainer Wrappers", test_explainer_wrappers),
        ("Masking and Baseline", test_masking_and_baseline),
        ("Dataset Loading", test_dataset_loading),
        ("Model Wrappers", test_model_wrappers),
        ("End-to-End Integration", test_end_to_end_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)