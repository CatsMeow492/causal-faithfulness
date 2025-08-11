#!/usr/bin/env python3
"""
Test script to validate core functionality without optional dependencies.
This focuses on the essential components needed for the experimental framework.
"""

import sys
import torch
import numpy as np
import warnings
from typing import Dict, List, Any

def test_core_imports():
    """Test that all core modules can be imported."""
    print("=== Testing Core Imports ===")
    
    try:
        # Core faithfulness metric
        from src.faithfulness import (
            FaithfulnessConfig,
            FaithfulnessResult,
            FaithfulnessMetric,
            compute_faithfulness_score
        )
        
        # Configuration and hardware
        from src.config import get_device, get_batch_size, print_system_info
        
        # Masking and baseline generation
        from src.masking import FeatureMasker, DataModality, MaskingStrategy
        from src.baseline import BaselineGenerator, BaselineStrategy
        
        # Core explainer functionality (without optional dependencies)
        from src.explainers import ExplainerWrapper, Attribution, RandomExplainer
        
        # Dataset and model structures
        from src.datasets import DatasetSample, DatasetManager
        from src.models import ModelManager, ModelPrediction, ModelConfig
        
        print("✓ All core modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Core import failed: {e}")
        return False


def test_random_explainer_functionality():
    """Test random explainer which doesn't require external dependencies."""
    print("\n=== Testing Random Explainer Functionality ===")
    
    try:
        from src.explainers import RandomExplainer
        from src.faithfulness import compute_faithfulness_score, FaithfulnessConfig
        
        # Create simple model
        def simple_model(x):
            if isinstance(x, torch.Tensor):
                # Simple binary classification based on sum
                score = torch.sum(x, dim=-1, keepdim=True)
                return torch.softmax(torch.cat([score, -score], dim=-1), dim=-1)
            return torch.tensor([[0.5, 0.5]])
        
        # Create random explainer
        explainer = RandomExplainer(random_seed=42)
        
        # Test data
        test_data = torch.randn(1, 10)
        
        # Test explanation generation
        attribution = explainer.explain(simple_model, test_data)
        print(f"✓ Random explainer generated {len(attribution.feature_scores)} feature scores")
        
        # Test faithfulness computation
        config = FaithfulnessConfig(n_samples=50, batch_size=8, random_seed=42)
        result = compute_faithfulness_score(
            model=simple_model,
            explainer=lambda m, d: explainer.explain(m, d),
            data=test_data,
            config=config
        )
        
        print(f"✓ Faithfulness score computed: {result.f_score:.4f}")
        print(f"✓ Confidence interval: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Random explainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_structures():
    """Test dataset structures and basic functionality."""
    print("\n=== Testing Dataset Structures ===")
    
    try:
        from src.datasets import DatasetSample, DatasetManager, DatasetInfo
        
        # Test dataset sample creation
        sample = DatasetSample(
            text="This is a test sentence for sentiment analysis.",
            tokens=torch.tensor([101, 2023, 2003, 1037, 3231, 6251, 2005, 15792, 4106, 1012, 102]),
            attention_mask=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            label=1,
            metadata={"dataset": "test", "index": 0}
        )
        print("✓ DatasetSample created successfully")
        
        # Test dataset manager
        manager = DatasetManager()
        print("✓ DatasetManager initialized")
        
        # Test dataset info structure
        info = DatasetInfo(
            name="Test Dataset",
            task_type="classification",
            license="MIT",
            citation="Test et al. (2024)",
            num_samples=100,
            max_length=512
        )
        print("✓ DatasetInfo structure created")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset structures test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_structures():
    """Test model structures and basic functionality."""
    print("\n=== Testing Model Structures ===")
    
    try:
        from src.models import ModelManager, ModelConfig, ModelPrediction
        
        # Test model configuration
        config = ModelConfig(
            model_name="test-model",
            device=torch.device('cpu'),
            batch_size=16,
            max_length=512
        )
        print("✓ ModelConfig created")
        
        # Test model prediction structure
        prediction = ModelPrediction(
            logits=torch.tensor([0.2, 0.8]),
            probabilities=torch.tensor([0.3, 0.7]),
            predicted_class=1,
            confidence=0.7,
            metadata={"model": "test"}
        )
        print("✓ ModelPrediction structure created")
        
        # Test model manager
        manager = ModelManager()
        print("✓ ModelManager initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Model structures test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_masking_and_baseline():
    """Test masking and baseline generation functionality."""
    print("\n=== Testing Masking and Baseline Generation ===")
    
    try:
        from src.masking import FeatureMasker, DataModality, MaskingStrategy
        from src.baseline import BaselineGenerator, BaselineStrategy
        
        # Test feature masking
        masker = FeatureMasker(
            modality=DataModality.TABULAR,
            strategy=MaskingStrategy.ZERO,
            random_seed=42
        )
        
        test_data = torch.randn(2, 10)
        features_to_mask = [0, 2, 4, 6]
        
        masked_data = masker.mask_features(test_data, features_to_mask, mask_explained=True)
        print(f"✓ Feature masking successful, shape: {masked_data.shape}")
        
        # Test baseline generation
        baseline_gen = BaselineGenerator(
            modality=DataModality.TABULAR,
            strategy=BaselineStrategy.RANDOM,
            random_seed=42
        )
        
        baseline_data = baseline_gen.generate_baseline(test_data, batch_size=2)
        print(f"✓ Baseline generation successful, shape: {baseline_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Masking and baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_faithfulness_computation():
    """Test comprehensive faithfulness computation."""
    print("\n=== Testing Faithfulness Computation ===")
    
    try:
        from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
        from src.masking import DataModality
        from src.explainers import RandomExplainer
        
        # Create configuration
        config = FaithfulnessConfig(
            n_samples=100,
            batch_size=16,
            random_seed=42,
            confidence_level=0.95
        )
        
        # Initialize metric
        metric = FaithfulnessMetric(config, modality=DataModality.TABULAR)
        print("✓ FaithfulnessMetric initialized")
        
        # Create test model that has clear behavior
        def predictable_model(x):
            """Model that strongly prefers positive features."""
            if isinstance(x, torch.Tensor):
                # Sum positive features, ignore negative
                positive_sum = torch.sum(torch.clamp(x, min=0), dim=-1, keepdim=True)
                negative_sum = torch.sum(torch.clamp(x, max=0), dim=-1, keepdim=True)
                
                # Create logits favoring positive features
                logits = torch.cat([negative_sum, positive_sum], dim=-1)
                return torch.softmax(logits, dim=-1)
            return torch.tensor([[0.5, 0.5]])
        
        # Create explainer
        explainer = RandomExplainer(random_seed=42)
        
        # Test data with clear positive/negative pattern
        test_data = torch.tensor([[1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 0.8, -0.8]])
        
        # Compute faithfulness
        result = metric.compute_faithfulness_score(
            model=predictable_model,
            explainer=lambda m, d: explainer.explain(m, d),
            data=test_data
        )
        
        print(f"✓ Faithfulness computation successful")
        print(f"  F-score: {result.f_score:.4f}")
        print(f"  CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        print(f"  Samples: {result.n_samples}")
        print(f"  Significant: {result.statistical_significance}")
        print(f"  P-value: {result.p_value:.4f}")
        
        # Validate result structure
        assert 0 <= result.f_score <= 1, "F-score should be between 0 and 1"
        assert result.n_samples == config.n_samples, "Sample count should match config"
        assert len(result.confidence_interval) == 2, "CI should have two values"
        
        print("✓ Result validation passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Faithfulness computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hardware_optimization():
    """Test hardware optimization and device handling."""
    print("\n=== Testing Hardware Optimization ===")
    
    try:
        from src.config import get_device, get_batch_size, print_system_info
        
        # Test device detection
        device = get_device()
        print(f"✓ Device detected: {device}")
        
        # Test batch size optimization
        batch_size = get_batch_size()
        print(f"✓ Optimal batch size: {batch_size}")
        
        # Test system info
        print("✓ System information:")
        print_system_info()
        
        # Test tensor operations on detected device
        test_tensor = torch.randn(10, 10)
        try:
            test_tensor = test_tensor.to(device)
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"✓ Tensor operations successful on {device}")
        except Exception as e:
            print(f"⚠️  Tensor operations failed on {device}, falling back to CPU: {e}")
            test_tensor = test_tensor.cpu()
            result = torch.matmul(test_tensor, test_tensor.T)
            print("✓ Tensor operations successful on CPU")
        
        return True
        
    except Exception as e:
        print(f"✗ Hardware optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    print("\n=== Testing End-to-End Workflow ===")
    
    try:
        from src.faithfulness import compute_faithfulness_score, FaithfulnessConfig
        from src.explainers import RandomExplainer
        from src.datasets import DatasetSample
        
        # Create multiple explainers for comparison
        explainers = {
            'random_high': RandomExplainer(random_seed=42, distribution="uniform", scale=1.0),
            'random_low': RandomExplainer(random_seed=43, distribution="uniform", scale=0.1),
            'random_normal': RandomExplainer(random_seed=44, distribution="normal", scale=1.0)
        }
        
        # Create test model
        def sentiment_model(x):
            if isinstance(x, torch.Tensor):
                # Simple sentiment: positive if mean > 0
                mean_val = torch.mean(x, dim=-1, keepdim=True)
                return torch.softmax(torch.cat([-mean_val, mean_val], dim=-1), dim=-1)
            return torch.tensor([[0.5, 0.5]])
        
        # Create test samples
        test_samples = [
            torch.randn(1, 20) * 2 + 1,  # Positive bias
            torch.randn(1, 20) * 2 - 1,  # Negative bias
            torch.randn(1, 20) * 0.1,    # Near zero
        ]
        
        # Configuration
        config = FaithfulnessConfig(
            n_samples=50,
            batch_size=8,
            random_seed=42
        )
        
        # Run experiments
        results = {}
        for sample_idx, sample in enumerate(test_samples):
            results[sample_idx] = {}
            for exp_name, explainer in explainers.items():
                result = compute_faithfulness_score(
                    model=sentiment_model,
                    explainer=lambda m, d: explainer.explain(m, d),
                    data=sample,
                    config=config
                )
                results[sample_idx][exp_name] = result
                print(f"✓ Sample {sample_idx}, {exp_name}: F-score = {result.f_score:.4f}")
        
        # Validate results
        total_results = sum(len(sample_results) for sample_results in results.values())
        print(f"✓ Generated {total_results} experimental results")
        
        # Check that all F-scores are valid
        all_valid = True
        for sample_results in results.values():
            for result in sample_results.values():
                if not (0 <= result.f_score <= 1):
                    all_valid = False
                    break
        
        if all_valid:
            print("✓ All F-scores are within valid range [0, 1]")
        else:
            print("✗ Some F-scores are outside valid range")
            return False
        
        print("✓ End-to-end workflow completed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ End-to-end workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all core functionality tests."""
    print("Testing Core Functionality (No External Dependencies)")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Random Explainer Functionality", test_random_explainer_functionality),
        ("Dataset Structures", test_dataset_structures),
        ("Model Structures", test_model_structures),
        ("Masking and Baseline", test_masking_and_baseline),
        ("Faithfulness Computation", test_faithfulness_computation),
        ("Hardware Optimization", test_hardware_optimization),
        ("End-to-End Workflow", test_end_to_end_workflow),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("CORE FUNCTIONALITY TEST SUMMARY")
    print("=" * 60)
    
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
        print("🎉 All core functionality tests passed!")
        print("✅ Implementation is working correctly")
        print("✅ Ready for experimental framework deployment")
        print("✅ Import issues have been resolved")
        return True
    else:
        print("⚠️  Some core functionality tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)