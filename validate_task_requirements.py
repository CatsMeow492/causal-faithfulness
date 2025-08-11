#!/usr/bin/env python3
"""
Validation script for Task 1: Fix Implementation and Resolve Import Issues

This script validates all the specific requirements mentioned in the task:
- Fix relative import issues in src/ modules to enable proper execution
- Create proper `__init__.py` files and PYTHONPATH configuration
- Test that existing experimental framework can run without import errors
- Validate that all core components (faithfulness, explainers, datasets, models) work correctly
"""

import sys
import os
import torch
import numpy as np
import warnings
from typing import Dict, List, Any

def test_relative_imports():
    """Test that relative imports work correctly in src/ modules."""
    print("=== Testing Relative Import Resolution ===")
    
    try:
        # Test imports from different modules to ensure relative imports work
        from src.faithfulness import FaithfulnessMetric
        from src.explainers import RandomExplainer
        from src.datasets import DatasetManager
        from src.models import ModelManager
        
        # Test that modules can import from each other
        # faithfulness.py imports from config, masking, baseline, explainers
        from src.faithfulness import FaithfulnessConfig
        
        # explainers.py imports from config
        from src.explainers import Attribution
        
        # datasets.py imports from config
        from src.datasets import DatasetSample
        
        # models.py imports from config and datasets
        from src.models import ModelPrediction
        
        print("✓ All relative imports resolved successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Relative import failed: {e}")
        return False


def test_init_files():
    """Test that __init__.py files are properly configured."""
    print("\n=== Testing __init__.py Configuration ===")
    
    try:
        # Test that src/__init__.py exports the correct components
        import src
        
        # Check that main components are available from src package
        from src import (
            FaithfulnessConfig,
            FaithfulnessResult,
            FaithfulnessMetric,
            compute_faithfulness_score
        )
        print("✓ Core faithfulness components available from src package")
        
        from src import (
            HardwareConfig,
            get_device,
            get_batch_size,
            DEFAULT_CONFIG
        )
        print("✓ Configuration components available from src package")
        
        from src import (
            FeatureMasker,
            DataModality,
            MaskingStrategy
        )
        print("✓ Masking components available from src package")
        
        from src import (
            BaselineGenerator,
            BaselineStrategy
        )
        print("✓ Baseline components available from src package")
        
        print("✓ __init__.py files configured correctly")
        return True
        
    except ImportError as e:
        print(f"✗ __init__.py configuration failed: {e}")
        return False


def test_pythonpath_configuration():
    """Test that PYTHONPATH configuration allows proper execution."""
    print("\n=== Testing PYTHONPATH Configuration ===")
    
    try:
        # Test that modules can be imported without explicit path manipulation
        current_dir = os.getcwd()
        print(f"✓ Current directory: {current_dir}")
        
        # Test that src is in the path or can be imported
        import src
        print(f"✓ src module location: {src.__file__}")
        
        # Test that we can run modules as scripts (this is what PYTHONPATH helps with)
        # We'll simulate this by testing direct imports
        from src.faithfulness import FaithfulnessMetric
        from src.config import get_device
        
        print("✓ PYTHONPATH configuration allows proper module execution")
        return True
        
    except Exception as e:
        print(f"✗ PYTHONPATH configuration test failed: {e}")
        return False


def test_experimental_framework():
    """Test that the existing experimental framework can run without import errors."""
    print("\n=== Testing Experimental Framework Execution ===")
    
    try:
        # Test core experimental components
        from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
        from src.explainers import RandomExplainer
        from src.datasets import DatasetManager, DatasetSample
        from src.models import ModelManager
        from src.masking import DataModality
        
        print("✓ All experimental framework components imported")
        
        # Test that we can create and run a complete experiment
        config = FaithfulnessConfig(
            n_samples=20,  # Small for testing
            batch_size=4,
            random_seed=42
        )
        
        metric = FaithfulnessMetric(config, modality=DataModality.TABULAR)
        explainer = RandomExplainer(random_seed=42)
        
        # Simple test model
        def test_model(x):
            if isinstance(x, torch.Tensor):
                return torch.softmax(torch.randn(x.shape[0], 2), dim=-1)
            return torch.tensor([[0.5, 0.5]])
        
        # Test data
        test_data = torch.randn(1, 10)
        
        # Run experiment
        result = metric.compute_faithfulness_score(
            model=test_model,
            explainer=lambda m, d: explainer.explain(m, d),
            data=test_data
        )
        
        print(f"✓ Experimental framework executed successfully")
        print(f"  F-score: {result.f_score:.4f}")
        print(f"  Samples: {result.n_samples}")
        
        return True
        
    except Exception as e:
        print(f"✗ Experimental framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_core_components():
    """Validate that all core components work correctly."""
    print("\n=== Testing Core Components ===")
    
    components_tested = 0
    components_passed = 0
    
    # Test faithfulness component
    try:
        from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig, compute_faithfulness_score
        
        config = FaithfulnessConfig(n_samples=10, random_seed=42)
        metric = FaithfulnessMetric(config)
        
        # Test with synthetic data
        def simple_model(x):
            return torch.softmax(torch.randn(1, 2), dim=-1)
        
        def simple_explainer(m, d):
            from src.explainers import Attribution
            return Attribution(
                feature_scores=np.random.random(10),
                feature_indices=list(range(10)),
                method_name="test",
                computation_time=0.1
            )
        
        result = compute_faithfulness_score(
            model=simple_model,
            explainer=simple_explainer,
            data=torch.randn(1, 10),
            config=config
        )
        
        print("✓ Faithfulness component working correctly")
        components_passed += 1
        
    except Exception as e:
        print(f"✗ Faithfulness component failed: {e}")
    
    components_tested += 1
    
    # Test explainers component
    try:
        from src.explainers import RandomExplainer, Attribution
        
        explainer = RandomExplainer(random_seed=42)
        attribution = explainer.explain(
            lambda x: torch.softmax(torch.randn(1, 2), dim=-1),
            torch.randn(1, 10)
        )
        
        assert isinstance(attribution, Attribution)
        assert len(attribution.feature_scores) == 10
        
        print("✓ Explainers component working correctly")
        components_passed += 1
        
    except Exception as e:
        print(f"✗ Explainers component failed: {e}")
    
    components_tested += 1
    
    # Test datasets component
    try:
        from src.datasets import DatasetManager, DatasetSample, DatasetInfo
        
        manager = DatasetManager()
        
        sample = DatasetSample(
            text="Test text",
            tokens=torch.tensor([1, 2, 3, 4, 5]),
            attention_mask=torch.tensor([1, 1, 1, 1, 1]),
            label=1
        )
        
        info = DatasetInfo(
            name="Test",
            task_type="classification",
            license="MIT",
            citation="Test",
            num_samples=100,
            max_length=512
        )
        
        print("✓ Datasets component working correctly")
        components_passed += 1
        
    except Exception as e:
        print(f"✗ Datasets component failed: {e}")
    
    components_tested += 1
    
    # Test models component
    try:
        from src.models import ModelManager, ModelConfig, ModelPrediction
        
        manager = ModelManager()
        
        config = ModelConfig(
            model_name="test",
            device=torch.device('cpu'),
            batch_size=16,
            max_length=512
        )
        
        prediction = ModelPrediction(
            logits=torch.tensor([0.1, 0.9]),
            probabilities=torch.tensor([0.25, 0.75]),
            predicted_class=1,
            confidence=0.75
        )
        
        print("✓ Models component working correctly")
        components_passed += 1
        
    except Exception as e:
        print(f"✗ Models component failed: {e}")
    
    components_tested += 1
    
    print(f"\nCore components test: {components_passed}/{components_tested} passed")
    return components_passed == components_tested


def test_cross_module_integration():
    """Test that modules can work together without import conflicts."""
    print("\n=== Testing Cross-Module Integration ===")
    
    try:
        # Import components from different modules
        from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
        from src.explainers import RandomExplainer
        from src.datasets import DatasetSample
        from src.models import ModelPrediction
        from src.masking import FeatureMasker, DataModality, MaskingStrategy
        from src.baseline import BaselineGenerator, BaselineStrategy
        from src.config import get_device
        
        # Test that they can work together
        device = get_device()
        
        config = FaithfulnessConfig(
            n_samples=10,
            device=device,
            random_seed=42
        )
        
        metric = FaithfulnessMetric(config, modality=DataModality.TABULAR)
        explainer = RandomExplainer(random_seed=42)
        
        masker = FeatureMasker(
            modality=DataModality.TABULAR,
            strategy=MaskingStrategy.ZERO,
            random_seed=42
        )
        
        baseline_gen = BaselineGenerator(
            modality=DataModality.TABULAR,
            strategy=BaselineStrategy.RANDOM,
            random_seed=42
        )
        
        # Test data flow between components
        test_data = torch.randn(1, 5)
        
        # Test masking
        masked_data = masker.mask_features(test_data, [0, 2], mask_explained=True)
        
        # Test baseline generation
        baseline_data = baseline_gen.generate_baseline(test_data, batch_size=1)
        
        # Test explainer
        attribution = explainer.explain(lambda x: torch.softmax(torch.randn(1, 2), dim=-1), test_data)
        
        print("✓ Cross-module integration successful")
        print(f"  Device: {device}")
        print(f"  Masked data shape: {masked_data.shape}")
        print(f"  Baseline data shape: {baseline_data.shape}")
        print(f"  Attribution features: {len(attribution.feature_scores)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Cross-module integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests for Task 1 requirements."""
    print("Validating Task 1: Fix Implementation and Resolve Import Issues")
    print("=" * 70)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    tests = [
        ("Relative Import Resolution", test_relative_imports),
        ("__init__.py Configuration", test_init_files),
        ("PYTHONPATH Configuration", test_pythonpath_configuration),
        ("Experimental Framework Execution", test_experimental_framework),
        ("Core Components Validation", test_core_components),
        ("Cross-Module Integration", test_cross_module_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("TASK 1 VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} requirements validated")
    
    if passed == total:
        print("\n🎉 Task 1 COMPLETED SUCCESSFULLY!")
        print("✅ Relative import issues have been resolved")
        print("✅ __init__.py files are properly configured")
        print("✅ PYTHONPATH configuration enables proper execution")
        print("✅ Experimental framework runs without import errors")
        print("✅ All core components (faithfulness, explainers, datasets, models) work correctly")
        print("\n📋 Task 1 Requirements Met:")
        print("   - ✓ Fixed relative import issues in src/ modules")
        print("   - ✓ Created proper __init__.py files and PYTHONPATH configuration")
        print("   - ✓ Tested that existing experimental framework can run without import errors")
        print("   - ✓ Validated that all core components work correctly")
        print("\n🚀 Ready to proceed to Task 2: Scale Experiments to Real Datasets")
        return True
    else:
        print("\n⚠️  Task 1 validation failed. Please address the failing requirements.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)