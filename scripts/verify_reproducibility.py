#!/usr/bin/env python3
"""
Reproducibility Verification Script

This script verifies that the reproducibility framework is working correctly
by running the same computation multiple times and checking for identical results.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from reproducibility import ensure_reproducibility, ReproducibilityManager, ReproducibilityConfig
from faithfulness import FaithfulnessMetric, FaithfulnessConfig
from explainers import RandomExplainer
from masking import DataModality


def create_dummy_model():
    """Create a simple deterministic model for testing."""
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)
            # Initialize with fixed weights for determinism
            torch.nn.init.constant_(self.linear.weight, 0.1)
            torch.nn.init.constant_(self.linear.bias, 0.0)
        
        def forward(self, x):
            return self.linear(x)
    
    return DummyModel()


def run_reproducibility_test(seed=42, n_runs=3):
    """Run the same computation multiple times to verify reproducibility."""
    
    print(f"Running reproducibility test with seed {seed} ({n_runs} runs)...")
    
    results = []
    
    for run_idx in range(n_runs):
        print(f"  Run {run_idx + 1}/{n_runs}")
        
        # Initialize reproducibility for this run
        metadata = ensure_reproducibility(seed=seed)
        
        # Create model and data
        model = create_dummy_model()
        model.eval()
        
        # Create fixed test data
        torch.manual_seed(seed)  # Ensure same data across runs
        test_data = torch.randn(1, 10)
        
        # Configure faithfulness metric
        config = FaithfulnessConfig(
            n_samples=100,  # Small for fast testing
            baseline_strategy="gaussian",
            masking_strategy="mean",
            confidence_level=0.95,
            batch_size=8,
            random_seed=seed
        )
        
        metric = FaithfulnessMetric(config, modality=DataModality.TABULAR)
        
        # Create explainer
        explainer = RandomExplainer(random_seed=seed, distribution="uniform")
        
        # Compute faithfulness score
        result = metric.compute_faithfulness_score(
            model=model,
            explainer=explainer.explain,
            data=test_data,
            target_class=None
        )
        
        # Store key results
        run_result = {
            'f_score': result.f_score,
            'baseline_performance': result.baseline_performance,
            'explained_performance': result.explained_performance,
            'p_value': result.p_value,
            'n_samples': result.n_samples
        }
        
        results.append(run_result)
        
        print(f"    F-score: {result.f_score:.6f}")
        print(f"    P-value: {result.p_value:.6f}")
    
    return results


def verify_results_identical(results):
    """Verify that all results are identical."""
    
    print("\nVerifying result consistency...")
    
    if len(results) < 2:
        print("  Need at least 2 runs to verify consistency")
        return False
    
    reference = results[0]
    all_identical = True
    
    for i, result in enumerate(results[1:], 1):
        print(f"  Comparing run 1 vs run {i+1}:")
        
        for key in reference.keys():
            ref_val = reference[key]
            curr_val = result[key]
            
            if isinstance(ref_val, float):
                # Use small tolerance for floating point comparison
                identical = abs(ref_val - curr_val) < 1e-10
            else:
                identical = ref_val == curr_val
            
            status = "✓" if identical else "✗"
            print(f"    {key}: {status} ({ref_val} vs {curr_val})")
            
            if not identical:
                all_identical = False
    
    return all_identical


def test_cross_platform_determinism():
    """Test deterministic behavior across different settings."""
    
    print("\nTesting cross-platform determinism...")
    
    # Test with different device settings
    devices_to_test = ['cpu']
    
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices_to_test.append('mps')
    
    device_results = {}
    
    for device_name in devices_to_test:
        print(f"  Testing on device: {device_name}")
        
        try:
            device = torch.device(device_name)
            
            # Initialize reproducibility
            ensure_reproducibility(seed=42)
            
            # Create model and move to device
            model = create_dummy_model()
            model.to(device)
            model.eval()
            
            # Create test data
            torch.manual_seed(42)
            test_data = torch.randn(1, 10).to(device)
            
            # Simple computation
            with torch.no_grad():
                output = model(test_data)
                result = torch.sum(output).item()
            
            device_results[device_name] = result
            print(f"    Result: {result:.10f}")
            
        except Exception as e:
            print(f"    Error on {device_name}: {e}")
            device_results[device_name] = None
    
    # Check consistency across devices (may differ due to hardware)
    valid_results = {k: v for k, v in device_results.items() if v is not None}
    
    if len(valid_results) > 1:
        print(f"  Device result consistency:")
        reference_device = list(valid_results.keys())[0]
        reference_result = valid_results[reference_device]
        
        for device_name, result in valid_results.items():
            if device_name != reference_device:
                diff = abs(result - reference_result)
                consistent = diff < 1e-6  # Looser tolerance for cross-device
                status = "✓" if consistent else "⚠"
                print(f"    {reference_device} vs {device_name}: {status} (diff: {diff:.2e})")
    
    return device_results


def test_package_versions():
    """Test that required packages are available and at correct versions."""
    
    print("\nTesting package versions...")
    
    required_packages = {
        'torch': '2.1.2',
        'numpy': '1.24.4',
        'scipy': '1.11.4',
        'scikit-learn': '1.3.2'
    }
    
    version_check_passed = True
    
    for package_name, expected_version in required_packages.items():
        try:
            if package_name == 'torch':
                import torch
                actual_version = torch.__version__
            elif package_name == 'numpy':
                import numpy
                actual_version = numpy.__version__
            elif package_name == 'scipy':
                import scipy
                actual_version = scipy.__version__
            elif package_name == 'scikit-learn':
                import sklearn
                actual_version = sklearn.__version__
            
            # Check if versions match (allowing for patch version differences)
            version_parts = actual_version.split('.')
            expected_parts = expected_version.split('.')
            
            major_minor_match = (
                version_parts[0] == expected_parts[0] and
                version_parts[1] == expected_parts[1]
            )
            
            status = "✓" if major_minor_match else "⚠"
            print(f"  {package_name}: {status} {actual_version} (expected: {expected_version})")
            
            if not major_minor_match:
                version_check_passed = False
                
        except ImportError:
            print(f"  {package_name}: ✗ Not installed")
            version_check_passed = False
    
    return version_check_passed


def main(quick_test=False):
    """Main function to run all reproducibility tests."""
    
    test_mode = "Quick Test" if quick_test else "Full Verification"
    print(f"=== Reproducibility {test_mode} ===\n")
    
    # Test 1: Basic reproducibility
    print("1. Testing basic reproducibility...")
    n_runs = 2 if quick_test else 3
    results = run_reproducibility_test(seed=42, n_runs=n_runs)
    basic_reproducibility = verify_results_identical(results)
    
    # Test 2: Different seeds produce different results
    print("\n2. Testing seed variation...")
    results_seed_42 = run_reproducibility_test(seed=42, n_runs=1)[0]
    results_seed_123 = run_reproducibility_test(seed=123, n_runs=1)[0]
    
    seed_variation = results_seed_42['f_score'] != results_seed_123['f_score']
    print(f"  Different seeds produce different results: {'✓' if seed_variation else '✗'}")
    print(f"    Seed 42 F-score: {results_seed_42['f_score']:.6f}")
    print(f"    Seed 123 F-score: {results_seed_123['f_score']:.6f}")
    
    # Test 3: Cross-platform determinism (skip in quick mode)
    if not quick_test:
        device_results = test_cross_platform_determinism()
    else:
        print("\n3. Skipping cross-platform determinism test (quick mode)")
        device_results = {}
    
    # Test 4: Package versions
    version_check = test_package_versions()
    
    # Test 5: Reproducibility manager functionality
    test_num = 4 if quick_test else 5
    print(f"\n{test_num}. Testing ReproducibilityManager...")
    
    try:
        config = ReproducibilityConfig(global_seed=42)
        manager = ReproducibilityManager(config)
        metadata = manager.initialize_reproducibility("test_experiment")
        
        print(f"  Experiment ID: {metadata.experiment_id}")
        print(f"  Seeds: {metadata.seeds}")
        print(f"  Config hash: {metadata.config_hash}")
        
        # Test metadata saving/loading
        test_metadata_file = "test_metadata.json"
        manager.save_metadata(test_metadata_file)
        loaded_metadata = manager.load_metadata(test_metadata_file)
        
        metadata_consistency = (
            loaded_metadata.config_hash == metadata.config_hash and
            loaded_metadata.seeds == metadata.seeds
        )
        
        print(f"  Metadata save/load: {'✓' if metadata_consistency else '✗'}")
        
        # Clean up
        import os
        if os.path.exists(test_metadata_file):
            os.remove(test_metadata_file)
        
        manager_test_passed = True
        
    except Exception as e:
        print(f"  ReproducibilityManager test failed: {e}")
        manager_test_passed = False
    
    # Summary
    print(f"\n{'='*50}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*50}")
    
    tests = [
        ("Basic reproducibility", basic_reproducibility),
        ("Seed variation", seed_variation),
        ("Package versions", version_check),
        ("ReproducibilityManager", manager_test_passed)
    ]
    
    all_passed = True
    for test_name, passed in tests:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:<25}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall status: {'PASS' if all_passed else 'FAIL'}")
    
    if all_passed:
        print("\n✓ Reproducibility verification completed successfully!")
        print("  The framework is properly configured for reproducible experiments.")
    else:
        print("\n✗ Reproducibility verification failed!")
        print("  Please check the failed tests and fix any issues before running experiments.")
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify reproducibility framework")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run quick test mode for CI")
    
    args = parser.parse_args()
    success = main(quick_test=args.quick_test)
    sys.exit(0 if success else 1)