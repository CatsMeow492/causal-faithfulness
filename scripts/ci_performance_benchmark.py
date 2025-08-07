#!/usr/bin/env python3
"""
CI Performance Benchmark

This script runs performance benchmarks for the faithfulness metric
to track performance regressions and ensure reasonable computation times.
"""

import sys
import os
import time
import json
import psutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification

from faithfulness import FaithfulnessMetric, FaithfulnessConfig
from explainers import SHAPWrapper, IntegratedGradientsWrapper, RandomExplainer
from masking import DataModality
from version_info import get_hardware_info, get_framework_version


class BenchmarkModel(nn.Module):
    """Benchmark neural network model."""
    
    def __init__(self, input_dim=20, hidden_dim=50, output_dim=2):
        super(BenchmarkModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def create_benchmark_data_and_model():
    """Create benchmark dataset and model."""
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate larger synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create and train model
    model = BenchmarkModel(input_dim=20)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    
    # Select test instances
    test_data = X_tensor[:20]  # Use 20 test instances
    
    return model, X, test_data


def benchmark_explainer(name, explainer, model, test_data, background_data, config):
    """Benchmark a single explainer."""
    
    print(f"   Benchmarking {name}...")
    
    # Memory before
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Time the computation
    start_time = time.time()
    
    try:
        faithfulness_metric = FaithfulnessMetric(config, modality=DataModality.TABULAR)
        
        # Use subset of test data for benchmarking
        test_subset = test_data[:5]  # 5 instances
        
        if name == "SHAP":
            result = faithfulness_metric.compute_faithfulness_score(
                model=model,
                explainer=lambda m, d: explainer.explain(m, d, background_data=background_data),
                data=test_subset,
                target_class=None
            )
        else:
            result = faithfulness_metric.compute_faithfulness_score(
                model=model,
                explainer=explainer.explain,
                data=test_subset,
                target_class=None
            )
        
        computation_time = time.time() - start_time
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Calculate throughput
        throughput = len(test_subset) / computation_time  # instances per second
        
        benchmark_result = {
            "success": True,
            "f_score": float(result.f_score),
            "computation_time_seconds": computation_time,
            "memory_used_mb": memory_used,
            "throughput_instances_per_second": throughput,
            "n_samples": config.n_samples,
            "n_test_instances": len(test_subset)
        }
        
        print(f"      Time: {computation_time:.2f}s")
        print(f"      Memory: {memory_used:.1f} MB")
        print(f"      Throughput: {throughput:.2f} instances/sec")
        print(f"      F-score: {result.f_score:.4f}")
        
        return benchmark_result
        
    except Exception as e:
        computation_time = time.time() - start_time
        print(f"      ERROR: {e}")
        
        return {
            "success": False,
            "error": str(e),
            "computation_time_seconds": computation_time,
            "memory_used_mb": 0,
            "throughput_instances_per_second": 0
        }


def run_performance_benchmark():
    """Run the performance benchmark suite."""
    
    print("=== CI Performance Benchmark ===")
    
    # Get system information
    hardware_info = get_hardware_info()
    print(f"Framework version: {get_framework_version()}")
    print(f"Platform: {hardware_info['platform']} {hardware_info['platform_release']}")
    print(f"CPU count: {hardware_info.get('cpu_count_logical', 'unknown')}")
    print(f"Memory: {hardware_info.get('memory_total_gb', 'unknown')} GB")
    
    if hardware_info.get('mps_available'):
        print("MPS (Apple Silicon): Available")
    if hardware_info.get('cuda_available'):
        print(f"CUDA: Available (version {hardware_info.get('cuda_version')})")
    
    # Create benchmark data and model
    print("\n1. Creating benchmark model and data...")
    model, X_train, test_data = create_benchmark_data_and_model()
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"   Test data shape: {test_data.shape}")
    
    # Configure benchmark parameters
    print("\n2. Configuring benchmark parameters...")
    
    # Different configurations to test
    configs = {
        "fast": FaithfulnessConfig(
            n_samples=200,
            baseline_strategy="gaussian",
            masking_strategy="mean",
            confidence_level=0.95,
            random_seed=42,
            batch_size=16
        ),
        "standard": FaithfulnessConfig(
            n_samples=500,
            baseline_strategy="gaussian",
            masking_strategy="mean",
            confidence_level=0.95,
            random_seed=42,
            batch_size=32
        )
    }
    
    # Initialize explainers
    print("\n3. Initializing explainers...")
    
    background_data = torch.FloatTensor(X_train[:50])
    
    explainers = {
        "SHAP": SHAPWrapper(
            explainer_type="kernel",
            n_samples=100,
            random_seed=42
        ),
        "IntegratedGradients": IntegratedGradientsWrapper(
            n_steps=20,
            baseline_strategy="zero",
            random_seed=42
        ),
        "Random": RandomExplainer(
            random_seed=42,
            distribution="uniform"
        )
    }
    
    print(f"   Initialized {len(explainers)} explainers")
    print(f"   Testing {len(configs)} configurations")
    
    # Run benchmarks
    print("\n4. Running benchmarks...")
    
    benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework_version": get_framework_version(),
        "hardware_info": hardware_info,
        "configurations": {},
        "summary": {}
    }
    
    for config_name, config in configs.items():
        print(f"\n   Configuration: {config_name} ({config.n_samples} samples)")
        
        config_results = {}
        
        for explainer_name, explainer in explainers.items():
            result = benchmark_explainer(
                explainer_name, explainer, model, test_data, 
                background_data, config
            )
            config_results[explainer_name] = result
        
        benchmark_results["configurations"][config_name] = config_results
    
    # Calculate summary statistics
    print("\n5. Summary Statistics:")
    print("   " + "="*70)
    print(f"   {'Config':<12} {'Explainer':<18} {'Time (s)':<10} {'Memory (MB)':<12} {'Throughput':<12}")
    print("   " + "="*70)
    
    total_times = []
    successful_runs = 0
    
    for config_name, config_results in benchmark_results["configurations"].items():
        for explainer_name, result in config_results.items():
            if result["success"]:
                time_str = f"{result['computation_time_seconds']:.2f}"
                memory_str = f"{result['memory_used_mb']:.1f}"
                throughput_str = f"{result['throughput_instances_per_second']:.2f}"
                
                print(f"   {config_name:<12} {explainer_name:<18} {time_str:<10} {memory_str:<12} {throughput_str:<12}")
                
                total_times.append(result['computation_time_seconds'])
                successful_runs += 1
            else:
                print(f"   {config_name:<12} {explainer_name:<18} {'FAILED':<10} {'-':<12} {'-':<12}")
    
    print("   " + "="*70)
    
    # Overall summary
    if total_times:
        avg_time = np.mean(total_times)
        max_time = np.max(total_times)
        
        benchmark_results["summary"] = {
            "successful_runs": successful_runs,
            "total_runs": len(configs) * len(explainers),
            "average_time_seconds": float(avg_time),
            "max_time_seconds": float(max_time),
            "success_rate": successful_runs / (len(configs) * len(explainers))
        }
        
        print(f"\n   Successful runs: {successful_runs}/{len(configs) * len(explainers)}")
        print(f"   Average time: {avg_time:.2f}s")
        print(f"   Maximum time: {max_time:.2f}s")
        print(f"   Success rate: {benchmark_results['summary']['success_rate']:.1%}")
        
        # Performance thresholds
        if max_time > 60:  # 1 minute per explainer
            print("   ⚠ WARNING: Some computations exceeded 60 seconds")
        else:
            print("   ✓ All computations completed within reasonable time")
        
        if benchmark_results['summary']['success_rate'] < 0.8:
            print("   ⚠ WARNING: Success rate below 80%")
        else:
            print("   ✓ Good success rate")
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    print(f"\n   Benchmark results saved to benchmark_results.json")
    
    return benchmark_results


if __name__ == "__main__":
    try:
        results = run_performance_benchmark()
        
        # Check if benchmark was successful
        summary = results.get("summary", {})
        success_rate = summary.get("success_rate", 0)
        max_time = summary.get("max_time_seconds", float('inf'))
        
        if success_rate >= 0.8 and max_time <= 120:  # 2 minutes max
            print("\n🎉 Performance benchmark completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠ Performance benchmark completed with warnings!")
            sys.exit(0)  # Don't fail CI for performance warnings
            
    except Exception as e:
        print(f"\n💥 Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)