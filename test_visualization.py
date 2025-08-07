#!/usr/bin/env python3
"""
Simple test script for visualization and analysis components.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Test basic imports
try:
    from src.faithfulness import FaithfulnessResult
    from src.visualization import ResultVisualizer, VisualizationConfig
    from src.analysis import StatisticalAnalyzer
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality of visualization and analysis components."""
    
    # Create test data
    print("📊 Creating test data...")
    
    # Mock faithfulness results
    test_results = {
        'SHAP': [
            FaithfulnessResult(
                f_score=0.75,
                confidence_interval=(0.70, 0.80),
                n_samples=1000,
                baseline_performance=0.4,
                explained_performance=0.1,
                statistical_significance=True,
                p_value=0.01,
                computation_metrics={'computation_time_seconds': 2.5}
            ),
            FaithfulnessResult(
                f_score=0.72,
                confidence_interval=(0.67, 0.77),
                n_samples=1000,
                baseline_performance=0.4,
                explained_performance=0.11,
                statistical_significance=True,
                p_value=0.02,
                computation_metrics={'computation_time_seconds': 2.3}
            )
        ],
        'Random': [
            FaithfulnessResult(
                f_score=0.15,
                confidence_interval=(0.10, 0.20),
                n_samples=1000,
                baseline_performance=0.4,
                explained_performance=0.34,
                statistical_significance=False,
                p_value=0.8,
                computation_metrics={'computation_time_seconds': 0.1}
            ),
            FaithfulnessResult(
                f_score=0.12,
                confidence_interval=(0.07, 0.17),
                n_samples=1000,
                baseline_performance=0.4,
                explained_performance=0.35,
                statistical_significance=False,
                p_value=0.9,
                computation_metrics={'computation_time_seconds': 0.1}
            )
        ]
    }
    
    # Test statistical analysis
    print("🔍 Testing statistical analysis...")
    analyzer = StatisticalAnalyzer()
    
    for explainer_name, results in test_results.items():
        summary = analyzer.compute_explainer_summary(results, explainer_name)
        print(f"   {explainer_name}: F = {summary.mean_f_score:.3f} ± {summary.std_f_score:.3f}")
    
    # Test visualization
    print("🎨 Testing visualization...")
    
    # Create output directory
    os.makedirs("test_output", exist_ok=True)
    
    config = VisualizationConfig(
        output_dir="test_output",
        figure_format="png",
        figure_size=(8, 6)
    )
    
    visualizer = ResultVisualizer(config)
    
    # Create bar chart
    try:
        bar_path = visualizer.create_faithfulness_bar_chart(
            test_results,
            title="Test Faithfulness Scores",
            filename="test_bar_chart"
        )
        print(f"   ✅ Bar chart created: {bar_path}")
    except Exception as e:
        print(f"   ❌ Bar chart failed: {e}")
    
    # Test comparison
    print("📈 Testing explainer comparison...")
    comparisons = analyzer.compare_explainers(test_results)
    
    if comparisons:
        for comp_name, comp_result in comparisons.items():
            winner = comp_result['winner']
            p_val = comp_result['t_p_value']
            print(f"   {comp_name}: {winner} wins (p = {p_val:.3f})")
    
    print("✅ All tests completed successfully!")
    
    # List generated files
    if os.path.exists("test_output"):
        files = os.listdir("test_output")
        if files:
            print(f"\n📁 Generated files:")
            for file in files:
                print(f"   • test_output/{file}")

if __name__ == "__main__":
    test_basic_functionality()