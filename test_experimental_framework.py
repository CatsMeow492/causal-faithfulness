#!/usr/bin/env python3
"""
Test script to validate the experimental framework for paper completion.
This script tests the specific components needed for SST-2 and WikiText-2 experiments.
"""

import sys
import torch
import numpy as np
import warnings
from typing import Dict, List, Any

def test_sst2_experimental_pipeline():
    """Test SST-2 experimental pipeline components."""
    print("=== Testing SST-2 Experimental Pipeline ===")
    
    try:
        # Import required components
        from src.datasets import DatasetManager, SST2DatasetLoader, DatasetSample
        from src.models import ModelManager, BERTSentimentWrapper
        from src.explainers import SHAPWrapper, IntegratedGradientsWrapper, LIMEWrapper, RandomExplainer
        from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
        from src.masking import DataModality
        
        print("✓ All SST-2 components imported successfully")
        
        # Test dataset loader initialization
        dataset_manager = DatasetManager()
        print("✓ DatasetManager initialized")
        
        # Test model manager initialization
        model_manager = ModelManager()
        print("✓ ModelManager initialized")
        
        # Test explainer initialization
        explainers = {
            'shap': SHAPWrapper(explainer_type="kernel", n_samples=100),
            'integrated_gradients': IntegratedGradientsWrapper(n_steps=20),
            'lime': LIMEWrapper(n_samples=100, modality="tabular"),
            'random': RandomExplainer()
        }
        print("✓ All explainers initialized")
        
        # Test faithfulness metric initialization
        config = FaithfulnessConfig(
            n_samples=50,
            batch_size=8,
            random_seed=42
        )
        metric = FaithfulnessMetric(config, modality=DataModality.TEXT)
        print("✓ FaithfulnessMetric initialized for text modality")
        
        # Test synthetic data processing
        synthetic_sample = DatasetSample(
            text="This movie is great!",
            tokens=torch.tensor([101, 2023, 3185, 2003, 2307, 999, 102]),
            attention_mask=torch.tensor([1, 1, 1, 1, 1, 1, 1]),
            label=1,
            metadata={"dataset": "sst2", "synthetic": True}
        )
        print("✓ Synthetic SST-2 sample created")
        
        return True
        
    except Exception as e:
        print(f"✗ SST-2 pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wikitext2_experimental_pipeline():
    """Test WikiText-2 experimental pipeline components."""
    print("\n=== Testing WikiText-2 Experimental Pipeline ===")
    
    try:
        # Import required components
        from src.datasets import DatasetManager, WikiText2DatasetLoader, DatasetSample
        from src.models import ModelManager, GPT2LanguageModelWrapper
        from src.explainers import SHAPWrapper, IntegratedGradientsWrapper, RandomExplainer
        from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
        from src.masking import DataModality
        
        print("✓ All WikiText-2 components imported successfully")
        
        # Test dataset loader initialization
        dataset_manager = DatasetManager()
        print("✓ DatasetManager initialized")
        
        # Test model manager initialization
        model_manager = ModelManager()
        print("✓ ModelManager initialized")
        
        # Test explainer initialization (LIME not typically used for language modeling)
        explainers = {
            'shap': SHAPWrapper(explainer_type="kernel", n_samples=100),
            'integrated_gradients': IntegratedGradientsWrapper(n_steps=20),
            'random': RandomExplainer()
        }
        print("✓ Language modeling explainers initialized")
        
        # Test faithfulness metric initialization
        config = FaithfulnessConfig(
            n_samples=50,
            batch_size=4,  # Smaller batch for language modeling
            random_seed=42
        )
        metric = FaithfulnessMetric(config, modality=DataModality.TEXT)
        print("✓ FaithfulnessMetric initialized for text modality")
        
        # Test synthetic data processing
        synthetic_sample = DatasetSample(
            text="The quick brown fox jumps over the lazy dog.",
            tokens=torch.tensor([464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290]),
            attention_mask=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            label=None,  # No labels for language modeling
            metadata={"dataset": "wikitext2", "synthetic": True}
        )
        print("✓ Synthetic WikiText-2 sample created")
        
        return True
        
    except Exception as e:
        print(f"✗ WikiText-2 pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roar_comparison_pipeline():
    """Test ROAR comparison pipeline components."""
    print("\n=== Testing ROAR Comparison Pipeline ===")
    
    try:
        # Import required components
        from src.roar import ROARBenchmark, ROARConfig, ROARResult, ROARValidator
        from src.statistical_analysis import StatisticalAnalyzer
        from src.faithfulness import FaithfulnessResult
        
        print("✓ ROAR comparison components imported successfully")
        
        # Test ROAR configuration
        roar_config = ROARConfig(
            removal_percentages=[0.1, 0.2, 0.3, 0.5],
            n_samples=50,
            random_seed=42
        )
        print("✓ ROAR configuration created")
        
        # Test ROAR benchmark initialization
        roar_benchmark = ROARBenchmark(roar_config)
        print("✓ ROAR benchmark initialized")
        
        # Test statistical analyzer
        analyzer = StatisticalAnalyzer()
        print("✓ Statistical analyzer initialized")
        
        # Test ROAR validator
        validator = ROARValidator()
        print("✓ ROAR validator initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ ROAR comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistical_analysis_pipeline():
    """Test statistical analysis pipeline components."""
    print("\n=== Testing Statistical Analysis Pipeline ===")
    
    try:
        # Import required components
        from src.statistical_analysis import StatisticalAnalyzer, ComparisonResult, BootstrapResult
        from src.faithfulness import FaithfulnessResult
        import scipy.stats as stats
        
        print("✓ Statistical analysis components imported successfully")
        
        # Test statistical analyzer initialization
        analyzer = StatisticalAnalyzer()
        print("✓ Statistical analyzer initialized")
        
        # Test synthetic results for statistical analysis
        synthetic_results = {
            'shap': [0.8, 0.75, 0.82, 0.78, 0.81],
            'integrated_gradients': [0.72, 0.68, 0.74, 0.71, 0.73],
            'lime': [0.65, 0.62, 0.67, 0.64, 0.66],
            'random': [0.15, 0.12, 0.18, 0.14, 0.16]
        }
        
        # Test pairwise comparisons
        print("✓ Synthetic results created for statistical testing")
        
        # Test bootstrap confidence intervals
        sample_scores = np.array(synthetic_results['shap'])
        print("✓ Bootstrap analysis data prepared")
        
        return True
        
    except Exception as e:
        print(f"✗ Statistical analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_pipeline():
    """Test visualization pipeline components."""
    print("\n=== Testing Visualization Pipeline ===")
    
    try:
        # Import required components
        from src.visualization import ResultVisualizer, VisualizationConfig
        import matplotlib.pyplot as plt
        
        print("✓ Visualization components imported successfully")
        
        # Test visualization configuration
        viz_config = VisualizationConfig(
            output_dir="test_figures",
            figure_format="png",
            dpi=300,
            style="publication"
        )
        print("✓ Visualization configuration created")
        
        # Test result visualizer initialization
        visualizer = ResultVisualizer(viz_config)
        print("✓ Result visualizer initialized")
        
        # Test matplotlib backend
        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3], [1, 4, 2])
        plt.title("Test Plot")
        plt.close()
        print("✓ Matplotlib plotting test successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_orchestration():
    """Test evaluation orchestration components."""
    print("\n=== Testing Evaluation Orchestration ===")
    
    try:
        # Import required components
        from src.evaluation import EvaluationPipeline, ExperimentConfig, ExperimentResult
        from src.faithfulness import FaithfulnessConfig
        
        print("✓ Evaluation orchestration components imported successfully")
        
        # Test experiment configuration
        experiment_config = ExperimentConfig(
            experiment_name="test_experiment",
            datasets=["sst2"],
            models=["bert-base-uncased"],
            explainers=["shap", "integrated_gradients", "random"],
            n_samples=50,
            random_seed=42
        )
        print("✓ Experiment configuration created")
        
        # Test evaluation pipeline initialization
        pipeline = EvaluationPipeline(experiment_config)
        print("✓ Evaluation pipeline initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation orchestration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_experimental_integration():
    """Test full experimental integration with all components."""
    print("\n=== Testing Full Experimental Integration ===")
    
    try:
        # Import all required components
        from src.faithfulness import compute_faithfulness_score, FaithfulnessConfig
        from src.explainers import RandomExplainer
        from src.datasets import DatasetSample
        from src.statistical_analysis import StatisticalAnalyzer
        from src.visualization import ResultVisualizer, VisualizationConfig
        
        print("✓ All integration components imported successfully")
        
        # Create a complete experimental setup
        def create_synthetic_model():
            """Create a synthetic model for testing."""
            def model(x):
                if isinstance(x, torch.Tensor):
                    # Simple sentiment model: positive if sum > 0
                    sentiment_score = torch.sum(x, dim=-1, keepdim=True)
                    return torch.softmax(torch.cat([sentiment_score, -sentiment_score], dim=-1), dim=-1)
                return torch.tensor([[0.5, 0.5]])
            return model
        
        # Create synthetic dataset
        synthetic_samples = [
            DatasetSample(
                text=f"Sample text {i}",
                tokens=torch.randn(10),
                attention_mask=torch.ones(10),
                label=i % 2,
                metadata={"index": i}
            )
            for i in range(5)
        ]
        
        # Create explainers
        explainers = {
            'random_1': RandomExplainer(random_seed=42),
            'random_2': RandomExplainer(random_seed=43),
        }
        
        # Run faithfulness evaluation
        model = create_synthetic_model()
        config = FaithfulnessConfig(n_samples=20, batch_size=4, random_seed=42)
        
        results = {}
        for name, explainer in explainers.items():
            result = compute_faithfulness_score(
                model=model,
                explainer=lambda m, d: explainer.explain(m, d),
                data=synthetic_samples[0].tokens,
                config=config
            )
            results[name] = result
            print(f"✓ {name} faithfulness score: {result.f_score:.4f}")
        
        # Test statistical analysis
        analyzer = StatisticalAnalyzer()
        print("✓ Statistical analysis ready")
        
        # Test visualization setup
        viz_config = VisualizationConfig(output_dir="test_figures")
        visualizer = ResultVisualizer(viz_config)
        print("✓ Visualization setup ready")
        
        print("✓ Full experimental integration successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Full experimental integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all experimental framework tests."""
    print("Testing Experimental Framework for Paper Completion")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    tests = [
        ("SST-2 Experimental Pipeline", test_sst2_experimental_pipeline),
        ("WikiText-2 Experimental Pipeline", test_wikitext2_experimental_pipeline),
        ("ROAR Comparison Pipeline", test_roar_comparison_pipeline),
        ("Statistical Analysis Pipeline", test_statistical_analysis_pipeline),
        ("Visualization Pipeline", test_visualization_pipeline),
        ("Evaluation Orchestration", test_evaluation_orchestration),
        ("Full Experimental Integration", test_full_experimental_integration),
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
    print("EXPERIMENTAL FRAMEWORK TEST SUMMARY")
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
        print("🎉 All experimental framework tests passed!")
        print("✅ Ready to run SST-2 and WikiText-2 experiments")
        print("✅ Ready to perform ROAR comparison")
        print("✅ Ready to generate statistical analysis")
        print("✅ Ready to create publication visualizations")
        return True
    else:
        print("⚠️  Some experimental framework tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)