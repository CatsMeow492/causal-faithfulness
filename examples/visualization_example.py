#!/usr/bin/env python3
"""
Example script demonstrating the visualization and analysis tools.
Shows how to create all visualizations and generate comprehensive reports.
"""

import os
import sys
import numpy as np
from typing import Dict, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.faithfulness import FaithfulnessResult, FaithfulnessConfig
from src.roar import ROARResult, ROARCorrelationResult
from src.evaluation import ExperimentResult, ExperimentConfig
from src.visualization import ResultVisualizer, VisualizationConfig, create_all_visualizations
from src.analysis import ComprehensiveAnalyzer, analyze_experiment_results


def create_mock_experiment_result() -> ExperimentResult:
    """Create mock experiment result for demonstration."""
    
    # Create mock faithfulness results for different explainers
    np.random.seed(42)
    
    explainer_results = {}
    
    # SHAP results (generally good performance)
    shap_results = []
    for i in range(50):
        f_score = np.clip(np.random.normal(0.75, 0.15), 0, 1)
        result = FaithfulnessResult(
            f_score=f_score,
            confidence_interval=(f_score - 0.05, f_score + 0.05),
            n_samples=1000,
            baseline_performance=0.4,
            explained_performance=0.1,
            statistical_significance=f_score > 0.5,
            p_value=0.01 if f_score > 0.5 else 0.1,
            computation_metrics={
                'computation_time_seconds': np.random.uniform(2, 5),
                'explanation_time': np.random.uniform(0.5, 1.5),
                'faithfulness_time': np.random.uniform(1.5, 3.5),
                'n_model_queries': 2000,
                'sample_index': i,
                'explainer_name': 'SHAP'
            }
        )
        shap_results.append(result)
    explainer_results['SHAP'] = shap_results
    
    # Integrated Gradients results (moderate performance)
    ig_results = []
    for i in range(50):
        f_score = np.clip(np.random.normal(0.60, 0.20), 0, 1)
        result = FaithfulnessResult(
            f_score=f_score,
            confidence_interval=(f_score - 0.08, f_score + 0.08),
            n_samples=1000,
            baseline_performance=0.4,
            explained_performance=0.16,
            statistical_significance=f_score > 0.4,
            p_value=0.02 if f_score > 0.4 else 0.15,
            computation_metrics={
                'computation_time_seconds': np.random.uniform(3, 7),
                'explanation_time': np.random.uniform(1, 2),
                'faithfulness_time': np.random.uniform(2, 5),
                'n_model_queries': 2000,
                'sample_index': i,
                'explainer_name': 'IntegratedGradients'
            }
        )
        ig_results.append(result)
    explainer_results['IntegratedGradients'] = ig_results
    
    # LIME results (variable performance)
    lime_results = []
    for i in range(50):
        f_score = np.clip(np.random.normal(0.45, 0.25), 0, 1)
        result = FaithfulnessResult(
            f_score=f_score,
            confidence_interval=(f_score - 0.10, f_score + 0.10),
            n_samples=1000,
            baseline_performance=0.4,
            explained_performance=0.22,
            statistical_significance=f_score > 0.3,
            p_value=0.05 if f_score > 0.3 else 0.25,
            computation_metrics={
                'computation_time_seconds': np.random.uniform(5, 12),
                'explanation_time': np.random.uniform(2, 5),
                'faithfulness_time': np.random.uniform(3, 7),
                'n_model_queries': 2000,
                'sample_index': i,
                'explainer_name': 'LIME'
            }
        )
        lime_results.append(result)
    explainer_results['LIME'] = lime_results
    
    # Random baseline (poor performance)
    random_results = []
    for i in range(50):
        f_score = np.clip(np.random.normal(0.15, 0.10), 0, 1)
        result = FaithfulnessResult(
            f_score=f_score,
            confidence_interval=(f_score - 0.03, f_score + 0.03),
            n_samples=1000,
            baseline_performance=0.4,
            explained_performance=0.34,
            statistical_significance=False,
            p_value=0.8,
            computation_metrics={
                'computation_time_seconds': np.random.uniform(0.1, 0.5),
                'explanation_time': np.random.uniform(0.01, 0.05),
                'faithfulness_time': np.random.uniform(0.09, 0.45),
                'n_model_queries': 2000,
                'sample_index': i,
                'explainer_name': 'Random'
            }
        )
        random_results.append(result)
    explainer_results['Random'] = random_results
    
    # Create summary statistics
    summary_statistics = {}
    computation_metrics = {}
    
    for explainer_name, results in explainer_results.items():
        f_scores = [r.f_score for r in results]
        p_values = [r.p_value for r in results]
        times = [r.computation_metrics['computation_time_seconds'] for r in results]
        
        summary_statistics[explainer_name] = {
            'mean_f_score': np.mean(f_scores),
            'std_f_score': np.std(f_scores),
            'median_f_score': np.median(f_scores),
            'min_f_score': np.min(f_scores),
            'max_f_score': np.max(f_scores),
            'mean_p_value': np.mean(p_values),
            'significant_results': sum(1 for p in p_values if p < 0.05),
            'total_results': len(results),
            'mean_computation_time': np.mean(times),
            'total_computation_time': np.sum(times)
        }
        
        computation_metrics[explainer_name] = {
            'total_runtime': np.sum(times),
            'samples_processed': len(results),
            'avg_time_per_sample': np.mean(times)
        }
    
    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name="demo_faithfulness_analysis",
        dataset_name="sst2",
        model_name="bert-base-uncased",
        explainer_names=list(explainer_results.keys()),
        num_samples=50,
        batch_size=32,
        random_seed=42
    )
    
    # Create experiment result
    experiment_result = ExperimentResult(
        experiment_config=config,
        dataset_info={
            'name': 'sst2',
            'num_samples': 50,
            'modality': 'TEXT'
        },
        model_info={
            'name': 'bert-base-uncased',
            'device': 'cpu',
            'type': 'BERTSentimentWrapper'
        },
        explainer_results=explainer_results,
        summary_statistics=summary_statistics,
        computation_metrics=computation_metrics,
        timestamp='2024-01-15T10:30:00',
        total_runtime=120.5
    )
    
    return experiment_result


def create_mock_correlation_results() -> List[ROARCorrelationResult]:
    """Create mock ROAR correlation results for demonstration."""
    
    np.random.seed(42)
    correlation_results = []
    
    explainers = ['SHAP', 'IntegratedGradients', 'LIME', 'Random']
    
    for explainer in explainers:
        # Generate correlated ROAR and faithfulness scores
        n_samples = 50
        
        if explainer == 'SHAP':
            # Strong positive correlation
            base_roar = np.random.normal(0.3, 0.1, n_samples)
            base_faith = base_roar * 2 + np.random.normal(0.15, 0.05, n_samples)
            pearson_r = 0.75
            p_value = 0.001
        elif explainer == 'IntegratedGradients':
            # Moderate positive correlation
            base_roar = np.random.normal(0.25, 0.12, n_samples)
            base_faith = base_roar * 1.5 + np.random.normal(0.25, 0.1, n_samples)
            pearson_r = 0.55
            p_value = 0.02
        elif explainer == 'LIME':
            # Weak correlation
            base_roar = np.random.normal(0.2, 0.15, n_samples)
            base_faith = base_roar * 0.8 + np.random.normal(0.3, 0.15, n_samples)
            pearson_r = 0.25
            p_value = 0.15
        else:  # Random
            # No correlation
            base_roar = np.random.normal(0.1, 0.08, n_samples)
            base_faith = np.random.normal(0.15, 0.1, n_samples)
            pearson_r = 0.05
            p_value = 0.85
        
        # Clip to valid ranges
        roar_scores = np.clip(base_roar, 0, 1).tolist()
        faith_scores = np.clip(base_faith, 0, 1).tolist()
        
        result = ROARCorrelationResult(
            explainer_name=explainer,
            pearson_r=pearson_r,
            p_value=p_value,
            spearman_rho=pearson_r * 0.9,  # Slightly lower Spearman
            spearman_p=p_value * 1.2,
            n_samples=n_samples,
            roar_scores=roar_scores,
            faithfulness_scores=faith_scores,
            is_significant=p_value < 0.05
        )
        
        correlation_results.append(result)
    
    return correlation_results


def main():
    """Main demonstration function."""
    print("🎨 Causal-Faithfulness Visualization & Analysis Demo")
    print("=" * 60)
    
    # Create output directories
    os.makedirs("demo_output/figures", exist_ok=True)
    os.makedirs("demo_output/results", exist_ok=True)
    
    # Create mock data
    print("📊 Creating mock experiment data...")
    experiment_result = create_mock_experiment_result()
    correlation_results = create_mock_correlation_results()
    
    print(f"   - Generated results for {len(experiment_result.explainer_results)} explainers")
    print(f"   - {sum(len(results) for results in experiment_result.explainer_results.values())} total samples")
    
    # 1. Create individual visualizations
    print("\n🎯 Creating individual visualizations...")
    
    config = VisualizationConfig(
        output_dir="demo_output/figures",
        figure_format="png",
        figure_dpi=300,
        save_data=True
    )
    
    visualizer = ResultVisualizer(config)
    
    # Bar chart
    print("   - Faithfulness scores bar chart...")
    bar_chart_path = visualizer.create_faithfulness_bar_chart(
        experiment_result.explainer_results,
        title="Faithfulness Scores by Explanation Method",
        filename="demo_faithfulness_bar_chart"
    )
    
    # Performance table
    print("   - Performance metrics table...")
    table_path = visualizer.create_performance_table(
        experiment_result,
        filename="demo_performance_table"
    )
    
    # Correlation scatter plot
    print("   - ROAR correlation scatter plot...")
    scatter_path = visualizer.create_roar_correlation_scatter(
        correlation_results,
        title="Faithfulness vs ROAR Correlation Analysis",
        filename="demo_roar_correlation"
    )
    
    # Comparison heatmap
    print("   - Comparison heatmap...")
    heatmap_path = visualizer.create_comparison_heatmap(
        experiment_result.explainer_results,
        title="Faithfulness Score Comparison Heatmap",
        filename="demo_comparison_heatmap"
    )
    
    # Summary dashboard
    print("   - Summary dashboard...")
    dashboard_path = visualizer.create_summary_dashboard(
        experiment_result,
        correlation_results,
        filename="demo_summary_dashboard"
    )
    
    # 2. Create all visualizations at once
    print("\n🚀 Creating complete visualization suite...")
    all_paths = create_all_visualizations(
        experiment_result,
        correlation_results,
        output_dir="demo_output/figures"
    )
    
    # 3. Comprehensive analysis
    print("\n🔍 Performing comprehensive analysis...")
    
    analysis_results = analyze_experiment_results(
        experiment_result,
        correlation_results,
        output_dir="demo_output/results"
    )
    
    # 4. Generate custom analysis report
    print("   - Generating detailed analysis report...")
    
    analyzer = ComprehensiveAnalyzer()
    custom_analysis = analyzer.analyze_experiment(experiment_result, correlation_results)
    
    report = analyzer.generate_report(
        custom_analysis,
        output_path="demo_output/results/custom_analysis_report.md"
    )
    
    # 5. Display summary results
    print("\n📈 Analysis Summary:")
    print("-" * 40)
    
    if 'findings' in custom_analysis:
        findings = custom_analysis['findings']
        print(f"🏆 Best Performer: {findings['highest_faithfulness_explainer']} "
              f"(F = {findings['highest_faithfulness_score']:.4f})")
        print(f"🎯 Most Consistent: {findings['most_consistent_explainer']} "
              f"(σ = {findings['most_consistent_std']:.4f})")
        print(f"📊 Most Significant: {findings['most_significant_explainer']} "
              f"({findings['most_significant_rate']:.1%} significant)")
        
        print(f"\n🔍 Key Insights:")
        for insight in findings['key_insights'][:3]:  # Show top 3
            print(f"   • {insight}")
    
    if 'correlation_analysis' in custom_analysis:
        corr = custom_analysis['correlation_analysis']
        print(f"\n🔗 ROAR Correlation: r = {corr['overall_correlation']:.3f} "
              f"(p = {corr['overall_p_value']:.3f})")
        if corr['strong_correlations']:
            print(f"   Strong correlations: {', '.join(corr['strong_correlations'])}")
    
    # 6. List generated files
    print(f"\n📁 Generated Files:")
    print("-" * 40)
    
    # Figures
    figure_files = [f for f in os.listdir("demo_output/figures") if f.endswith(('.png', '.csv'))]
    print(f"📊 Figures ({len(figure_files)} files):")
    for file in sorted(figure_files):
        print(f"   • demo_output/figures/{file}")
    
    # Results
    result_files = [f for f in os.listdir("demo_output/results") if f.endswith(('.json', '.md', '.csv'))]
    print(f"\n📋 Analysis Results ({len(result_files)} files):")
    for file in sorted(result_files):
        print(f"   • demo_output/results/{file}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"   Check the 'demo_output' directory for all generated files.")
    
    return {
        'experiment_result': experiment_result,
        'correlation_results': correlation_results,
        'analysis_results': custom_analysis,
        'visualization_paths': all_paths
    }


if __name__ == "__main__":
    results = main()