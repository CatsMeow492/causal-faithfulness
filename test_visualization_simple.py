#!/usr/bin/env python3
"""
Simple test script for visualization components without heavy dependencies.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Simple mock classes to avoid dependency issues
@dataclass
class MockFaithfulnessResult:
    """Mock faithfulness result for testing."""
    f_score: float
    confidence_interval: Tuple[float, float]
    n_samples: int
    baseline_performance: float
    explained_performance: float
    statistical_significance: bool
    p_value: float
    computation_metrics: Dict[str, float]

def test_visualization_core():
    """Test core visualization functionality."""
    
    print("🎨 Testing Visualization Core Components")
    print("=" * 50)
    
    # Create test data
    test_results = {
        'SHAP': [
            MockFaithfulnessResult(
                f_score=0.75,
                confidence_interval=(0.70, 0.80),
                n_samples=1000,
                baseline_performance=0.4,
                explained_performance=0.1,
                statistical_significance=True,
                p_value=0.01,
                computation_metrics={'computation_time_seconds': 2.5}
            ),
            MockFaithfulnessResult(
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
        'IntegratedGradients': [
            MockFaithfulnessResult(
                f_score=0.60,
                confidence_interval=(0.55, 0.65),
                n_samples=1000,
                baseline_performance=0.4,
                explained_performance=0.16,
                statistical_significance=True,
                p_value=0.03,
                computation_metrics={'computation_time_seconds': 3.1}
            ),
            MockFaithfulnessResult(
                f_score=0.58,
                confidence_interval=(0.53, 0.63),
                n_samples=1000,
                baseline_performance=0.4,
                explained_performance=0.17,
                statistical_significance=True,
                p_value=0.04,
                computation_metrics={'computation_time_seconds': 3.0}
            )
        ],
        'Random': [
            MockFaithfulnessResult(
                f_score=0.15,
                confidence_interval=(0.10, 0.20),
                n_samples=1000,
                baseline_performance=0.4,
                explained_performance=0.34,
                statistical_significance=False,
                p_value=0.8,
                computation_metrics={'computation_time_seconds': 0.1}
            ),
            MockFaithfulnessResult(
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
    
    # Test 1: Basic bar chart creation
    print("📊 Test 1: Creating bar chart...")
    
    try:
        # Create output directory
        os.makedirs("test_figures", exist_ok=True)
        
        # Prepare data for bar chart
        explainer_names = []
        mean_scores = []
        std_scores = []
        
        for explainer_name, results in test_results.items():
            scores = [r.f_score for r in results]
            explainer_names.append(explainer_name)
            mean_scores.append(np.mean(scores))
            std_scores.append(np.std(scores))
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(explainer_names, mean_scores, yerr=std_scores, 
                     capsize=5, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_title('Mean Faithfulness Scores by Explainer', fontsize=14, fontweight='bold')
        ax.set_xlabel('Explanation Method', fontsize=12)
        ax.set_ylabel('Mean Faithfulness Score (F)', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean_val, std_val in zip(bars, mean_scores, std_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                   f'{mean_val:.3f}±{std_val:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('test_figures/test_bar_chart.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("   ✅ Bar chart created successfully")
        
    except Exception as e:
        print(f"   ❌ Bar chart creation failed: {e}")
    
    # Test 2: Performance table data
    print("📋 Test 2: Creating performance table data...")
    
    try:
        performance_data = []
        
        for explainer_name, results in test_results.items():
            f_scores = [r.f_score for r in results]
            p_values = [r.p_value for r in results]
            times = [r.computation_metrics['computation_time_seconds'] for r in results]
            
            performance_data.append({
                'Explainer': explainer_name,
                'Mean F-Score': f"{np.mean(f_scores):.4f}",
                'Std F-Score': f"{np.std(f_scores):.4f}",
                'Samples': len(results),
                'Significant Results': f"{sum(1 for p in p_values if p < 0.05)}/{len(results)}",
                'Avg Time (s)': f"{np.mean(times):.3f}",
                'Total Time (s)': f"{np.sum(times):.2f}"
            })
        
        # Create simple table visualization
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # Convert to table format
        table_data = []
        headers = list(performance_data[0].keys())
        for row in performance_data:
            table_data.append(list(row.values()))
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Performance Metrics Table', fontsize=14, fontweight='bold', pad=20)
        plt.savefig('test_figures/test_performance_table.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("   ✅ Performance table created successfully")
        
    except Exception as e:
        print(f"   ❌ Performance table creation failed: {e}")
    
    # Test 3: Statistical analysis
    print("🔍 Test 3: Statistical analysis...")
    
    try:
        for explainer_name, results in test_results.items():
            f_scores = np.array([r.f_score for r in results])
            p_values = np.array([r.p_value for r in results])
            
            mean_f = np.mean(f_scores)
            std_f = np.std(f_scores)
            significant_count = np.sum(p_values < 0.05)
            
            print(f"   {explainer_name}:")
            print(f"     Mean F-score: {mean_f:.4f} ± {std_f:.4f}")
            print(f"     Significant results: {significant_count}/{len(results)}")
        
        print("   ✅ Statistical analysis completed")
        
    except Exception as e:
        print(f"   ❌ Statistical analysis failed: {e}")
    
    # Test 4: Mock correlation data
    print("🔗 Test 4: Creating correlation scatter plot...")
    
    try:
        # Generate mock correlation data
        np.random.seed(42)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (explainer_name, results) in enumerate(list(test_results.items())[:3]):
            ax = axes[i]
            
            # Generate correlated data
            n_points = len(results)
            if explainer_name == 'SHAP':
                roar_scores = np.random.normal(0.3, 0.1, n_points)
                faith_scores = roar_scores * 2 + np.random.normal(0.15, 0.05, n_points)
                correlation = 0.75
            elif explainer_name == 'IntegratedGradients':
                roar_scores = np.random.normal(0.25, 0.12, n_points)
                faith_scores = roar_scores * 1.5 + np.random.normal(0.25, 0.1, n_points)
                correlation = 0.55
            else:  # Random
                roar_scores = np.random.normal(0.1, 0.08, n_points)
                faith_scores = np.random.normal(0.15, 0.1, n_points)
                correlation = 0.05
            
            # Clip to valid ranges
            roar_scores = np.clip(roar_scores, 0, 1)
            faith_scores = np.clip(faith_scores, 0, 1)
            
            # Create scatter plot
            ax.scatter(roar_scores, faith_scores, alpha=0.7, s=50)
            
            # Add trend line
            z = np.polyfit(roar_scores, faith_scores, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(0, 1, 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)
            
            ax.set_title(f'{explainer_name}\nr={correlation:.3f}')
            ax.set_xlabel('ROAR Accuracy Drop')
            ax.set_ylabel('Faithfulness Score')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Faithfulness vs ROAR Correlation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('test_figures/test_correlation_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("   ✅ Correlation scatter plot created successfully")
        
    except Exception as e:
        print(f"   ❌ Correlation scatter plot failed: {e}")
    
    # List generated files
    print(f"\n📁 Generated Files:")
    if os.path.exists("test_figures"):
        files = [f for f in os.listdir("test_figures") if f.endswith('.png')]
        for file in sorted(files):
            print(f"   • test_figures/{file}")
    
    print(f"\n✅ All visualization tests completed!")

if __name__ == "__main__":
    test_visualization_core()