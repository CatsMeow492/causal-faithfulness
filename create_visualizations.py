#!/usr/bin/env python3
"""
Create visualizations from experimental results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_results_visualization():
    """Create bar chart of F-scores by explainer."""
    
    # Load results
    results_file = Path("results/experiment_results.json")
    if not results_file.exists():
        print("No results file found. Run experiments first.")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract F-scores
    explainers = []
    f_scores = []
    confidence_intervals = []
    
    for explainer_name, result in data["results"].items():
        if "error" not in result:
            explainers.append(explainer_name)
            f_scores.append(result["f_score"])
            ci = result["confidence_interval"]
            confidence_intervals.append([result["f_score"] - ci[0], ci[1] - result["f_score"]])
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    bars = plt.bar(explainers, f_scores, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add error bars
    if confidence_intervals:
        ci_array = np.array(confidence_intervals).T
        plt.errorbar(explainers, f_scores, yerr=ci_array, fmt='none', color='black', capsize=5)
    
    # Customize plot
    plt.title('Causal-Faithfulness Scores by Explainer Method', fontsize=14, fontweight='bold')
    plt.xlabel('Explanation Method', fontsize=12)
    plt.ylabel('Faithfulness Score (F)', fontsize=12)
    plt.ylim(0, 1)
    
    # Add horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
    
    # Add value labels on bars
    for bar, score in zip(bars, f_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    plt.savefig(figures_dir / "faithfulness_scores_by_explainer.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "faithfulness_scores_by_explainer.pdf", bbox_inches='tight')
    
    print(f"✓ Visualization saved to: {figures_dir}/faithfulness_scores_by_explainer.png")
    
    # Show plot
    plt.show()
    
    # Create summary statistics table
    create_summary_table(data)


def create_summary_table(data):
    """Create a summary table of results."""
    
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    print(f"Experiment: {data['experiment_type']}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Model: {data['model_info']['architecture']} ({data['model_info']['input_dim']} features)")
    print(f"Test samples: {data['model_info']['test_samples']}")
    print(f"Faithfulness samples: {data['faithfulness_config']['n_samples']}")
    
    print(f"\n{'Explainer':<20} {'F-score':<10} {'95% CI':<20} {'P-value':<10} {'Significant'}")
    print("-" * 80)
    
    for explainer_name, result in data["results"].items():
        if "error" not in result:
            f_score = result["f_score"]
            ci = result["confidence_interval"]
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            p_val = result["p_value"]
            sig = "Yes" if result["statistical_significance"] else "No"
            
            print(f"{explainer_name:<20} {f_score:<10.4f} {ci_str:<20} {p_val:<10.4f} {sig}")
        else:
            print(f"{explainer_name:<20} {'FAILED':<10} {'-':<20} {'-':<10} {'-'}")
    
    print("-" * 80)
    
    # Key findings
    print(f"\nKey Findings:")
    print(f"• Best explainer: {data['summary']['best_explainer']} (F-score: {data['summary']['best_f_score']:.4f})")
    print(f"• Successful evaluations: {data['summary']['successful_explainers']}/{data['summary']['total_explainers']}")
    
    # Validation notes
    valid_results = {name: result["f_score"] for name, result in data["results"].items() if "error" not in result}
    
    if "Random" in valid_results:
        random_score = valid_results["Random"]
        informed_explainers = [name for name in valid_results.keys() if name != "Random"]
        better_than_random = sum(1 for name in informed_explainers if valid_results[name] > random_score)
        
        print(f"• Random baseline: {random_score:.4f}")
        print(f"• Explainers outperforming random: {better_than_random}/{len(informed_explainers)}")
    
    print(f"• All F-scores in valid range [0,1]: {'Yes' if all(0 <= s <= 1 for s in valid_results.values()) else 'No'}")


if __name__ == "__main__":
    try:
        create_results_visualization()
        print("\n🎉 Visualizations created successfully!")
        
    except Exception as e:
        print(f"\n💥 Visualization failed: {e}")
        import traceback
        traceback.print_exc()