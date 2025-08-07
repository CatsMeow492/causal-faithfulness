#!/usr/bin/env python3
"""
Reproducible Experiment Script for Causal-Faithfulness Metric

This script reproduces the main experiments from the paper with fixed seeds
and documented hyperparameters for full reproducibility.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from reproducibility import (
    ReproducibilityConfig, 
    ReproducibilityManager,
    ensure_reproducibility
)
from faithfulness import FaithfulnessMetric, FaithfulnessConfig
from explainers import SHAPWrapper, IntegratedGradientsWrapper, LIMEWrapper, RandomExplainer
from masking import DataModality
from datasets import load_sst2_dataset, load_wikitext2_dataset
from models import load_bert_sst2_model, load_gpt2_model
from evaluation import evaluate_explainer


# Experiment configurations
EXPERIMENT_CONFIGS = {
    "sst2_bert": {
        "description": "SST-2 sentiment classification with BERT",
        "dataset": "sst2",
        "model": "bert-base-uncased-sst2",
        "modality": DataModality.TEXT,
        "n_samples": 200,  # Validation instances
        "faithfulness_config": {
            "n_samples": 2000,
            "baseline_strategy": "random",
            "masking_strategy": "pad",
            "confidence_level": 0.95,
            "batch_size": 16,
            "random_seed": 42
        },
        "explainers": {
            "SHAP_Kernel": {
                "class": SHAPWrapper,
                "params": {
                    "explainer_type": "kernel",
                    "n_samples": 1000,
                    "random_seed": 42
                }
            },
            "IntegratedGradients": {
                "class": IntegratedGradientsWrapper,
                "params": {
                    "n_steps": 50,
                    "baseline_strategy": "zero",
                    "random_seed": 42
                }
            },
            "LIME": {
                "class": LIMEWrapper,
                "params": {
                    "n_samples": 500,
                    "modality": "text",
                    "random_seed": 42
                }
            },
            "Random": {
                "class": RandomExplainer,
                "params": {
                    "random_seed": 42,
                    "distribution": "uniform"
                }
            }
        }
    },
    
    "wikitext2_gpt2": {
        "description": "WikiText-2 language modeling with GPT-2",
        "dataset": "wikitext2",
        "model": "gpt2-small",
        "modality": DataModality.TEXT,
        "n_samples": 200,
        "faithfulness_config": {
            "n_samples": 2000,
            "baseline_strategy": "random",
            "masking_strategy": "pad",
            "confidence_level": 0.95,
            "batch_size": 8,  # Smaller for GPT-2
            "random_seed": 42
        },
        "explainers": {
            "SHAP_Kernel": {
                "class": SHAPWrapper,
                "params": {
                    "explainer_type": "kernel",
                    "n_samples": 1000,
                    "random_seed": 42
                }
            },
            "IntegratedGradients": {
                "class": IntegratedGradientsWrapper,
                "params": {
                    "n_steps": 50,
                    "baseline_strategy": "zero",
                    "random_seed": 42
                }
            },
            "Random": {
                "class": RandomExplainer,
                "params": {
                    "random_seed": 42,
                    "distribution": "uniform"
                }
            }
        }
    }
}

# Fixed random seeds for reproducibility
MASTER_SEED = 42
DATASET_SEEDS = {
    "sst2": 42,
    "wikitext2": 43
}
EVALUATION_SEEDS = {
    "sst2_bert": 42,
    "wikitext2_gpt2": 43
}


def setup_reproducibility(experiment_name: str) -> ReproducibilityManager:
    """Set up reproducibility for an experiment."""
    seed = EVALUATION_SEEDS.get(experiment_name, MASTER_SEED)
    
    config = ReproducibilityConfig(
        global_seed=seed,
        numpy_seed=seed,
        torch_seed=seed,
        python_seed=seed,
        torch_deterministic=True,
        torch_benchmark=False,
        cuda_deterministic=True,
        mps_deterministic=True,
        track_versions=True,
        track_environment=True
    )
    
    manager = ReproducibilityManager(config)
    metadata = manager.initialize_reproducibility(experiment_id=f"reproduce_{experiment_name}")
    
    print(f"Initialized reproducibility for {experiment_name}")
    print(f"Experiment ID: {metadata.experiment_id}")
    print(f"Seeds: {metadata.seeds}")
    
    return manager


def load_dataset_and_model(experiment_config: Dict[str, Any]):
    """Load dataset and model for an experiment."""
    dataset_name = experiment_config["dataset"]
    model_name = experiment_config["model"]
    
    print(f"Loading dataset: {dataset_name}")
    print(f"Loading model: {model_name}")
    
    # Load dataset with fixed seed
    dataset_seed = DATASET_SEEDS.get(dataset_name, MASTER_SEED)
    
    if dataset_name == "sst2":
        dataset = load_sst2_dataset(
            split="validation",
            max_samples=experiment_config["n_samples"],
            random_seed=dataset_seed
        )
        model, tokenizer = load_bert_sst2_model()
        
    elif dataset_name == "wikitext2":
        dataset = load_wikitext2_dataset(
            split="validation", 
            max_samples=experiment_config["n_samples"],
            random_seed=dataset_seed
        )
        model, tokenizer = load_gpt2_model()
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loaded {len(dataset)} samples")
    
    return dataset, model, tokenizer


def initialize_explainers(explainer_configs: Dict[str, Dict]) -> Dict[str, Any]:
    """Initialize explainers from configuration."""
    explainers = {}
    
    for name, config in explainer_configs.items():
        explainer_class = config["class"]
        params = config["params"]
        
        print(f"Initializing {name} with params: {params}")
        explainers[name] = explainer_class(**params)
    
    return explainers


def run_experiment(experiment_name: str, output_dir: str, verbose: bool = True) -> Dict[str, Any]:
    """Run a single experiment with full reproducibility."""
    
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Get experiment configuration
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    experiment_config = EXPERIMENT_CONFIGS[experiment_name]
    print(f"Description: {experiment_config['description']}")
    
    # Set up reproducibility
    repro_manager = setup_reproducibility(experiment_name)
    
    # Create output directory
    exp_output_dir = Path(output_dir) / experiment_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save reproducibility metadata
    repro_metadata_path = exp_output_dir / "reproducibility_metadata.json"
    repro_manager.save_metadata(str(repro_metadata_path))
    
    # Save experiment configuration
    config_path = exp_output_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2, default=str)
    
    try:
        # Load dataset and model
        dataset, model, tokenizer = load_dataset_and_model(experiment_config)
        
        # Initialize faithfulness metric
        faithfulness_config = FaithfulnessConfig(**experiment_config["faithfulness_config"])
        faithfulness_metric = FaithfulnessMetric(
            faithfulness_config, 
            modality=experiment_config["modality"]
        )
        
        # Initialize explainers
        explainers = initialize_explainers(experiment_config["explainers"])
        
        # Run evaluation
        print(f"\nRunning evaluation on {len(dataset)} samples...")
        results = {}
        
        for explainer_name, explainer in explainers.items():
            print(f"\nEvaluating {explainer_name}...")
            
            # Run evaluation with proper error handling
            try:
                explainer_results = evaluate_explainer(
                    model=model,
                    tokenizer=tokenizer,
                    explainer=explainer,
                    dataset=dataset,
                    faithfulness_metric=faithfulness_metric,
                    verbose=verbose
                )
                
                results[explainer_name] = explainer_results
                
                # Print summary
                f_scores = [r.f_score for r in explainer_results]
                mean_f_score = np.mean(f_scores)
                std_f_score = np.std(f_scores)
                
                print(f"  Mean F-score: {mean_f_score:.4f} ± {std_f_score:.4f}")
                print(f"  Samples evaluated: {len(explainer_results)}")
                
            except Exception as e:
                print(f"  Error evaluating {explainer_name}: {e}")
                results[explainer_name] = {"error": str(e)}
        
        # Save detailed results
        results_path = exp_output_dir / "detailed_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary statistics
        summary = generate_experiment_summary(results, experiment_config)
        
        # Save summary
        summary_path = exp_output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print_experiment_summary(summary)
        
        # Save reproducibility report
        report_path = exp_output_dir / "reproducibility_report.txt"
        with open(report_path, 'w') as f:
            f.write(repro_manager.get_reproducibility_report())
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {exp_output_dir}")
        
        return summary
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        
        # Save error information
        error_path = exp_output_dir / "error.json"
        with open(error_path, 'w') as f:
            json.dump({
                "error": str(e),
                "experiment_name": experiment_name,
                "timestamp": repro_manager._metadata.timestamp if repro_manager._metadata else None
            }, f, indent=2)
        
        raise e


def generate_experiment_summary(results: Dict[str, Any], experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics for an experiment."""
    
    summary = {
        "experiment_config": experiment_config,
        "explainer_summaries": {},
        "comparison": {}
    }
    
    valid_results = {}
    
    # Process each explainer's results
    for explainer_name, explainer_results in results.items():
        if isinstance(explainer_results, dict) and "error" in explainer_results:
            summary["explainer_summaries"][explainer_name] = {
                "status": "failed",
                "error": explainer_results["error"]
            }
            continue
        
        # Extract F-scores
        f_scores = [r.f_score for r in explainer_results]
        p_values = [r.p_value for r in explainer_results]
        significant_count = sum(1 for r in explainer_results if r.statistical_significance)
        
        explainer_summary = {
            "status": "success",
            "n_samples": len(f_scores),
            "mean_f_score": float(np.mean(f_scores)),
            "std_f_score": float(np.std(f_scores)),
            "median_f_score": float(np.median(f_scores)),
            "min_f_score": float(np.min(f_scores)),
            "max_f_score": float(np.max(f_scores)),
            "mean_p_value": float(np.mean(p_values)),
            "significant_fraction": significant_count / len(f_scores),
            "confidence_interval_95": [
                float(np.percentile(f_scores, 2.5)),
                float(np.percentile(f_scores, 97.5))
            ]
        }
        
        summary["explainer_summaries"][explainer_name] = explainer_summary
        valid_results[explainer_name] = f_scores
    
    # Generate comparison statistics
    if len(valid_results) > 1:
        explainer_names = list(valid_results.keys())
        
        # Find best explainer
        mean_scores = {name: np.mean(scores) for name, scores in valid_results.items()}
        best_explainer = max(mean_scores.keys(), key=lambda x: mean_scores[x])
        
        summary["comparison"] = {
            "best_explainer": best_explainer,
            "best_mean_f_score": mean_scores[best_explainer],
            "explainer_ranking": sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
        }
        
        # Check random baseline performance
        if "Random" in valid_results:
            random_mean = np.mean(valid_results["Random"])
            summary["comparison"]["random_baseline_mean"] = random_mean
            summary["comparison"]["sanity_check_passed"] = random_mean < 0.3
    
    return summary


def print_experiment_summary(summary: Dict[str, Any]):
    """Print a formatted experiment summary."""
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    # Print explainer results
    print(f"\nExplainer Results:")
    print(f"{'Explainer':<20} {'Status':<10} {'Mean F-score':<12} {'95% CI':<20} {'Significant %'}")
    print("-" * 80)
    
    for explainer_name, summary_data in summary["explainer_summaries"].items():
        if summary_data["status"] == "failed":
            print(f"{explainer_name:<20} {'FAILED':<10} {'-':<12} {'-':<20} {'-'}")
        else:
            mean_f = summary_data["mean_f_score"]
            ci = summary_data["confidence_interval_95"]
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            sig_pct = summary_data["significant_fraction"] * 100
            
            print(f"{explainer_name:<20} {'SUCCESS':<10} {mean_f:<12.4f} {ci_str:<20} {sig_pct:.1f}%")
    
    # Print comparison
    if "comparison" in summary and summary["comparison"]:
        print(f"\nComparison Results:")
        print(f"Best explainer: {summary['comparison']['best_explainer']} "
              f"(F-score: {summary['comparison']['best_mean_f_score']:.4f})")
        
        if "sanity_check_passed" in summary["comparison"]:
            sanity_status = "PASSED" if summary["comparison"]["sanity_check_passed"] else "FAILED"
            random_score = summary["comparison"]["random_baseline_mean"]
            print(f"Sanity check: {sanity_status} (Random baseline: {random_score:.4f})")


def main():
    """Main function for running reproducible experiments."""
    
    parser = argparse.ArgumentParser(description="Run reproducible causal-faithfulness experiments")
    parser.add_argument(
        "--experiment", 
        type=str, 
        choices=list(EXPERIMENT_CONFIGS.keys()) + ["all"],
        default="all",
        help="Experiment to run (default: all)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results/reproducible_experiments",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--verify-reproducibility",
        type=str,
        help="Path to reference metadata file for verification"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify reproducibility if requested
    if args.verify_reproducibility:
        print("Verifying reproducibility...")
        from reproducibility import load_and_verify_reproducibility
        
        verification_results = load_and_verify_reproducibility(args.verify_reproducibility)
        
        print("Verification Results:")
        for check, result in verification_results.items():
            status = "PASS" if result else "FAIL" if result is not None else "SKIP"
            print(f"  {check}: {status}")
        
        if not all(r for r in verification_results.values() if r is not None):
            print("WARNING: Reproducibility verification failed!")
        else:
            print("Reproducibility verification passed!")
    
    # Run experiments
    if args.experiment == "all":
        experiments_to_run = list(EXPERIMENT_CONFIGS.keys())
    else:
        experiments_to_run = [args.experiment]
    
    print(f"Running {len(experiments_to_run)} experiment(s): {experiments_to_run}")
    
    all_summaries = {}
    
    for experiment_name in experiments_to_run:
        try:
            summary = run_experiment(experiment_name, str(output_dir), args.verbose)
            all_summaries[experiment_name] = summary
            
        except Exception as e:
            print(f"Failed to run experiment {experiment_name}: {e}")
            all_summaries[experiment_name] = {"error": str(e)}
    
    # Save combined results
    combined_results_path = output_dir / "combined_results.json"
    with open(combined_results_path, 'w') as f:
        json.dump(all_summaries, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Combined results: {combined_results_path}")


if __name__ == "__main__":
    main()