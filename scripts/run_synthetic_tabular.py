#!/usr/bin/env python3
"""
Tiny synthetic tabular experiment to demonstrate cross-modal applicability (Task 4.2).

Generates a simple binary classification dataset with 10 features and a linear decision rule.
Evaluates IG, LIME (tabular), and Random explainers under the faithfulness metric (tabular modality).
Saves detailed_results.json and f_scores_summary.json compatible with existing analysis scripts.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# Ensure project root on sys.path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig, FaithfulnessResult
from src.masking import DataModality
from src.explainers import create_integrated_gradients_explainer, create_lime_explainer, create_random_explainer


def generate_synthetic_data(num_samples: int = 50, num_features: int = 10, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, size=(num_samples, num_features)).astype(np.float32)
    w_true = rng.normal(0, 1, size=(num_features,)).astype(np.float32)
    logits = X @ w_true
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.int64)
    return X, y, w_true


def make_model_fn(w: np.ndarray):
    W = torch.tensor(w, dtype=torch.float32)

    def model_fn(inputs):
        # inputs: torch.Tensor or np.ndarray; shape (batch, features) or (features,)
        if isinstance(inputs, np.ndarray):
            x = torch.from_numpy(inputs).float()
        elif isinstance(inputs, torch.Tensor):
            x = inputs.float()
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Linear logit
        z = x @ W
        # Produce 2-class logits [-z, z]
        logits = torch.stack([-z, z], dim=1)
        return logits

    return model_fn


def compute_results(
    X: np.ndarray,
    y: np.ndarray,
    w_true: np.ndarray,
    out_dir: Path,
    n_mc: int = 256,
    seed: int = 42,
) -> Dict[str, List[FaithfulnessResult]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model_fn = make_model_fn(w_true)

    # Faithfulness config (tabular)
    cfg = FaithfulnessConfig(
        n_samples=n_mc,
        baseline_strategy="random",
        masking_strategy="zero",
        confidence_level=0.95,
        batch_size=32,
        random_seed=seed,
        numerical_epsilon=1e-8,
    )
    metric = FaithfulnessMetric(config=cfg, modality=DataModality.TABULAR)

    # Explainers
    ig = create_integrated_gradients_explainer(n_steps=20, baseline_strategy="zero", random_seed=seed)
    lime = create_lime_explainer(n_samples=200, modality="tabular", random_seed=seed)
    rnd = create_random_explainer(random_seed=seed, distribution="uniform", scale=1.0)

    explainers = {
        "IntegratedGradients": ig,
        "LIME": lime,
        "Random": rnd,
    }

    # Prepare LIME training data
    training_data = X.copy()
    class_names = ["neg", "pos"]

    results: Dict[str, List[FaithfulnessResult]] = {k: [] for k in explainers.keys()}

    start = time.time()
    for i in range(X.shape[0]):
        x = X[i]
        target = int(y[i])
        x_t = torch.from_numpy(x).float()

        # Wrapper for precomputed attribution when needed
        for name, explainer in explainers.items():
            try:
                if name == "LIME":
                    attribution = explainer.explain(
                        model=model_fn,
                        input_data=x_t,
                        target_class=target,
                        training_data=training_data,
                        feature_names=[f"f_{j}" for j in range(x.shape[0])],
                        class_names=class_names,
                    )
                    res = metric.compute_faithfulness_score(
                        model=model_fn,
                        explainer=lambda m, d: attribution,
                        data=x_t,
                        target_class=target,
                    )
                else:
                    # IG and Random can be called directly by the metric via a small wrapper
                    def explainer_fn(model, data):
                        return explainer.explain(model, data, target_class=target)

                    res = metric.compute_faithfulness_score(
                        model=model_fn,
                        explainer=explainer_fn,
                        data=x_t,
                        target_class=target,
                    )
            except Exception as e:
                # Fallback zero result on failure
                res = FaithfulnessResult(
                    f_score=0.0,
                    confidence_interval=(0.0, 0.0),
                    n_samples=0,
                    baseline_performance=0.0,
                    explained_performance=0.0,
                    statistical_significance=False,
                    p_value=1.0,
                    computation_metrics={"error": str(e), "sample_index": i},
                )

            results[name].append(res)

    # Summaries
    summary: Dict[str, Dict[str, float]] = {}
    for name, lst in results.items():
        if not lst:
            continue
        f_scores = [r.f_score for r in lst]
        p_vals = [r.p_value for r in lst]
        sig = sum(1 for r in lst if r.statistical_significance)
        summary[name] = {
            "status": "success",
            "n_samples": len(f_scores),
            "mean_f_score": float(np.mean(f_scores)),
            "std_f_score": float(np.std(f_scores)),
            "median_f_score": float(np.median(f_scores)),
            "min_f_score": float(np.min(f_scores)),
            "max_f_score": float(np.max(f_scores)),
            "mean_p_value": float(np.mean(p_vals)),
            "significant_fraction": sig / len(f_scores),
            "confidence_interval_95": [
                float(np.percentile(f_scores, 2.5)),
                float(np.percentile(f_scores, 97.5)),
            ],
        }

    # Save artifacts compatible with analysis runner
    with open(out_dir / "f_scores_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    detailed: Dict[str, Any] = {}
    for name, lst in results.items():
        detailed[name] = []
        for r in lst:
            detailed[name].append({
                "f_score": r.f_score,
                "confidence_interval": r.confidence_interval,
                "n_samples": r.n_samples,
                "baseline_performance": r.baseline_performance,
                "explained_performance": r.explained_performance,
                "statistical_significance": r.statistical_significance,
                "p_value": r.p_value,
                "computation_metrics": r.computation_metrics,
            })

    with open(out_dir / "detailed_results.json", "w") as f:
        json.dump(detailed, f, indent=2)

    print(f"Synthetic tabular results saved to: {out_dir}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run synthetic tabular faithfulness experiment")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--faithfulness-samples", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/synthetic_tabular_ig_lime_random")
    args = parser.parse_args()

    X, y, w = generate_synthetic_data(args.num_samples, args.num_features, args.random_seed)
    out = Path(args.output_dir)
    compute_results(X, y, w, out, n_mc=args.faithfulness_samples, seed=args.random_seed)


if __name__ == "__main__":
    main()


