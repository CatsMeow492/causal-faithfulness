#!/usr/bin/env python3
"""
Make theoretical validation figures (Task 5.2):
 - Monotonicity: F-score vs top_fraction of kept features
 - Normalization: F-score histograms bounded in [0, 1]

Uses a tiny synthetic tabular setup and light explainers (IG, Random) for speed.
Outputs: figures/theory_monotonicity.(png|pdf), figures/theory_normalization.(png|pdf)
"""

import os
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure project root on path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig, FaithfulnessResult
from src.masking import DataModality
from src.explainers import create_integrated_gradients_explainer, create_random_explainer


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
        if isinstance(inputs, np.ndarray):
            x = torch.from_numpy(inputs).float()
        elif isinstance(inputs, torch.Tensor):
            x = inputs.float()
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = x @ W
        logits = torch.stack([-z, z], dim=1)
        return logits

    return model_fn


def compute_f_scores(
    X: np.ndarray,
    y: np.ndarray,
    model_fn,
    explainer_name: str,
    top_fraction: float,
    seed: int = 42,
    n_mc: int = 128,
) -> List[float]:
    cfg = FaithfulnessConfig(
        n_samples=n_mc,
        baseline_strategy="random",
        masking_strategy="zero",
        confidence_level=0.95,
        batch_size=32,
        random_seed=seed,
        numerical_epsilon=1e-8,
        top_fraction=top_fraction,
        device=torch.device('cpu'),
    )
    metric = FaithfulnessMetric(config=cfg, modality=DataModality.TABULAR)

    if explainer_name == 'IG':
        explainer = create_integrated_gradients_explainer(n_steps=20, baseline_strategy="zero", random_seed=seed)
    elif explainer_name == 'Random':
        explainer = create_random_explainer(random_seed=seed, distribution="uniform", scale=1.0)
    else:
        raise ValueError("Unsupported explainer for theoretical figures")

    f_scores: List[float] = []
    for i in range(X.shape[0]):
        x_t = torch.from_numpy(X[i]).float()
        target = int(y[i])
        def ex_fn(model, data):
            return explainer.explain(model, data, target_class=target)
        res: FaithfulnessResult = metric.compute_faithfulness_score(
            model=model_fn,
            explainer=ex_fn,
            data=x_t,
            target_class=target,
        )
        f_scores.append(res.f_score)
    return f_scores


def figure_monotonicity(X, y, w, outdir: Path):
    model_fn = make_model_fn(w)
    fractions = [0.05, 0.1, 0.2, 0.4, 0.8]
    explainers = ['IG', 'Random']
    curves: Dict[str, List[float]] = {e: [] for e in explainers}

    for frac in fractions:
        for e in explainers:
            fs = compute_f_scores(X, y, model_fn, e, top_fraction=frac)
            curves[e].append(float(np.mean(fs)))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for e in explainers:
        ax.plot(fractions, curves[e], marker='o', label=e)
    ax.set_xlabel('Top fraction of features kept')
    ax.set_ylabel('Mean F-score')
    ax.set_title('Monotonicity: F vs Kept Feature Fraction (Synthetic Tabular)')
    ax.set_ylim(0, 1.0)
    ax.legend()
    out_png = outdir / 'theory_monotonicity.png'
    out_pdf = outdir / 'theory_monotonicity.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('WROTE', out_png, out_pdf)


def figure_normalization(X, y, w, outdir: Path):
    model_fn = make_model_fn(w)
    explainers = ['IG', 'Random']
    all_scores: Dict[str, List[float]] = {}
    for e in explainers:
        fs = compute_f_scores(X, y, model_fn, e, top_fraction=0.2)
        all_scores[e] = fs

    # Plot histograms
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0.0, 1.0, 21)
    colors = {'IG': '#4C72B0', 'Random': '#DD8452'}
    for e in explainers:
        ax.hist(all_scores[e], bins=bins, alpha=0.6, label=e, color=colors.get(e), edgecolor='black', linewidth=0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('F-score')
    ax.set_ylabel('Count')
    ax.set_title('Normalization: F-score Distribution in [0, 1] (Synthetic Tabular)')
    ax.legend()
    out_png = outdir / 'theory_normalization.png'
    out_pdf = outdir / 'theory_normalization.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('WROTE', out_png, out_pdf)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate theoretical validation figures')
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--num-features', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=str, default='figures')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, w = generate_synthetic_data(args.num_samples, args.num_features, args.seed)
    figure_monotonicity(X, y, w, outdir)
    figure_normalization(X, y, w, outdir)


if __name__ == '__main__':
    main()


