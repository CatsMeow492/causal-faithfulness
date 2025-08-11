#!/usr/bin/env python3
"""
ROAR Validation Pipeline for Causal-Faithfulness Metric (Task 3)

Runs ROAR (RemOve And Retrain) style feature removal benchmarking for
SST-2 (BERT) or WikiText-2 (GPT-2) using available explainers
(IntegratedGradients, Random, optional SHAP), computes correlations
with causal-faithfulness F-scores, and saves a concise report.
"""

import os
import sys
import json
import time
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from scipy import stats

# Ensure project root on path for `src` package
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Core imports
import src.datasets as datasets
import src.models as models
import src.config as config
import src.explainers as explainers
import src.roar as roar
import src.evaluation as evaluation

from src.config import get_device
from src.datasets import DatasetSample, DatasetManager
from src.models import ModelManager, BaseModelWrapper
from src.explainers import (
    ExplainerWrapper,
    IntegratedGradientsWrapper,
    RandomExplainer,
    SHAPWrapper,
    OcclusionExplainer,
)
from src.faithfulness import FaithfulnessConfig
from src.roar import ROARConfig, run_roar_evaluation, compute_roar_faithfulness_correlations, ROARValidator
from src.evaluation import (
    ExperimentConfig,
    EvaluationPipeline,
)


def _dataclass_or_obj_to_dict(obj: Any) -> Any:
    """Best-effort serialization helper for dataclasses and nested objects."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_dataclass_or_obj_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_or_obj_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def _init_explainers(
    random_seed: int = 42,
    include_shap: bool = False,
    selected: Optional[List[str]] = None,
    occlusion_max_tokens: int = 64,
) -> Dict[str, ExplainerWrapper]:
    # Build available explainers
    available: Dict[str, ExplainerWrapper] = {}

    # Random baseline (fast)
    available["Random"] = RandomExplainer(random_seed=random_seed, distribution="uniform", scale=1.0)

    # Optional Integrated Gradients (may be slow/ill-defined on discrete tokens)
    try:
        available["IntegratedGradients"] = IntegratedGradientsWrapper(
            n_steps=10, baseline_strategy="zero", random_seed=random_seed, internal_batch_size=8
        )
    except Exception as e:
        warnings.warn(f"IG not available, skipping: {e}")

    # Occlusion (token masking) — fast, informed baseline for text
    try:
        available["Occlusion"] = OcclusionExplainer(max_tokens=occlusion_max_tokens, random_seed=random_seed)
    except Exception as e:
        warnings.warn(f"Occlusion not available, skipping: {e}")

    # Optional SHAP
    if include_shap:
        try:
            available["SHAP"] = SHAPWrapper(explainer_type="kernel", n_samples=200, random_seed=random_seed)
        except Exception as e:
            warnings.warn(f"SHAP not available, skipping: {e}")

    # Filter by selection
    if selected:
        selected_set = set(selected)
        return {k: v for k, v in available.items() if k in selected_set}
    return available


def _load_components(dataset_name: str, num_samples: int, device: torch.device) -> tuple[list[DatasetSample], BaseModelWrapper]:
    dm = DatasetManager(device=device)
    mm = ModelManager(device=device)

    if dataset_name.lower() == "sst2":
        samples = dm.load_sst2(split="validation", num_samples=num_samples, model_name="bert-base-uncased")
        model = mm.load_bert_sst2(model_name="textattack/bert-base-uncased-SST-2")
    elif dataset_name.lower() == "wikitext2":
        samples = dm.load_wikitext2(split="validation", num_samples=num_samples, model_name="gpt2")
        model = mm.load_gpt2_small(model_name="gpt2")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return samples, model


def _run_faithfulness(
    dataset_name: str,
    model_name: str,
    samples: List[DatasetSample],
    explainer_instances: Dict[str, ExplainerWrapper],
    random_seed: int,
    batch_size: int,
    faithfulness_samples: int,
) -> evaluation.ExperimentResult:
    exp_name = f"{dataset_name}_roar_validation_faithfulness"
    cfg = ExperimentConfig(
        experiment_name=exp_name,
        dataset_name=dataset_name,
        model_name=model_name,
        explainer_names=list(explainer_instances.keys()),
        num_samples=len(samples),
        batch_size=batch_size,
        random_seed=random_seed,
        faithfulness_config=FaithfulnessConfig(
            n_samples=faithfulness_samples,
            baseline_strategy="random",
            masking_strategy="pad",
            batch_size=batch_size,
            random_seed=random_seed,
        ),
    )

    pipeline = EvaluationPipeline(cfg)
    # Supply already-initialized explainers
    result = pipeline.run_evaluation(explainer_instances)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run ROAR validation and correlation analysis")
    parser.add_argument("--dataset", type=str, choices=["sst2", "wikitext2"], required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--removal-percentages", type=str, default="0.1,0.2")
    parser.add_argument("--include-shap", action="store_true")
    parser.add_argument(
        "--explainers",
        type=str,
        default="Random",
        help="Comma-separated list of explainers to run (e.g., Random,IntegratedGradients,SHAP)",
    )
    parser.add_argument("--output-dir", type=str, default="results/roar_validation")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--faithfulness-samples", type=int, default=64, help="Monte Carlo samples for F(E)")
    parser.add_argument("--skip-faithfulness", action="store_true")
    parser.add_argument("--occlusion-max-tokens", type=int, default=64, help="Max tokens for Occlusion explainer")

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    removal_percentages = [float(x) for x in args.removal_percentages.split(",") if x.strip()]

    print("Running ROAR validation with:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Samples: {args.num_samples}")
    print(f"  - Removal %: {removal_percentages}")
    print(f"  - Include SHAP: {args.include_shap}")
    print(f"  - Output dir: {args.output_dir}")
    print(f"  - Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data and model
    samples, model = _load_components(args.dataset, args.num_samples, device)

    # Initialize explainers
    explainer_names = [s.strip() for s in args.explainers.split(",") if s.strip()]
    exps = _init_explainers(
        random_seed=args.random_seed,
        include_shap=args.include_shap,
        selected=explainer_names,
        occlusion_max_tokens=args.occlusion_max_tokens,
    )

    # Run faithfulness to get F-scores for correlation
    model_name = (
        "textattack/bert-base-uncased-SST-2" if args.dataset == "sst2" else "gpt2"
    )
    if args.skip_faithfulness:
        exp_result = evaluation.ExperimentResult(
            config=None,
            explainer_results={},
            summary_statistics={},
            computation_time=0.0,
        )
    else:
        print("Computing faithfulness results for correlation...")
        exp_result = _run_faithfulness(
            dataset_name=args.dataset,
            model_name=model_name,
            samples=samples,
            explainer_instances=exps,
            random_seed=args.random_seed,
            batch_size=args.batch_size,
            faithfulness_samples=args.faithfulness_samples,
        )

    # Run ROAR benchmark
    print("Running ROAR evaluation...")
    rcfg = ROARConfig(
        removal_percentages=removal_percentages,
        n_samples=len(samples),
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        device=device,
    )
    roar_results = run_roar_evaluation(model, exps, samples, rcfg)

    # Compute correlations (aggregate-level, across removal percentages)
    print("Computing ROAR-faithfulness correlations...")
    correlation_results = compute_roar_faithfulness_correlations(
        roar_results, exp_result.explainer_results if exp_result and exp_result.explainer_results else {}
    )

    # Optional: per-sample probability-drop vs faithfulness pairing (use first removal pct)
    try:
        first_pct = removal_percentages[0]
        per_sample = {}
        for name, expl in exps.items():
            benchmark = roar.ROARBenchmark(rcfg)
            drops = benchmark.evaluate_probability_drop(model, expl, samples, first_pct)
            # Align with faithfulness f-scores list order
            f_scores = [r.f_score for r in (exp_result.explainer_results.get(name, []) if exp_result and exp_result.explainer_results else [])]
            per_sample[name] = {"drops": drops[: len(f_scores)], "f_scores": f_scores}
    except Exception as e:
        per_sample = {"error": str(e)}

    # Generate correlation matrix and significance tables
    # Prepare CSV-friendly rows
    corr_rows = [[
        "explainer", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "n"
    ]]
    for r in correlation_results:
        corr_rows.append([
            r.explainer_name,
            float(r.pearson_r), float(r.p_value),
            float(r.spearman_rho), float(r.spearman_p), int(r.n_samples)
        ])

    # Bonferroni correction across all tests (pearson+spearman per explainer)
    m_tests = max(1, len(correlation_results) * 2)
    alpha = 0.05
    bonf_alpha = alpha / m_tests
    signif_rows = [[
        "explainer", "pearson_p", "pearson_significant", "spearman_p", "spearman_significant", "bonferroni_alpha", "tests_m"
    ]]
    for r in correlation_results:
        signif_rows.append([
            r.explainer_name,
            float(r.p_value), bool(r.p_value < bonf_alpha),
            float(r.spearman_p), bool(r.spearman_p < bonf_alpha),
            bonf_alpha, m_tests
        ])

    # Generate validation report
    validator = ROARValidator(random_seed=args.random_seed)
    report_md = validator.generate_validation_report(
        roar_results=roar_results,
        faithfulness_results=exp_result.explainer_results,
        correlation_results=correlation_results,
    )

    # Save all artifacts
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = Path(args.output_dir) / f"{args.dataset}_roar_validation_{ts}"
    base.parent.mkdir(parents=True, exist_ok=True)

    # Serialize results
    # Convert numpy types robustly to JSON
    def _safe(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Sanitize complex objects for JSON
    def _sanitize(o):
        if is_dataclass(o):
            return _sanitize(asdict(o))
        if isinstance(o, dict):
            return {str(k): _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, (torch.device, torch.dtype)):
            return str(o)
        if isinstance(o, (bool, int, float, str)) or o is None:
            return o
        return str(o)

    # Compute per-sample correlations if available
    per_sample_corr = {}
    if isinstance(payload := {
        "per_sample": None
    }, dict):
        pass  # placeholder to keep scope clean

    if isinstance(locals().get('per_sample'), dict):
        for name, vals in per_sample.items():
            d = vals.get("drops", []) if isinstance(vals, dict) else []
            f = vals.get("f_scores", []) if isinstance(vals, dict) else []
            n = min(len(d), len(f))
            if n >= 2:
                r, p = stats.pearsonr(d[:n], f[:n])
                rho, sp = stats.spearmanr(d[:n], f[:n])
                per_sample_corr[name] = {
                    "pearson_r": float(r),
                    "pearson_p": float(p),
                    "spearman_rho": float(rho),
                    "spearman_p": float(sp),
                    "n": n,
                }
            else:
                per_sample_corr[name] = {"n": n}

    payload = {
        "config": {
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "removal_percentages": removal_percentages,
            "include_shap": args.include_shap,
            "device": str(device),
            "random_seed": args.random_seed,
        },
        "roar_results": _dataclass_or_obj_to_dict(roar_results),
        "faithfulness_summary": {k: {kk: _safe(vv) for kk, vv in v.items()} for k, v in exp_result.summary_statistics.items()},
        "correlation_results": _dataclass_or_obj_to_dict(correlation_results),
        "per_sample": per_sample,
        "per_sample_correlation": per_sample_corr,
    }

    with open(str(base) + ".json", "w") as f:
        json.dump(_sanitize(payload), f, indent=2)

    with open(str(base) + ".md", "w") as f:
        f.write(report_md)

    # Write CSVs
    def _write_csv(path: str, rows: list[list[Any]]):
        import csv
        with open(path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerows(rows)

    _write_csv(str(base) + "_correlations.csv", corr_rows)
    _write_csv(str(base) + "_significance_bonferroni.csv", signif_rows)

    print(f"Saved ROAR validation artifacts to: {base}.json and {base}.md")


if __name__ == "__main__":
    main()


