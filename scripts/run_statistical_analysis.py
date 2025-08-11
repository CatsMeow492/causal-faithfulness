#!/usr/bin/env python3
"""
Task 4 runner: Comprehensive statistical analysis for saved experiment results.

Loads detailed faithfulness results from result directories and generates:
- Pairwise statistical tests (paired t-test, Wilcoxon) within each dataset
- Bootstrap confidence intervals per explainer
- Sanity checks
- Cross-dataset comparisons for common explainers (independent t-test, Cohen's d)
- Simple power analysis approximations
- Markdown reports and JSON artifacts
"""

import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.faithfulness import FaithfulnessResult
from src.statistical_analysis import (
    ValidationSuite as StatValidationSuite,
    StatisticalAnalyzer as StatTests,
    StatisticalTestResult,
)


def load_detailed_results(run_dir: Path) -> Dict[str, List[FaithfulnessResult]]:
    """Load detailed_results.json and convert to FaithfulnessResult objects per explainer."""
    detailed_path = run_dir / "detailed_results.json"
    if not detailed_path.exists():
        raise FileNotFoundError(f"Missing detailed_results.json in {run_dir}")

    with open(detailed_path, "r") as f:
        blob = json.load(f)

    results: Dict[str, List[FaithfulnessResult]] = {}
    for explainer_name, items in blob.items():
        converted: List[FaithfulnessResult] = []
        for it in items:
            # Defensive conversions
            f_score = float(it.get("f_score", 0.0))
            ci = it.get("confidence_interval", [0.0, 0.0])
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                confidence_interval = (float(ci[0]), float(ci[1]))
            else:
                confidence_interval = (0.0, 0.0)
            n_samples = int(it.get("n_samples", 0))
            baseline_perf = float(it.get("baseline_performance", 0.0))
            explained_perf = float(it.get("explained_performance", 0.0))
            stat_sig = bool(it.get("statistical_significance", False))
            p_val = float(it.get("p_value", 1.0))
            comp_metrics = it.get("computation_metrics", {}) or {}
            converted.append(
                FaithfulnessResult(
                    f_score=f_score,
                    confidence_interval=confidence_interval,
                    n_samples=n_samples,
                    baseline_performance=baseline_perf,
                    explained_performance=explained_perf,
                    statistical_significance=stat_sig,
                    p_value=p_val,
                    computation_metrics={k: (float(v) if isinstance(v, (int, float)) else v) for k, v in comp_metrics.items()},
                )
            )
        results[explainer_name] = converted
    return results


def simple_power_independent_t(effect_size_d: float, n1: int, n2: int, alpha: float = 0.05) -> float:
    """Approximate post-hoc power for two-sample t-test with equal variance assumption.
    Using normal approximation: ncp = d * sqrt(n_eff), where n_eff = (n1*n2)/(n1+n2).
    """
    if effect_size_d <= 0 or n1 < 2 or n2 < 2:
        return 0.0
    from math import sqrt
    n_eff = (n1 * n2) / (n1 + n2)
    ncp = effect_size_d * sqrt(n_eff)
    # Two-sided test critical z
    from scipy.stats import norm
    z_crit = norm.ppf(1 - alpha / 2)
    # Power approximation: 1 - beta = P(|Z| > z_crit | NCP)
    # Approx symmetrical tail with shift ncp
    power = norm.cdf(-z_crit + ncp) + (1 - norm.cdf(z_crit + ncp))
    return float(max(0.0, min(1.0, power)))


def analyze_run(run_name: str, run_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Analyze a single run directory and save artifacts."""
    results = load_detailed_results(run_dir)

    suite = StatValidationSuite(alpha=0.05, random_seed=42)
    # Identify Random explainer results if present
    random_key = None
    for k in results.keys():
        if k.lower().startswith("random"):
            random_key = k
            break
    random_results = results.get(random_key, None)
    report = suite.run_full_validation(results, roar_results=None, random_explainer_results=random_results)

    # Save validation report and JSON
    run_out = output_dir / run_name
    run_out.mkdir(parents=True, exist_ok=True)
    with open(run_out / "validation_report.md", "w") as f:
        f.write(suite.generate_validation_report_text(report))
    with open(run_out / "validation_report.json", "w") as f:
        json.dump(serialize(report), f, indent=2)

    # Pairwise comparisons and bootstrap via analysis.StatisticalAnalyzer
    from src.analysis import StatisticalAnalyzer as AnalysisStat
    analyzer = AnalysisStat()
    comparisons = analyzer.compare_explainers(results)
    with open(run_out / "pairwise_comparisons.json", "w") as f:
        json.dump(serialize(comparisons), f, indent=2)

    # Simple power analysis for each pairwise comparison (independent t-test context)
    power: Dict[str, float] = {}
    for key, comp in comparisons.items():
        d = float(comp.get("cohens_d", 0.0))
        n1 = int(np.floor(comp.get("std1") is not None and len(results[key.split("_vs_")[0]]) or 0))
        # n1/n2: infer from results per explainer
        e1, e2 = comp.get("explainer1"), comp.get("explainer2")
        n1 = len(results.get(e1, []))
        n2 = len(results.get(e2, []))
        power[key] = simple_power_independent_t(abs(d), n1, n2, alpha=0.05)
    with open(run_out / "power_analysis.json", "w") as f:
        json.dump(power, f, indent=2)

    return {
        "validation": report,
        "comparisons": comparisons,
        "power": power,
    }


def cross_dataset_compare(
    results_a: Dict[str, List[FaithfulnessResult]],
    results_b: Dict[str, List[FaithfulnessResult]],
    name_a: str,
    name_b: str,
) -> Dict[str, Any]:
    """Compare the same explainers across two datasets using independent t-tests and effect sizes."""
    from scipy import stats
    comparisons: Dict[str, Any] = {}
    common = sorted(set(results_a.keys()) & set(results_b.keys()))
    for expl in common:
        s1 = np.array([r.f_score for r in results_a[expl]])
        s2 = np.array([r.f_score for r in results_b[expl]])
        if len(s1) < 2 or len(s2) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(s1, s2, equal_var=False)
        pooled_std = math.sqrt(((len(s1) - 1) * s1.var(ddof=1) + (len(s2) - 1) * s2.var(ddof=1)) / (len(s1) + len(s2) - 2)) if (len(s1) + len(s2) - 2) > 0 else 0.0
        d = (s1.mean() - s2.mean()) / pooled_std if pooled_std > 0 else 0.0
        comparisons[expl] = {
            "dataset_a": name_a,
            "dataset_b": name_b,
            "mean_a": float(s1.mean()),
            "mean_b": float(s2.mean()),
            "std_a": float(s1.std(ddof=1) if len(s1) > 1 else 0.0),
            "std_b": float(s2.std(ddof=1) if len(s2) > 1 else 0.0),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "power": simple_power_independent_t(abs(d), len(s1), len(s2), alpha=0.05),
        }
    return comparisons


def serialize(obj: Any) -> Any:
    """JSON-safe serializer for nested objects with dataclasses."""
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(x) for x in obj]
    # StatisticalTestResult or other objects
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        d = {}
        for k, v in obj.__dict__.items():
            d[k] = serialize(v)
        return d
    if isinstance(obj, tuple):
        return [serialize(x) for x in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Task 4 statistical analysis on saved results")
    parser.add_argument("--runs", type=str, nargs="+", required=True, help="Paths to run directories containing detailed_results.json")
    parser.add_argument("--output-dir", type=str, default="results/analysis/task4", help="Output directory for analysis artifacts")
    args = parser.parse_args()

    out = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    # Analyze each run individually
    per_run_results: Dict[str, Any] = {}
    loaded_results: Dict[str, Dict[str, List[FaithfulnessResult]]] = {}
    for run_path in args.runs:
        run_dir = Path(run_path)
        run_name = run_dir.name
        try:
            loaded = load_detailed_results(run_dir)
            loaded_results[run_name] = loaded
            res = analyze_run(run_name, run_dir, out)
            per_run_results[run_name] = res
        except Exception as e:
            per_run_results[run_name] = {"error": str(e)}

    # Cross-dataset comparisons for common explainers where possible
    # Heuristic: detect SST-2 and WikiText-2 runs by directory name
    sst2_keys = [k for k in loaded_results.keys() if "sst2" in k.lower()]
    wt2_keys = [k for k in loaded_results.keys() if "wikitext2" in k.lower()]
    if sst2_keys and wt2_keys:
        # Use the largest runs if multiple
        sst2_key = sorted(sst2_keys, key=lambda k: len(next(iter(loaded_results[k].values()))), reverse=True)[0]
        wt2_key = sorted(wt2_keys, key=lambda k: len(next(iter(loaded_results[k].values()))), reverse=True)[0]
        cross = cross_dataset_compare(loaded_results[sst2_key], loaded_results[wt2_key], sst2_key, wt2_key)
        with open(out / "cross_dataset_comparisons.json", "w") as f:
            json.dump(serialize(cross), f, indent=2)

    # Save index file
    with open(out / "index.json", "w") as f:
        json.dump(serialize(per_run_results), f, indent=2)

    print(f"Task 4 analysis artifacts saved to: {out}")


if __name__ == "__main__":
    main()


