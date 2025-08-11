#!/usr/bin/env python3
"""
Generate publication-quality figures (Task 5.1) from saved run directories.

Inputs: one or more run directories that contain detailed_results.json and f_scores_summary.json
Outputs: figures in both PNG and PDF under ./figures
 - Faithfulness bar chart
 - Score comparison heatmap
 - Performance table (CSV + image)

If ROAR correlation inputs are provided later, we can add correlation scatter as well.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.faithfulness import FaithfulnessResult
from src.evaluation import ExperimentResult, ExperimentConfig
from src.visualization import VisualizationConfig, ResultVisualizer


def load_results_from_run(run_dir: Path) -> Dict[str, List[FaithfulnessResult]]:
    detailed = run_dir / "detailed_results.json"
    if not detailed.exists():
        raise FileNotFoundError(f"No detailed_results.json in {run_dir}")
    with open(detailed, "r") as f:
        blob = json.load(f)
    out: Dict[str, List[FaithfulnessResult]] = {}
    for name, items in blob.items():
        lst: List[FaithfulnessResult] = []
        for it in items:
            lst.append(FaithfulnessResult(
                f_score=float(it.get("f_score", 0.0)),
                confidence_interval=tuple(it.get("confidence_interval", [0.0, 0.0])),
                n_samples=int(it.get("n_samples", 0)),
                baseline_performance=float(it.get("baseline_performance", 0.0)),
                explained_performance=float(it.get("explained_performance", 0.0)),
                statistical_significance=bool(it.get("statistical_significance", False)),
                p_value=float(it.get("p_value", 1.0)),
                computation_metrics=it.get("computation_metrics", {}) or {},
            ))
        out[name] = lst
    return out


def build_experiment_result(run_dir: Path, results: Dict[str, List[FaithfulnessResult]]) -> ExperimentResult:
    # Minimal config/metadata for visualization labels
    exp_name = run_dir.name
    cfg = ExperimentConfig(
        experiment_name=exp_name,
        dataset_name=exp_name,
        model_name=exp_name,
        explainer_names=list(results.keys()),
        num_samples=max((len(v) for v in results.values()), default=0),
        output_dir=str(run_dir)
    )
    # Compute basic summary and computation metrics
    summary: Dict[str, Dict[str, float]] = {}
    comp_metrics: Dict[str, Dict[str, float]] = {}
    for name, lst in results.items():
        f_scores = [r.f_score for r in lst]
        p_vals = [r.p_value for r in lst]
        total_expl_time = sum(r.computation_metrics.get('explanation_time', 0.0) for r in lst)
        total_f_time = sum(r.computation_metrics.get('faithfulness_time', 0.0) for r in lst)
        total_runtime = total_expl_time + total_f_time
        summary[name] = {
            'mean_f_score': float(np.mean(f_scores) if f_scores else 0.0),
            'std_f_score': float(np.std(f_scores) if f_scores else 0.0),
            'median_f_score': float(np.median(f_scores) if f_scores else 0.0),
            'min_f_score': float(np.min(f_scores) if f_scores else 0.0),
            'max_f_score': float(np.max(f_scores) if f_scores else 0.0),
            'mean_p_value': float(np.mean(p_vals) if p_vals else 1.0),
            'significant_results': int(sum(1 for p in p_vals if p < 0.05)),
            'total_results': len(f_scores),
            'total_computation_time': float(total_runtime),
        }
        comp_metrics[name] = {
            'total_runtime': float(total_runtime),
            'samples_processed': len(lst),
            'avg_time_per_sample': float(total_runtime / max(len(lst), 1)),
        }
    # Assemble ExperimentResult
    er = ExperimentResult(
        experiment_config=cfg,
        dataset_info={'name': exp_name},
        model_info={'name': exp_name},
        explainer_results=results,
        summary_statistics=summary,
        computation_metrics=comp_metrics,
        timestamp=str(Path()),
        total_runtime=sum(m['total_runtime'] for m in comp_metrics.values()),
    )
    return er


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate figures from run directories")
    parser.add_argument("--runs", type=str, nargs="+", required=True, help="Run directories with detailed_results.json")
    parser.add_argument("--figdir", type=str, default="figures", help="Output directory for figures")
    args = parser.parse_args()

    # Two formats: png and pdf
    for run in args.runs:
        run_dir = Path(run)
        results = load_results_from_run(run_dir)
        exp = build_experiment_result(run_dir, results)

        # PNG
        viz_png = ResultVisualizer(VisualizationConfig(output_dir=args.figdir, figure_format="png"))
        viz_png.create_faithfulness_bar_chart(results, title=f"Faithfulness Scores - {exp.experiment_config.experiment_name}")
        viz_png.create_comparison_heatmap(results, title=f"Score Comparison - {exp.experiment_config.experiment_name}")
        viz_png.create_performance_table(exp)

        # PDF
        viz_pdf = ResultVisualizer(VisualizationConfig(output_dir=args.figdir, figure_format="pdf"))
        viz_pdf.create_faithfulness_bar_chart(results, title=f"Faithfulness Scores - {exp.experiment_config.experiment_name}")
        viz_pdf.create_comparison_heatmap(results, title=f"Score Comparison - {exp.experiment_config.experiment_name}")
        viz_pdf.create_performance_table(exp)

        print(f"Figures generated for {run_dir}")


if __name__ == "__main__":
    main()


