#!/usr/bin/env python3
"""
WikiText-2 Experimental Pipeline for Causal-Faithfulness Metric

Implements task 2.2: Load WikiText-2 dataset with GPT-2 tokenization,
configure GPT-2-small model, run faithfulness evaluation on 200 validation
instances, and generate F-scores for applicable explainers (SHAP, IG, Random).
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# Ensure project root on path for `src` package
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Core imports
import src.faithfulness as faithfulness
import src.explainers as explainers
import src.masking as masking
import src.datasets as datasets
import src.models as models
import src.config as config

from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig, FaithfulnessResult
from src.explainers import SHAPWrapper, IntegratedGradientsWrapper, RandomExplainer
from src.masking import DataModality
from src.datasets import DatasetManager, DatasetSample
from src.models import ModelManager
from src.config import get_device


class WikiText2ExperimentalPipeline:
    """Complete experimental pipeline for WikiText-2 with GPT-2."""

    def __init__(self, random_seed: int = 42, device: Optional[torch.device] = None):
        self.random_seed = random_seed
        self.device = device or get_device()

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

        self.dataset_manager = DatasetManager(device=self.device)
        self.model_manager = ModelManager(device=self.device)

        self.dataset_samples: Optional[List[DatasetSample]] = None
        self.model = None
        self.explainers: Dict[str, Any] = {}
        self.faithfulness_metric: Optional[FaithfulnessMetric] = None

        print(f"Initialized WikiText-2 pipeline with seed {random_seed} on {self.device}")

    def load_wikitext2_dataset(self, num_samples: int = 200, split: str = "validation", max_length: int = 512) -> List[DatasetSample]:
        print(f"Loading WikiText-2 dataset: {num_samples} samples from {split} split")
        try:
            self.dataset_samples = self.dataset_manager.load_wikitext2(
                split=split,
                num_samples=num_samples,
                model_name="gpt2",
                max_length=max_length
            )

            print(f"Successfully loaded {len(self.dataset_samples)} WikiText-2 samples")
            if self.dataset_samples:
                token_lengths = [sample.tokens.numel() for sample in self.dataset_samples]
                print(
                    f"Token length stats: mean={np.mean(token_lengths):.1f}, std={np.std(token_lengths):.1f}, max={np.max(token_lengths)}"
                )
            return self.dataset_samples
        except Exception as e:
            raise RuntimeError(f"Failed to load WikiText-2 dataset: {str(e)}")

    def configure_gpt2_model(self, model_name: str = "gpt2"):
        print(f"Configuring GPT-2 model: {model_name}")
        try:
            self.model = self.model_manager.load_gpt2_small(
                model_name=model_name,
                use_quantization=False
            )
            print(f"Successfully loaded GPT-2 model on {self.model.device}")

            if self.dataset_samples and len(self.dataset_samples) > 0:
                test_sample = self.dataset_samples[0]
                test_pred = self.model.predict(test_sample)
                print(
                    f"Test perplexity (approx): {test_pred.perplexity if test_pred.perplexity is not None else 'N/A'}"
                )
            return self.model
        except Exception as e:
            raise RuntimeError(f"Failed to configure GPT-2 model: {str(e)}")

    def initialize_explainers(self, selected: Optional[List[str]] = None) -> Dict[str, Any]:
        print("Initializing explainers (SHAP, IG, Random)...")
        try:
            available: Dict[str, Any] = {}
            try:
                available["SHAP"] = SHAPWrapper(
                    explainer_type="kernel",
                    n_samples=200,
                    random_seed=self.random_seed
                )
                print("  - SHAP available")
            except Exception as e:
                warnings.warn(f"SHAP not available, skipping: {e}")

            available["IntegratedGradients"] = IntegratedGradientsWrapper(
                n_steps=20,
                baseline_strategy="zero",
                random_seed=self.random_seed,
                internal_batch_size=16
            )
            print("  - IntegratedGradients available")

            available["Random"] = RandomExplainer(
                random_seed=self.random_seed,
                distribution="uniform",
                scale=1.0
            )
            print("  - Random available")

            # Filter selection
            if selected:
                names = {n.strip() for n in selected}
                self.explainers = {k: v for k, v in available.items() if k in names}
            else:
                self.explainers = available

            print(f"Successfully initialized {len(self.explainers)} explainers")
            return self.explainers
        except Exception as e:
            raise RuntimeError(f"Failed to initialize explainers: {str(e)}")

    def setup_faithfulness_metric(self, n_samples: int = 500) -> FaithfulnessMetric:
        print(f"Setting up faithfulness metric with {n_samples} Monte Carlo samples")
        try:
            cfg = FaithfulnessConfig(
                n_samples=n_samples,
                baseline_strategy="random",
                masking_strategy="pad",
                confidence_level=0.95,
                batch_size=8,
                random_seed=self.random_seed,
                device=self.device,
                numerical_epsilon=1e-8
            )
            self.faithfulness_metric = FaithfulnessMetric(config=cfg, modality=DataModality.TEXT)
            print("Successfully configured faithfulness metric")
            return self.faithfulness_metric
        except Exception as e:
            raise RuntimeError(f"Failed to setup faithfulness metric: {str(e)}")

    def create_model_prediction_function(self):
        """Create a model function returning logits over vocab at final position."""

        def model_fn(inputs):
            # Accept DatasetSample or token id tensors/arrays; return (batch, vocab_size)
            if isinstance(inputs, DatasetSample):
                input_ids = inputs.tokens.unsqueeze(0).to(self.model.device)
                attention_mask = inputs.attention_mask.unsqueeze(0).to(self.model.device)
            elif isinstance(inputs, torch.Tensor):
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)
                input_ids = inputs.to(self.model.device)
                attention_mask = torch.ones_like(input_ids)
            elif isinstance(inputs, np.ndarray):
                tensor_input = torch.from_numpy(inputs).long()
                if tensor_input.dim() == 1:
                    tensor_input = tensor_input.unsqueeze(0)
                input_ids = tensor_input.to(self.model.device)
                attention_mask = torch.ones_like(input_ids)
            elif isinstance(inputs, dict) and 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                if isinstance(input_ids, np.ndarray):
                    input_ids = torch.from_numpy(input_ids).long()
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                input_ids = input_ids.to(self.model.device)
                attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
                if isinstance(attention_mask, np.ndarray):
                    attention_mask = torch.from_numpy(attention_mask)
                attention_mask = attention_mask.to(self.model.device)
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")

            with torch.no_grad():
                outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                logits = outputs.logits  # (batch, seq_len, vocab)
                last_logits = logits[:, -1, :]  # (batch, vocab)
                return last_logits

        return model_fn

    def evaluate_single_explainer(self, explainer_name: str, explainer: Any, sample: DatasetSample, model_fn: Any) -> FaithfulnessResult:
        try:
            def explainer_fn(model, data):
                if explainer_name == "SHAP":
                    background = sample.tokens.cpu().numpy().reshape(1, -1)
                    return explainer.explain(model, data, background_data=background)
                elif explainer_name == "IntegratedGradients":
                    hf = self.model.model if hasattr(self.model, 'model') else None
                    attn = sample.attention_mask.unsqueeze(0)
                    return explainer.explain(
                        model,
                        sample.tokens,
                        use_embeddings=True,
                        hf_model=hf,
                        attention_mask=attn
                    )
                else:
                    return explainer.explain(model, data)

            # Use the last token id as a proxy target class (vocab index)
            target_class = int(sample.tokens[-1].item())

            result = self.faithfulness_metric.compute_faithfulness_score(
                model=model_fn,
                explainer=explainer_fn,
                data=sample,
                target_class=target_class
            )
            return result
        except Exception as e:
            warnings.warn(f"Failed to evaluate {explainer_name} on sample: {str(e)}")
            return FaithfulnessResult(
                f_score=0.0,
                confidence_interval=(0.0, 0.0),
                n_samples=0,
                baseline_performance=0.0,
                explained_performance=0.0,
                statistical_significance=False,
                p_value=1.0,
                computation_metrics={"error": str(e)}
            )

    def run_faithfulness_evaluation(self, max_samples: Optional[int] = None) -> Dict[str, List[FaithfulnessResult]]:
        if not self.dataset_samples:
            raise RuntimeError("Dataset not loaded. Call load_wikitext2_dataset() first.")
        if not self.model:
            raise RuntimeError("Model not configured. Call configure_gpt2_model() first.")
        if not self.explainers:
            raise RuntimeError("Explainers not initialized. Call initialize_explainers() first.")
        if not self.faithfulness_metric:
            raise RuntimeError("Faithfulness metric not setup. Call setup_faithfulness_metric() first.")

        samples = self.dataset_samples if max_samples is None else self.dataset_samples[:max_samples]
        print(f"Running faithfulness evaluation on {len(samples)} samples")
        print(f"Explainers: {list(self.explainers.keys())}")

        model_fn = self.create_model_prediction_function()
        results: Dict[str, List[FaithfulnessResult]] = {name: [] for name in self.explainers.keys()}

        for i, sample in enumerate(samples):
            text_preview = sample.text.replace("\n", " ")
            print(f"\nEvaluating sample {i+1}/{len(samples)}")
            print(f"Text: {text_preview[:120]}..." if len(text_preview) > 120 else f"Text: {text_preview}")

            for explainer_name, ex in self.explainers.items():
                print(f"  - Evaluating {explainer_name}...")
                start = time.time()
                res = self.evaluate_single_explainer(explainer_name, ex, sample, model_fn)
                elapsed = time.time() - start
                results[explainer_name].append(res)
                print(
                    f"    F-score: {res.f_score:.4f}, CI: [{res.confidence_interval[0]:.4f}, {res.confidence_interval[1]:.4f}], Time: {elapsed:.2f}s"
                )

        return results

    def generate_f_scores_summary(self, results: Dict[str, List[FaithfulnessResult]]) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for name, lst in results.items():
            if not lst:
                summary[name] = {"status": "no_results", "n_samples": 0}
                continue
            f_scores = [r.f_score for r in lst]
            p_vals = [r.p_value for r in lst]
            significant = sum(1 for r in lst if r.statistical_significance)
            summary[name] = {
                "status": "success",
                "n_samples": len(f_scores),
                "mean_f_score": float(np.mean(f_scores)),
                "std_f_score": float(np.std(f_scores)),
                "median_f_score": float(np.median(f_scores)),
                "min_f_score": float(np.min(f_scores)),
                "max_f_score": float(np.max(f_scores)),
                "q25_f_score": float(np.percentile(f_scores, 25)),
                "q75_f_score": float(np.percentile(f_scores, 75)),
                "mean_p_value": float(np.mean(p_vals)),
                "significant_fraction": significant / len(f_scores),
                "confidence_interval_95": [
                    float(np.percentile(f_scores, 2.5)),
                    float(np.percentile(f_scores, 97.5)),
                ],
            }
        return summary

    def print_results_summary(self, summary: Dict[str, Dict[str, float]]):
        print(f"\n{'='*80}")
        print("WikiText-2 FAITHFULNESS EVALUATION RESULTS")
        print(f"{'='*80}")

        print(f"\n{'Explainer':<20} {'Status':<10} {'N':<5} {'Mean F-score':<12} {'Std':<8} {'95% CI':<20} {'Sig %':<8}")
        print("-" * 90)
        for name, stats in summary.items():
            if stats["status"] != "success":
                print(f"{name:<20} {'FAILED':<10} {'-':<5} {'-':<12} {'-':<8} {'-':<20} {'-':<8}")
                continue
            mean_f = stats["mean_f_score"]
            std_f = stats["std_f_score"]
            ci = stats["confidence_interval_95"]
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            sig_pct = stats["significant_fraction"] * 100
            n = stats["n_samples"]
            print(f"{name:<20} {'SUCCESS':<10} {n:<5} {mean_f:<12.4f} {std_f:<8.4f} {ci_str:<20} {sig_pct:<8.1f}")

        # Sanity check for Random baseline
        if "Random" in summary and summary["Random"]["status"] == "success":
            random_mean = summary["Random"]["mean_f_score"]
            print(
                f"\nSanity Check (Random low): {'PASSED' if random_mean < 0.3 else 'FAILED'} (Random F-score: {random_mean:.4f})"
            )

        successful = {k: v for k, v in summary.items() if v.get("status") == "success"}
        if successful:
            best = max(successful.keys(), key=lambda k: successful[k]["mean_f_score"])
            print(f"Best Explainer: {best} (F-score: {successful[best]['mean_f_score']:.4f})")

    def save_results(self, results: Dict[str, List[FaithfulnessResult]], summary: Dict[str, Dict[str, float]], output_dir: str = "results/wikitext2_experiments"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "f_scores_summary.json", 'w') as f:
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

        with open(out / "detailed_results.json", 'w') as f:
            json.dump(detailed, f, indent=2)

        config_blob = {
            "experiment": "WikiText-2 Faithfulness Evaluation",
            "dataset": "WikiText-2 validation set",
            "model": "GPT-2-small",
            "n_samples_dataset": len(self.dataset_samples) if self.dataset_samples else 0,
            "explainers": list(self.explainers.keys()) if self.explainers else [],
            "random_seed": self.random_seed,
            "device": str(self.device),
            "faithfulness_config": {
                "n_samples": self.faithfulness_metric.config.n_samples if self.faithfulness_metric else None,
                "baseline_strategy": self.faithfulness_metric.config.baseline_strategy if self.faithfulness_metric else None,
                "masking_strategy": self.faithfulness_metric.config.masking_strategy if self.faithfulness_metric else None,
                "confidence_level": self.faithfulness_metric.config.confidence_level if self.faithfulness_metric else None,
            },
        }

        with open(out / "experiment_config.json", 'w') as f:
            json.dump(config_blob, f, indent=2)

        print(f"\nResults saved to: {out}")

    def run_complete_pipeline(self, num_samples: int = 200, max_eval_samples: Optional[int] = None, output_dir: str = "results/wikitext2_experiments") -> Dict[str, Dict[str, float]]:
        print("Starting complete WikiText-2 experimental pipeline...")
        try:
            self.load_wikitext2_dataset(num_samples=num_samples)
            self.configure_gpt2_model()
            self.initialize_explainers()
            self.setup_faithfulness_metric()
            results = self.run_faithfulness_evaluation(max_samples=max_eval_samples)
            summary = self.generate_f_scores_summary(results)
            self.print_results_summary(summary)
            self.save_results(results, summary, output_dir)
            print("\nWikiText-2 experimental pipeline completed successfully!")
            return summary
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            raise e


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run WikiText-2 faithfulness experiments")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of dataset samples to load (default: 200)")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Maximum samples to evaluate (default: all loaded)")
    parser.add_argument("--output-dir", type=str, default="results/wikitext2_experiments", help="Output directory for results")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--faithfulness-samples", type=int, default=500, help="Monte Carlo samples for metric")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--explainers", type=str, default=None, help="Comma-separated list of explainers")

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print("Running WikiText-2 experiments with:")
    print(f"  - Dataset samples: {args.num_samples}")
    print(f"  - Max eval samples: {args.max_eval_samples or 'all'}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Random seed: {args.random_seed}")
    print(f"  - Device: {device}")

    pipeline = WikiText2ExperimentalPipeline(random_seed=args.random_seed, device=device)
    # Selection
    selected = [x.strip() for x in args.explainers.split(',')] if args.explainers else None
    # Run with overrides
    print("Starting complete WikiText-2 experimental pipeline...")
    pipeline.load_wikitext2_dataset(num_samples=args.num_samples, max_length=args.max_length)
    pipeline.configure_gpt2_model()
    pipeline.initialize_explainers(selected=selected)
    pipeline.setup_faithfulness_metric(n_samples=args.faithfulness_samples)
    results = pipeline.run_faithfulness_evaluation(max_samples=args.max_eval_samples)
    summary = pipeline.generate_f_scores_summary(results)
    pipeline.print_results_summary(summary)
    pipeline.save_results(results, summary, args.output_dir)
    print("\nWikiText-2 experimental pipeline completed successfully!")
    return summary


if __name__ == "__main__":
    main()


