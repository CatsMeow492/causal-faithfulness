#!/usr/bin/env python3
"""
SST-2 Experimental Pipeline for Causal-Faithfulness Metric

This script implements task 2.1: Load SST-2 dataset with proper BERT tokenization,
configure BERT-base-uncased model, run faithfulness evaluation on 200 validation instances,
and generate F-scores for SHAP, Integrated Gradients, LIME, and Random explainers.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch

# Ensure project root is on path and import from package `src`
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Core imports from src package
import src.faithfulness as faithfulness
import src.explainers as explainers
import src.masking as masking
import src.datasets as datasets
import src.models as models
import src.config as config

# Import specific classes and functions
from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig, FaithfulnessResult
from src.explainers import SHAPWrapper, IntegratedGradientsWrapper, LIMEWrapper, RandomExplainer
from src.masking import DataModality
from src.datasets import DatasetManager, DatasetSample
from src.models import ModelManager, BERTSentimentWrapper
from src.config import get_device


class SST2ExperimentalPipeline:
    """
    Complete experimental pipeline for SST-2 sentiment analysis with BERT.
    Implements task 2.1 requirements with proper error handling and reproducibility.
    """
    
    def __init__(self, random_seed: int = 42, device: Optional[torch.device] = None):
        """Initialize the experimental pipeline."""
        self.random_seed = random_seed
        self.device = device or get_device()
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        
        # Initialize managers
        self.dataset_manager = DatasetManager(device=self.device)
        self.model_manager = ModelManager(device=self.device)
        
        # Storage for loaded components
        self.dataset_samples = None
        self.model = None
        self.explainers = {}
        self.faithfulness_metric = None
        
        print(f"Initialized SST-2 experimental pipeline with seed {random_seed} on {self.device}")
    
    def load_sst2_dataset(self, num_samples: int = 200, split: str = "validation", max_length: int = 512) -> List[DatasetSample]:
        """
        Load SST-2 dataset with proper BERT tokenization and preprocessing.
        
        Args:
            num_samples: Number of validation instances to load (default: 200)
            split: Dataset split to use (default: "validation")
            
        Returns:
            List of DatasetSample objects with tokenized text
        """
        print(f"Loading SST-2 dataset: {num_samples} samples from {split} split")
        
        try:
            # Load dataset using the DatasetManager
            self.dataset_samples = self.dataset_manager.load_sst2(
                split=split,
                num_samples=num_samples,
                model_name="bert-base-uncased",
                max_length=max_length
            )
            
            print(f"Successfully loaded {len(self.dataset_samples)} SST-2 samples")
            
            # Print sample statistics
            if self.dataset_samples:
                text_lengths = [len(sample.text.split()) for sample in self.dataset_samples]
                token_lengths = [sample.tokens.numel() for sample in self.dataset_samples]
                
                print(f"Text length stats: mean={np.mean(text_lengths):.1f}, "
                      f"std={np.std(text_lengths):.1f}, "
                      f"max={np.max(text_lengths)}")
                print(f"Token length stats: mean={np.mean(token_lengths):.1f}, "
                      f"std={np.std(token_lengths):.1f}, "
                      f"max={np.max(token_lengths)}")
            
            return self.dataset_samples
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SST-2 dataset: {str(e)}")
    
    def configure_bert_model(self, model_name: str = "textattack/bert-base-uncased-SST-2") -> BERTSentimentWrapper:
        """
        Configure BERT-base-uncased model for sentiment classification.
        
        Args:
            model_name: HuggingFace model name for BERT SST-2
            
        Returns:
            Loaded BERT model wrapper
        """
        print(f"Configuring BERT model: {model_name}")
        
        try:
            # Load BERT model using ModelManager
            self.model = self.model_manager.load_bert_sst2(
                model_name=model_name,
                use_quantization=False  # Disable for reproducibility
            )
            
            print(f"Successfully loaded BERT model on {self.model.device}")
            
            # Test model with a sample prediction
            if self.dataset_samples and len(self.dataset_samples) > 0:
                test_sample = self.dataset_samples[0]
                test_pred = self.model.predict(test_sample)
                print(f"Test prediction - Class: {test_pred.predicted_class}, "
                      f"Confidence: {test_pred.confidence:.4f}")
            
            return self.model
            
        except Exception as e:
            raise RuntimeError(f"Failed to configure BERT model: {str(e)}")
    
    def initialize_explainers(self, selected: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Initialize SHAP, Integrated Gradients, LIME, and Random explainers.
        
        Returns:
            Dictionary of initialized explainer objects
        """
        print("Initializing explainers...")
        
        try:
            available: Dict[str, Any] = {}
            # SHAP Kernel Explainer (optional)
            try:
                available["SHAP"] = SHAPWrapper(
                    explainer_type="kernel",
                    n_samples=500,
                    random_seed=self.random_seed
                )
                print("  - SHAP available")
            except Exception as e:
                warnings.warn(f"SHAP not available, skipping: {e}")
            # Integrated Gradients
            available["IntegratedGradients"] = IntegratedGradientsWrapper(
                n_steps=20,
                baseline_strategy="zero",
                random_seed=self.random_seed,
                internal_batch_size=16
            )
            print("  - IntegratedGradients available")
            # LIME (optional)
            try:
                available["LIME"] = LIMEWrapper(
                    n_samples=200,
                    modality="tabular",
                    random_seed=self.random_seed
                )
                print("  - LIME available")
            except Exception as e:
                warnings.warn(f"LIME not available, skipping: {e}")
            # Random Baseline
            available["Random"] = RandomExplainer(
                random_seed=self.random_seed,
                distribution="uniform",
                scale=1.0
            )
            print("  - Random available")

            # Filter by selection
            if selected:
                names = {n.strip() for n in selected}
                self.explainers = {k: v for k, v in available.items() if k in names}
            else:
                self.explainers = available

            print(f"Successfully initialized {len(self.explainers)} explainers")
            return self.explainers

        except Exception as e:
            raise RuntimeError(f"Failed to initialize explainers: {str(e)}")
    
    def setup_faithfulness_metric(self, n_samples: int = 1000) -> FaithfulnessMetric:
        """
        Set up the faithfulness metric with appropriate configuration.
        
        Args:
            n_samples: Number of Monte Carlo samples for metric computation
            
        Returns:
            Configured FaithfulnessMetric instance
        """
        print(f"Setting up faithfulness metric with {n_samples} Monte Carlo samples")
        
        try:
            # Create faithfulness configuration
            config = FaithfulnessConfig(
                n_samples=n_samples,
                baseline_strategy="random",
                masking_strategy="pad",
                confidence_level=0.95,
                batch_size=16,
                random_seed=self.random_seed,
                device=self.device,
                numerical_epsilon=1e-8,
                top_fraction=0.1
            )
            
            # Initialize faithfulness metric
            self.faithfulness_metric = FaithfulnessMetric(
                config=config,
                modality=DataModality.TEXT
            )
            
            print("Successfully configured faithfulness metric")
            return self.faithfulness_metric
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup faithfulness metric: {str(e)}")
    
    def create_model_prediction_function(self):
        """Create a model prediction function compatible with explainers."""
        def model_fn(inputs):
            """Model prediction function for explainers."""
            if isinstance(inputs, torch.Tensor):
                # Handle tensor inputs - convert to DatasetSample
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)
                
                # Create a proper DatasetSample object
                sample = DatasetSample(
                    text="",  # Placeholder
                    tokens=inputs.squeeze(0) if inputs.dim() > 1 else inputs,
                    attention_mask=torch.ones_like(inputs.squeeze(0) if inputs.dim() > 1 else inputs),
                    label=None,
                    metadata={}
                )
                pred = self.model.predict(sample)
                return pred.logits.unsqueeze(0) if pred.logits.dim() == 1 else pred.logits
                
            elif isinstance(inputs, np.ndarray):
                # Convert numpy to tensor
                tensor_input = torch.from_numpy(inputs).long()
                return model_fn(tensor_input)
                
            elif isinstance(inputs, DatasetSample):
                pred = self.model.predict(inputs)
                return pred.logits.unsqueeze(0) if pred.logits.dim() == 1 else pred.logits
                
            else:
                raise ValueError(f"Unsupported input type: {type(inputs)}")
        
        return model_fn
    
    def evaluate_single_explainer(
        self, 
        explainer_name: str, 
        explainer: Any, 
        sample: DatasetSample,
        model_fn: Any
    ) -> FaithfulnessResult:
        """
        Evaluate a single explainer on a single sample.
        
        Args:
            explainer_name: Name of the explainer
            explainer: Explainer instance
            sample: Dataset sample to evaluate
            model_fn: Model prediction function
            
        Returns:
            FaithfulnessResult for this sample
        """
        try:
            # Create explainer function for faithfulness metric
            def explainer_fn(model, data):
                """Explainer function wrapper."""
                # For text data, pass the DatasetSample directly to explainers
                if explainer_name == "SHAP":
                    # For SHAP, we need background data
                    background_data = torch.zeros_like(sample.tokens).unsqueeze(0)
                    return explainer.explain(
                        model,
                        sample.tokens,
                        background_data=background_data
                    )
                elif explainer_name == "LIME":
                    # For LIME, we need training data
                    # Use the current sample as training data (simplified)
                    training_data = sample.tokens.cpu().numpy().reshape(1, -1)
                    return explainer.explain(
                        model,
                        sample.tokens,
                        training_data=training_data
                    )
                else:
                    # For IG (embedding-level) and Random
                    if explainer_name == "IntegratedGradients":
                        # Access underlying HF model for embeddings
                        hf = self.model.model if hasattr(self.model, 'model') else None
                        attn = sample.attention_mask.unsqueeze(0)
                        return explainer.explain(
                            model,
                            sample.tokens,
                            use_embeddings=True,
                            hf_model=hf,
                            attention_mask=attn
                        )
                    return explainer.explain(model, sample.tokens)
            
            # Compute faithfulness score
            result = self.faithfulness_metric.compute_faithfulness_score(
                model=model_fn,
                explainer=explainer_fn,
                data=sample,  # Metric handles DatasetSample and will pass tokens to masker/baseline
                target_class=sample.label
            )
            
            return result
            
        except Exception as e:
            warnings.warn(f"Failed to evaluate {explainer_name} on sample: {str(e)}")
            # Return a default result with zero score
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
        """
        Run faithfulness evaluation on all samples with all explainers.
        
        Args:
            max_samples: Maximum number of samples to evaluate (None for all)
            
        Returns:
            Dictionary mapping explainer names to lists of FaithfulnessResults
        """
        if not self.dataset_samples:
            raise RuntimeError("Dataset not loaded. Call load_sst2_dataset() first.")
        if not self.model:
            raise RuntimeError("Model not configured. Call configure_bert_model() first.")
        if not self.explainers:
            raise RuntimeError("Explainers not initialized. Call initialize_explainers() first.")
        if not self.faithfulness_metric:
            raise RuntimeError("Faithfulness metric not setup. Call setup_faithfulness_metric() first.")
        
        # Determine samples to evaluate
        samples_to_evaluate = self.dataset_samples
        if max_samples is not None:
            samples_to_evaluate = samples_to_evaluate[:max_samples]
        
        print(f"Running faithfulness evaluation on {len(samples_to_evaluate)} samples")
        print(f"Explainers: {list(self.explainers.keys())}")
        
        # Create model prediction function
        model_fn = self.create_model_prediction_function()
        
        # Results storage
        results = {name: [] for name in self.explainers.keys()}
        
        # Evaluate each explainer on each sample
        for i, sample in enumerate(samples_to_evaluate):
            print(f"\nEvaluating sample {i+1}/{len(samples_to_evaluate)}")
            print(f"Text: {sample.text[:100]}..." if len(sample.text) > 100 else f"Text: {sample.text}")
            print(f"Label: {sample.label}")
            
            for explainer_name, explainer in self.explainers.items():
                print(f"  - Evaluating {explainer_name}...")
                
                start_time = time.time()
                result = self.evaluate_single_explainer(
                    explainer_name, explainer, sample, model_fn
                )
                eval_time = time.time() - start_time
                
                results[explainer_name].append(result)
                
                print(f"    F-score: {result.f_score:.4f}, "
                      f"CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}], "
                      f"Time: {eval_time:.2f}s")
        
        return results
    
    def generate_f_scores_summary(self, results: Dict[str, List[FaithfulnessResult]]) -> Dict[str, Dict[str, float]]:
        """
        Generate F-score summary statistics for all explainers.
        
        Args:
            results: Results from run_faithfulness_evaluation()
            
        Returns:
            Dictionary with summary statistics for each explainer
        """
        summary = {}
        
        for explainer_name, explainer_results in results.items():
            if not explainer_results:
                summary[explainer_name] = {
                    "status": "no_results",
                    "n_samples": 0
                }
                continue
            
            # Extract F-scores
            f_scores = [r.f_score for r in explainer_results]
            p_values = [r.p_value for r in explainer_results]
            significant_count = sum(1 for r in explainer_results if r.statistical_significance)
            
            # Compute statistics
            summary[explainer_name] = {
                "status": "success",
                "n_samples": len(f_scores),
                "mean_f_score": float(np.mean(f_scores)),
                "std_f_score": float(np.std(f_scores)),
                "median_f_score": float(np.median(f_scores)),
                "min_f_score": float(np.min(f_scores)),
                "max_f_score": float(np.max(f_scores)),
                "q25_f_score": float(np.percentile(f_scores, 25)),
                "q75_f_score": float(np.percentile(f_scores, 75)),
                "mean_p_value": float(np.mean(p_values)),
                "significant_fraction": significant_count / len(f_scores),
                "confidence_interval_95": [
                    float(np.percentile(f_scores, 2.5)),
                    float(np.percentile(f_scores, 97.5))
                ]
            }
        
        return summary
    
    def print_results_summary(self, summary: Dict[str, Dict[str, float]]):
        """Print a formatted summary of results."""
        print(f"\n{'='*80}")
        print("SST-2 FAITHFULNESS EVALUATION RESULTS")
        print(f"{'='*80}")
        
        print(f"\n{'Explainer':<20} {'Status':<10} {'N':<5} {'Mean F-score':<12} {'Std':<8} {'95% CI':<20} {'Sig %':<8}")
        print("-" * 90)
        
        for explainer_name, stats in summary.items():
            if stats["status"] != "success":
                print(f"{explainer_name:<20} {'FAILED':<10} {'-':<5} {'-':<12} {'-':<8} {'-':<20} {'-':<8}")
                continue
            
            mean_f = stats["mean_f_score"]
            std_f = stats["std_f_score"]
            ci = stats["confidence_interval_95"]
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            sig_pct = stats["significant_fraction"] * 100
            n_samples = stats["n_samples"]
            
            print(f"{explainer_name:<20} {'SUCCESS':<10} {n_samples:<5} {mean_f:<12.4f} {std_f:<8.4f} {ci_str:<20} {sig_pct:<8.1f}")
        
        # Sanity check
        if "Random" in summary and summary["Random"]["status"] == "success":
            random_mean = summary["Random"]["mean_f_score"]
            sanity_passed = random_mean < 0.3  # Random should be low
            print(f"\nSanity Check: {'PASSED' if sanity_passed else 'FAILED'} "
                  f"(Random baseline F-score: {random_mean:.4f})")
        
        # Best explainer
        successful_explainers = {name: stats for name, stats in summary.items() 
                               if stats["status"] == "success"}
        if successful_explainers:
            best_explainer = max(successful_explainers.keys(), 
                               key=lambda x: successful_explainers[x]["mean_f_score"])
            best_score = successful_explainers[best_explainer]["mean_f_score"]
            print(f"Best Explainer: {best_explainer} (F-score: {best_score:.4f})")
    
    def save_results(self, results: Dict[str, List[FaithfulnessResult]], 
                    summary: Dict[str, Dict[str, float]], 
                    output_dir: str = "results/sst2_experiments"):
        """
        Save experimental results to files.
        
        Args:
            results: Full results from evaluation
            summary: Summary statistics
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_path = output_path / "f_scores_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results (convert FaithfulnessResult to dict)
        detailed_results = {}
        for explainer_name, explainer_results in results.items():
            detailed_results[explainer_name] = []
            for result in explainer_results:
                detailed_results[explainer_name].append({
                    "f_score": result.f_score,
                    "confidence_interval": result.confidence_interval,
                    "n_samples": result.n_samples,
                    "baseline_performance": result.baseline_performance,
                    "explained_performance": result.explained_performance,
                    "statistical_significance": result.statistical_significance,
                    "p_value": result.p_value,
                    "computation_metrics": result.computation_metrics
                })
        
        detailed_path = output_path / "detailed_results.json"
        with open(detailed_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save experiment configuration
        config = {
            "experiment": "SST-2 Faithfulness Evaluation",
            "dataset": "SST-2 validation set",
            "model": "BERT-base-uncased fine-tuned on SST-2",
            "n_samples": len(self.dataset_samples) if self.dataset_samples else 0,
            "explainers": list(self.explainers.keys()) if self.explainers else [],
            "random_seed": self.random_seed,
            "device": str(self.device),
            "faithfulness_config": {
                "n_samples": self.faithfulness_metric.config.n_samples if self.faithfulness_metric else None,
                "baseline_strategy": self.faithfulness_metric.config.baseline_strategy if self.faithfulness_metric else None,
                "masking_strategy": self.faithfulness_metric.config.masking_strategy if self.faithfulness_metric else None,
                "confidence_level": self.faithfulness_metric.config.confidence_level if self.faithfulness_metric else None
            }
        }
        
        config_path = output_path / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print(f"  - Summary: {summary_path}")
        print(f"  - Detailed results: {detailed_path}")
        print(f"  - Configuration: {config_path}")
    
    def run_complete_pipeline(self, 
                            num_samples: int = 200, 
                            max_eval_samples: Optional[int] = None,
                            output_dir: str = "results/sst2_experiments") -> Dict[str, Dict[str, float]]:
        """
        Run the complete SST-2 experimental pipeline.
        
        Args:
            num_samples: Number of dataset samples to load
            max_eval_samples: Maximum samples to evaluate (None for all loaded)
            output_dir: Directory to save results
            
        Returns:
            Summary statistics dictionary
        """
        print("Starting complete SST-2 experimental pipeline...")
        
        try:
            # Step 1: Load dataset
            self.load_sst2_dataset(num_samples=num_samples)
            
            # Step 2: Configure model
            self.configure_bert_model()
            
            # Step 3: Initialize explainers
            self.initialize_explainers()
            
            # Step 4: Setup faithfulness metric
            self.setup_faithfulness_metric()
            
            # Step 5: Run evaluation
            results = self.run_faithfulness_evaluation(max_samples=max_eval_samples)
            
            # Step 6: Generate summary
            summary = self.generate_f_scores_summary(results)
            
            # Step 7: Print results
            self.print_results_summary(summary)
            
            # Step 8: Save results
            self.save_results(results, summary, output_dir)
            
            print(f"\nSST-2 experimental pipeline completed successfully!")
            return summary
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            raise e


def main():
    """Main function to run SST-2 experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SST-2 faithfulness experiments")
    parser.add_argument("--num-samples", type=int, default=200, 
                       help="Number of dataset samples to load (default: 200)")
    parser.add_argument("--max-eval-samples", type=int, default=None,
                       help="Maximum samples to evaluate (default: all loaded)")
    parser.add_argument("--output-dir", type=str, default="results/sst2_experiments",
                       help="Output directory for results")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--faithfulness-samples", type=int, default=1000,
                       help="Monte Carlo samples for faithfulness metric")
    parser.add_argument(
        "--explainers",
        type=str,
        default=None,
        help="Comma-separated list of explainers to run (e.g., IntegratedGradients,Random)"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    print(f"Running SST-2 experiments with:")
    print(f"  - Dataset samples: {args.num_samples}")
    print(f"  - Max eval samples: {args.max_eval_samples or 'all'}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Random seed: {args.random_seed}")
    print(f"  - Device: {device}")
    
    # Create and run pipeline
    pipeline = SST2ExperimentalPipeline(
        random_seed=args.random_seed,
        device=device
    )
    
    # Parse explainer selection
    selected = [x.strip() for x in args.explainers.split(',')] if args.explainers else None
    
    # Override pipeline steps to pass parameters
    print("Starting complete SST-2 experimental pipeline...")
    pipeline.load_sst2_dataset(num_samples=args.num_samples)
    pipeline.configure_bert_model()
    pipeline.initialize_explainers(selected=selected)
    pipeline.setup_faithfulness_metric(n_samples=args.faithfulness_samples)
    results = pipeline.run_faithfulness_evaluation(max_samples=args.max_eval_samples)
    summary = pipeline.generate_f_scores_summary(results)
    pipeline.print_results_summary(summary)
    pipeline.save_results(results, summary, args.output_dir)
    print("\nSST-2 experimental pipeline completed successfully!")
    
    # Old path kept for reference:
    # summary = pipeline.run_complete_pipeline(
    #     num_samples=args.num_samples,
    #     max_eval_samples=args.max_eval_samples,
    #     output_dir=args.output_dir
    # )
    
    return summary


if __name__ == "__main__":
    main()