"""
Evaluation pipeline for causal-faithfulness metric experiments.
Provides model-agnostic evaluation with batch processing and progress tracking.
"""

import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

from .config import get_device, get_batch_size, DEFAULT_CONFIG
from .datasets import DatasetSample, DatasetManager, SST2DatasetLoader, WikiText2DatasetLoader
from .models import BaseModelWrapper, BERTSentimentWrapper, GPT2LanguageModelWrapper, ModelManager, ModelPrediction
from .explainers import ExplainerWrapper, Attribution
from .faithfulness import FaithfulnessMetric, FaithfulnessConfig, FaithfulnessResult, DataModality


@dataclass
class ExperimentConfig:
    """Configuration for evaluation experiments."""
    experiment_name: str
    dataset_name: str
    model_name: str
    explainer_names: List[str]
    num_samples: int = 200
    batch_size: int = 32
    random_seed: int = 42
    save_intermediate: bool = True
    output_dir: str = "results"
    faithfulness_config: Optional[FaithfulnessConfig] = None
    
    def __post_init__(self):
        """Set default faithfulness config if not provided."""
        if self.faithfulness_config is None:
            self.faithfulness_config = FaithfulnessConfig(
                n_samples=1000,
                random_seed=self.random_seed,
                batch_size=self.batch_size
            )


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_config: ExperimentConfig
    dataset_info: Dict[str, Any]
    model_info: Dict[str, Any]
    explainer_results: Dict[str, List[FaithfulnessResult]]
    summary_statistics: Dict[str, Dict[str, float]]
    computation_metrics: Dict[str, Dict[str, float]]
    timestamp: str
    total_runtime: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save experiment results to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as JSON for readability
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        # Also save as pickle for full object preservation
        pickle_path = filepath.replace('.json', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentResult':
        """Load experiment results from file."""
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            # Try to load from JSON (limited functionality)
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls(**data)


class EvaluationPipeline:
    """
    Main evaluation pipeline for causal-faithfulness experiments.
    Handles dataset loading, model integration, and batch evaluation.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize evaluation pipeline with configuration."""
        self.config = config
        self.device = get_device()
        
        # Initialize managers
        self.dataset_manager = DatasetManager(device=self.device)
        self.model_manager = ModelManager(device=self.device)
        
        # Storage for loaded components
        self.dataset_samples = []
        self.model = None
        self.explainers = {}
        self.faithfulness_metric = None
        
        # Progress tracking
        self.progress_callback = None
        self.intermediate_results = []
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set callback function for progress updates."""
        self.progress_callback = callback
    
    def _update_progress(self, message: str, progress: float = None):
        """Update progress with message and optional progress value."""
        if self.progress_callback:
            self.progress_callback(message, progress)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def load_dataset(self) -> List[DatasetSample]:
        """Load dataset based on configuration."""
        self._update_progress(f"Loading {self.config.dataset_name} dataset...")
        
        if self.config.dataset_name.lower() == "sst2":
            self.dataset_samples = self.dataset_manager.load_sst2(
                split="validation",
                num_samples=self.config.num_samples
            )
            self.modality = DataModality.TEXT
            
        elif self.config.dataset_name.lower() == "wikitext2":
            self.dataset_samples = self.dataset_manager.load_wikitext2(
                split="validation", 
                num_samples=self.config.num_samples
            )
            self.modality = DataModality.TEXT
            
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
        
        self._update_progress(f"Loaded {len(self.dataset_samples)} samples")
        return self.dataset_samples
    
    def load_model(self) -> BaseModelWrapper:
        """Load model based on configuration."""
        self._update_progress(f"Loading {self.config.model_name} model...")
        
        if "bert" in self.config.model_name.lower() or self.config.dataset_name.lower() == "sst2":
            self.model = self.model_manager.load_bert_sst2(
                model_name=self.config.model_name if "bert" in self.config.model_name.lower() else "textattack/bert-base-uncased-SST-2"
            )
            
        elif "gpt2" in self.config.model_name.lower() or self.config.dataset_name.lower() == "wikitext2":
            self.model = self.model_manager.load_gpt2_small(
                model_name=self.config.model_name if "gpt2" in self.config.model_name.lower() else "gpt2"
            )
            
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
        
        self._update_progress("Model loaded successfully")
        return self.model
    
    def load_explainers(self, explainer_instances: Dict[str, ExplainerWrapper]):
        """Load explainer instances."""
        self._update_progress("Loading explainers...")
        
        for name in self.config.explainer_names:
            if name not in explainer_instances:
                raise ValueError(f"Explainer {name} not provided in explainer_instances")
            self.explainers[name] = explainer_instances[name]
        
        self._update_progress(f"Loaded {len(self.explainers)} explainers: {list(self.explainers.keys())}")
    
    def initialize_faithfulness_metric(self):
        """Initialize faithfulness metric with appropriate modality."""
        self.faithfulness_metric = FaithfulnessMetric(
            config=self.config.faithfulness_config,
            modality=self.modality
        )
        self._update_progress("Faithfulness metric initialized")
    
    def evaluate_explainer(
        self, 
        explainer_name: str, 
        explainer: ExplainerWrapper,
        samples: List[DatasetSample]
    ) -> List[FaithfulnessResult]:
        """
        Evaluate a single explainer on the dataset samples.
        
        Args:
            explainer_name: Name of the explainer
            explainer: Explainer instance
            samples: Dataset samples to evaluate
            
        Returns:
            List of faithfulness results for each sample
        """
        self._update_progress(f"Evaluating {explainer_name}...")
        
        results = []
        batch_size = self.config.batch_size
        
        # Create model prediction function
        def model_predict_fn(inputs):
            if isinstance(inputs, list):
                predictions = self.model.predict_batch(inputs)
                # Extract logits or probabilities for faithfulness computation
                if hasattr(predictions[0], 'logits'):
                    return torch.stack([p.logits for p in predictions])
                elif hasattr(predictions[0], 'probabilities'):
                    return torch.stack([p.probabilities for p in predictions])
                else:
                    raise ValueError("Model predictions must have logits or probabilities")
            else:
                prediction = self.model.predict(inputs)
                if hasattr(prediction, 'logits'):
                    return prediction.logits
                elif hasattr(prediction, 'probabilities'):
                    return prediction.probabilities
                else:
                    raise ValueError("Model prediction must have logits or probabilities")
        
        # Process samples in batches
        for i in tqdm(range(0, len(samples), batch_size), desc=f"Evaluating {explainer_name}"):
            batch_samples = samples[i:i + batch_size]
            
            for j, sample in enumerate(batch_samples):
                try:
                    # Generate explanation
                    start_time = time.time()
                    attribution = explainer.explain(
                        model=model_predict_fn,
                        input_data=sample,
                        target_class=sample.label
                    )
                    explanation_time = time.time() - start_time
                    
                    # Compute faithfulness score
                    start_time = time.time()
                    faithfulness_result = self.faithfulness_metric.compute_faithfulness_score(
                        model=model_predict_fn,
                        explainer=lambda x: attribution,  # Pre-computed explanation
                        data=sample,
                        target_class=sample.label
                    )
                    faithfulness_time = time.time() - start_time
                    
                    # Add computation metrics
                    faithfulness_result.computation_metrics.update({
                        'explanation_time': explanation_time,
                        'faithfulness_time': faithfulness_time,
                        'sample_index': i + j,
                        'explainer_name': explainer_name
                    })
                    
                    results.append(faithfulness_result)
                    
                    # Save intermediate results if configured
                    if self.config.save_intermediate and len(results) % 50 == 0:
                        self._save_intermediate_results(explainer_name, results)
                    
                except Exception as e:
                    warnings.warn(f"Failed to evaluate sample {i+j} with {explainer_name}: {str(e)}")
                    continue
        
        self._update_progress(f"Completed {explainer_name}: {len(results)} results")
        return results
    
    def _save_intermediate_results(self, explainer_name: str, results: List[FaithfulnessResult]):
        """Save intermediate results to prevent data loss."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.experiment_name}_{explainer_name}_intermediate_{timestamp}.pkl"
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    def compute_summary_statistics(self, results: Dict[str, List[FaithfulnessResult]]) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for all explainers."""
        summary = {}
        
        for explainer_name, explainer_results in results.items():
            if not explainer_results:
                continue
                
            f_scores = [r.f_score for r in explainer_results]
            p_values = [r.p_value for r in explainer_results]
            computation_times = [r.computation_metrics.get('explanation_time', 0) + 
                               r.computation_metrics.get('faithfulness_time', 0) 
                               for r in explainer_results]
            
            summary[explainer_name] = {
                'mean_f_score': np.mean(f_scores),
                'std_f_score': np.std(f_scores),
                'median_f_score': np.median(f_scores),
                'min_f_score': np.min(f_scores),
                'max_f_score': np.max(f_scores),
                'mean_p_value': np.mean(p_values),
                'significant_results': np.sum([p < 0.05 for p in p_values]),
                'total_results': len(explainer_results),
                'mean_computation_time': np.mean(computation_times),
                'total_computation_time': np.sum(computation_times)
            }
        
        return summary
    
    def run_evaluation(self, explainer_instances: Dict[str, ExplainerWrapper]) -> ExperimentResult:
        """
        Run complete evaluation pipeline.
        
        Args:
            explainer_instances: Dictionary mapping explainer names to instances
            
        Returns:
            Complete experiment results
        """
        start_time = time.time()
        self._update_progress(f"Starting experiment: {self.config.experiment_name}")
        
        # Load components
        self.load_dataset()
        self.load_model()
        self.load_explainers(explainer_instances)
        self.initialize_faithfulness_metric()
        
        # Run evaluation for each explainer
        explainer_results = {}
        computation_metrics = {}
        
        for explainer_name in self.config.explainer_names:
            explainer = self.explainers[explainer_name]
            
            explainer_start_time = time.time()
            results = self.evaluate_explainer(explainer_name, explainer, self.dataset_samples)
            explainer_runtime = time.time() - explainer_start_time
            
            explainer_results[explainer_name] = results
            computation_metrics[explainer_name] = {
                'total_runtime': explainer_runtime,
                'samples_processed': len(results),
                'avg_time_per_sample': explainer_runtime / max(len(results), 1)
            }
        
        # Compute summary statistics
        summary_statistics = self.compute_summary_statistics(explainer_results)
        
        # Create experiment result
        total_runtime = time.time() - start_time
        
        experiment_result = ExperimentResult(
            experiment_config=self.config,
            dataset_info={
                'name': self.config.dataset_name,
                'num_samples': len(self.dataset_samples),
                'modality': str(self.modality)
            },
            model_info={
                'name': self.config.model_name,
                'device': str(self.device),
                'type': type(self.model).__name__
            },
            explainer_results=explainer_results,
            summary_statistics=summary_statistics,
            computation_metrics=computation_metrics,
            timestamp=datetime.now().isoformat(),
            total_runtime=total_runtime
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{self.config.experiment_name}_results_{timestamp}.json"
        result_filepath = os.path.join(self.config.output_dir, result_filename)
        experiment_result.save(result_filepath)
        
        self._update_progress(f"Experiment completed in {total_runtime:.2f}s. Results saved to {result_filepath}")
        
        return experiment_result


def evaluate_explainer(
    model: BaseModelWrapper,
    explainer: ExplainerWrapper,
    data: List[DatasetSample],
    config: Optional[FaithfulnessConfig] = None
) -> List[FaithfulnessResult]:
    """
    Convenience function for evaluating a single explainer.
    
    Args:
        model: Loaded model wrapper
        explainer: Explainer instance
        data: Dataset samples
        config: Faithfulness configuration
        
    Returns:
        List of faithfulness results
    """
    if config is None:
        config = FaithfulnessConfig()
    
    # Determine modality based on data
    modality = DataModality.TEXT  # Default for current implementation
    
    # Initialize faithfulness metric
    faithfulness_metric = FaithfulnessMetric(config=config, modality=modality)
    
    # Create model prediction function
    def model_predict_fn(inputs):
        if isinstance(inputs, list):
            predictions = model.predict_batch(inputs)
            return torch.stack([p.logits if hasattr(p, 'logits') else p.probabilities for p in predictions])
        else:
            prediction = model.predict(inputs)
            return prediction.logits if hasattr(prediction, 'logits') else prediction.probabilities
    
    # Evaluate each sample
    results = []
    for i, sample in enumerate(tqdm(data, desc="Evaluating samples")):
        try:
            # Generate explanation
            attribution = explainer.explain(
                model=model_predict_fn,
                input_data=sample,
                target_class=sample.label
            )
            
            # Compute faithfulness score
            faithfulness_result = faithfulness_metric.compute_faithfulness_score(
                model=model_predict_fn,
                explainer=lambda x: attribution,
                data=sample,
                target_class=sample.label
            )
            
            results.append(faithfulness_result)
            
        except Exception as e:
            warnings.warn(f"Failed to evaluate sample {i}: {str(e)}")
            continue
    
    return results


def create_experiment_config(
    experiment_name: str,
    dataset_name: str,
    model_name: str,
    explainer_names: List[str],
    num_samples: int = 200,
    **kwargs
) -> ExperimentConfig:
    """
    Create experiment configuration with sensible defaults.
    
    Args:
        experiment_name: Name for the experiment
        dataset_name: Dataset to use ("sst2" or "wikitext2")
        model_name: Model to use
        explainer_names: List of explainer names to evaluate
        num_samples: Number of samples to evaluate
        **kwargs: Additional configuration parameters
        
    Returns:
        Experiment configuration
    """
    return ExperimentConfig(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        model_name=model_name,
        explainer_names=explainer_names,
        num_samples=num_samples,
        **kwargs
    )


# Example usage functions
def run_sst2_bert_experiment(explainer_instances: Dict[str, ExplainerWrapper], num_samples: int = 200) -> ExperimentResult:
    """Run SST-2 + BERT experiment with provided explainers."""
    config = create_experiment_config(
        experiment_name="sst2_bert_faithfulness",
        dataset_name="sst2",
        model_name="textattack/bert-base-uncased-SST-2",
        explainer_names=list(explainer_instances.keys()),
        num_samples=num_samples
    )
    
    pipeline = EvaluationPipeline(config)
    return pipeline.run_evaluation(explainer_instances)


def run_wikitext2_gpt2_experiment(explainer_instances: Dict[str, ExplainerWrapper], num_samples: int = 200) -> ExperimentResult:
    """Run WikiText-2 + GPT-2 experiment with provided explainers."""
    config = create_experiment_config(
        experiment_name="wikitext2_gpt2_faithfulness",
        dataset_name="wikitext2", 
        model_name="gpt2",
        explainer_names=list(explainer_instances.keys()),
        num_samples=num_samples
    )
    
    pipeline = EvaluationPipeline(config)
    return pipeline.run_evaluation(explainer_instances)