"""
ROAR (RemOve And Retrain) benchmark implementation for explanation validation.
Implements feature removal and model evaluation to compare with causal-faithfulness metric.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
import warnings
import time
from scipy import stats
from tqdm import tqdm

from .config import get_device
from .datasets import DatasetSample
from .models import BaseModelWrapper, ModelPrediction
from .explainers import ExplainerWrapper, Attribution
from .faithfulness import FaithfulnessResult


@dataclass
class ROARConfig:
    """Configuration for ROAR benchmark."""
    removal_percentages: List[float] = None  # Percentages of features to remove
    n_samples: int = 200  # Number of samples to evaluate
    batch_size: int = 32
    random_seed: int = 42
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        """Set default values."""
        if self.removal_percentages is None:
            self.removal_percentages = [0.1, 0.2, 0.3, 0.4, 0.5]
        if self.device is None:
            self.device = get_device()


@dataclass
class ROARResult:
    """Result from ROAR benchmark evaluation."""
    explainer_name: str
    removal_percentage: float
    original_accuracy: float
    modified_accuracy: float
    accuracy_drop: float
    n_samples: int
    computation_time: float
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation of ROAR result."""
        return (f"ROAR {self.explainer_name}: "
                f"{self.removal_percentage:.1%} removal -> "
                f"{self.accuracy_drop:.4f} accuracy drop")


@dataclass
class ROARCorrelationResult:
    """Result from correlation analysis between ROAR and faithfulness scores."""
    explainer_name: str
    pearson_r: float
    p_value: float
    spearman_rho: float
    spearman_p: float
    n_samples: int
    roar_scores: List[float]
    faithfulness_scores: List[float]
    is_significant: bool
    
    def __str__(self) -> str:
        """String representation of correlation result."""
        return (f"ROAR-Faithfulness Correlation ({self.explainer_name}): "
                f"r={self.pearson_r:.4f} (p={self.p_value:.4f}), "
                f"ρ={self.spearman_rho:.4f} (p={self.spearman_p:.4f})")


class ROARBenchmark:
    """
    ROAR (RemOve And Retrain) benchmark implementation.
    
    Evaluates explanation quality by removing top-k features according to
    explanation importance and measuring the drop in model accuracy.
    """
    
    def __init__(self, config: ROARConfig):
        """Initialize ROAR benchmark with configuration."""
        self.config = config
        self.device = config.device
        self.rng = np.random.RandomState(config.random_seed)
        torch.manual_seed(config.random_seed)
        
    def evaluate_explainer(
        self,
        model: BaseModelWrapper,
        explainer: ExplainerWrapper,
        samples: List[DatasetSample],
        explainer_name: Optional[str] = None
    ) -> List[ROARResult]:
        """
        Evaluate an explainer using ROAR benchmark.
        
        Args:
            model: Model wrapper for evaluation
            explainer: Explainer to evaluate
            samples: Dataset samples for evaluation
            explainer_name: Name of the explainer (for results)
            
        Returns:
            List of ROAR results for different removal percentages
        """
        if explainer_name is None:
            explainer_name = explainer.method_name
        
        print(f"Running ROAR evaluation for {explainer_name}...")
        
        # Limit samples if needed
        if len(samples) > self.config.n_samples:
            samples = samples[:self.config.n_samples]
        
        # Get original model accuracy
        original_accuracy = self._compute_model_accuracy(model, samples)
        print(f"Original model accuracy: {original_accuracy:.4f}")
        
        results = []
        
        # Evaluate for each removal percentage
        for removal_pct in tqdm(self.config.removal_percentages, desc=f"ROAR {explainer_name}"):
            start_time = time.time()
            
            try:
                # Generate explanations and remove features
                modified_samples = self._remove_features_by_explanation(
                    model, explainer, samples, removal_pct
                )
                
                # Compute accuracy on modified samples
                modified_accuracy = self._compute_model_accuracy(model, modified_samples)
                accuracy_drop = original_accuracy - modified_accuracy
                
                computation_time = time.time() - start_time
                
                result = ROARResult(
                    explainer_name=explainer_name,
                    removal_percentage=removal_pct,
                    original_accuracy=original_accuracy,
                    modified_accuracy=modified_accuracy,
                    accuracy_drop=accuracy_drop,
                    n_samples=len(samples),
                    computation_time=computation_time,
                    metadata={
                        'n_modified_samples': len(modified_samples),
                        'removal_strategy': 'top_k_features'
                    }
                )
                
                results.append(result)
                print(f"  {removal_pct:.1%} removal: {accuracy_drop:.4f} drop")
                
            except Exception as e:
                warnings.warn(f"ROAR evaluation failed for {removal_pct:.1%} removal: {e}")
                continue
        
        return results
    
    def _compute_model_accuracy(
        self,
        model: BaseModelWrapper,
        samples: List[DatasetSample]
    ) -> float:
        """Compute model accuracy on given samples."""
        if not samples:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        # Process in batches
        batch_size = self.config.batch_size
        
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            
            try:
                # Get model predictions
                predictions = model.predict_batch(batch_samples)
                
                # Count correct predictions
                for pred, sample in zip(predictions, batch_samples):
                    if sample.label is not None:
                        predicted_class = pred.predicted_class
                        if predicted_class == sample.label:
                            correct_predictions += 1
                        total_predictions += 1
                
            except Exception as e:
                warnings.warn(f"Batch prediction failed: {e}")
                continue
        
        if total_predictions == 0:
            return 0.0
        
        return correct_predictions / total_predictions
    
    def _remove_features_by_explanation(
        self,
        model: BaseModelWrapper,
        explainer: ExplainerWrapper,
        samples: List[DatasetSample],
        removal_percentage: float
    ) -> List[DatasetSample]:
        """
        Remove features based on explanation importance.
        
        Args:
            model: Model wrapper
            explainer: Explainer to use
            samples: Original samples
            removal_percentage: Percentage of features to remove
            
        Returns:
            Modified samples with features removed
        """
        modified_samples = []
        
        # Create model prediction function for explainer
        def model_predict_fn(inputs):
            if isinstance(inputs, list):
                predictions = model.predict_batch(inputs)
                return torch.stack([p.logits for p in predictions])
            else:
                prediction = model.predict(inputs)
                return prediction.logits
        
        for sample in tqdm(samples, desc="Removing features", leave=False):
            try:
                # Generate explanation
                attribution = explainer.explain(
                    model=model_predict_fn,
                    input_data=sample,
                    target_class=sample.label
                )
                
                # Get top features to remove
                top_features = attribution.get_top_features()
                n_features_to_remove = int(len(top_features) * removal_percentage)
                features_to_remove = top_features[:n_features_to_remove]
                
                # Create modified sample by masking top features
                modified_sample = self._mask_features(sample, features_to_remove)
                modified_samples.append(modified_sample)
                
            except Exception as e:
                warnings.warn(f"Feature removal failed for sample: {e}")
                # Use original sample as fallback
                modified_samples.append(sample)
                continue
        
        return modified_samples
    
    def _mask_features(
        self,
        sample: DatasetSample,
        features_to_mask: List[int]
    ) -> DatasetSample:
        """
        Mask specified features in a sample.
        
        Args:
            sample: Original sample
            features_to_mask: Indices of features to mask
            
        Returns:
            Modified sample with features masked
        """
        # Create a copy of the sample
        modified_tokens = sample.tokens.clone()
        modified_attention_mask = sample.attention_mask.clone()
        
        # Mask features (set to padding token or zero)
        for feature_idx in features_to_mask:
            if feature_idx < len(modified_tokens):
                # For text data, we can set to pad token (0) or use a special mask token
                modified_tokens[feature_idx] = 0  # Assuming 0 is pad token
                # Keep attention mask as is to maintain sequence structure
        
        return DatasetSample(
            text=sample.text,  # Keep original text for reference
            tokens=modified_tokens,
            attention_mask=modified_attention_mask,
            label=sample.label,
            metadata={
                **(sample.metadata or {}),
                'roar_modified': True,
                'masked_features': features_to_mask,
                'n_masked_features': len(features_to_mask)
            }
        )
    
    def compute_correlation_with_faithfulness(
        self,
        roar_results: List[ROARResult],
        faithfulness_results: List[FaithfulnessResult],
        explainer_name: str
    ) -> ROARCorrelationResult:
        """
        Compute correlation between ROAR accuracy drops and faithfulness scores.
        
        Args:
            roar_results: ROAR benchmark results
            faithfulness_results: Faithfulness metric results
            explainer_name: Name of the explainer
            
        Returns:
            Correlation analysis result
        """
        if len(roar_results) != len(faithfulness_results):
            warnings.warn(
                f"Mismatched result lengths: ROAR={len(roar_results)}, "
                f"Faithfulness={len(faithfulness_results)}"
            )
            # Take minimum length
            min_len = min(len(roar_results), len(faithfulness_results))
            roar_results = roar_results[:min_len]
            faithfulness_results = faithfulness_results[:min_len]
        
        # Extract scores
        roar_scores = [r.accuracy_drop for r in roar_results]
        faithfulness_scores = [f.f_score for f in faithfulness_results]
        
        # Compute correlations
        try:
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(roar_scores, faithfulness_scores)
            
            # Spearman correlation (rank-based, more robust)
            spearman_rho, spearman_p = stats.spearmanr(roar_scores, faithfulness_scores)
            
            # Check significance (p < 0.05)
            is_significant = pearson_p < 0.05 or spearman_p < 0.05
            
        except Exception as e:
            warnings.warn(f"Correlation computation failed: {e}")
            pearson_r = pearson_p = spearman_rho = spearman_p = 0.0
            is_significant = False
        
        return ROARCorrelationResult(
            explainer_name=explainer_name,
            pearson_r=pearson_r,
            p_value=pearson_p,
            spearman_rho=spearman_rho,
            spearman_p=spearman_p,
            n_samples=len(roar_scores),
            roar_scores=roar_scores,
            faithfulness_scores=faithfulness_scores,
            is_significant=is_significant
        )


class ROARValidator:
    """
    Validator for comparing ROAR and faithfulness metrics.
    Provides statistical analysis and validation functions.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize validator with random seed."""
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    def validate_explainer_ranking(
        self,
        roar_results: Dict[str, List[ROARResult]],
        faithfulness_results: Dict[str, List[FaithfulnessResult]]
    ) -> Dict[str, Any]:
        """
        Validate that explainer rankings are consistent between ROAR and faithfulness.
        
        Args:
            roar_results: ROAR results by explainer name
            faithfulness_results: Faithfulness results by explainer name
            
        Returns:
            Validation results with ranking consistency metrics
        """
        # Compute average scores for each explainer
        roar_avg_scores = {}
        faithfulness_avg_scores = {}
        
        for explainer_name in roar_results.keys():
            if explainer_name in faithfulness_results:
                # Average ROAR accuracy drop
                roar_scores = [r.accuracy_drop for r in roar_results[explainer_name]]
                roar_avg_scores[explainer_name] = np.mean(roar_scores)
                
                # Average faithfulness score
                faith_scores = [f.f_score for f in faithfulness_results[explainer_name]]
                faithfulness_avg_scores[explainer_name] = np.mean(faith_scores)
        
        # Rank explainers by each metric
        roar_ranking = sorted(roar_avg_scores.keys(), 
                             key=lambda x: roar_avg_scores[x], reverse=True)
        faithfulness_ranking = sorted(faithfulness_avg_scores.keys(),
                                    key=lambda x: faithfulness_avg_scores[x], reverse=True)
        
        # Compute ranking correlation
        try:
            # Convert rankings to numerical scores for correlation
            roar_ranks = {name: i for i, name in enumerate(roar_ranking)}
            faith_ranks = {name: i for i, name in enumerate(faithfulness_ranking)}
            
            common_explainers = set(roar_ranks.keys()) & set(faith_ranks.keys())
            
            if len(common_explainers) >= 2:
                roar_rank_values = [roar_ranks[name] for name in common_explainers]
                faith_rank_values = [faith_ranks[name] for name in common_explainers]
                
                rank_correlation, rank_p_value = stats.spearmanr(roar_rank_values, faith_rank_values)
            else:
                rank_correlation = rank_p_value = 0.0
                
        except Exception as e:
            warnings.warn(f"Ranking correlation failed: {e}")
            rank_correlation = rank_p_value = 0.0
        
        return {
            'roar_ranking': roar_ranking,
            'faithfulness_ranking': faithfulness_ranking,
            'roar_avg_scores': roar_avg_scores,
            'faithfulness_avg_scores': faithfulness_avg_scores,
            'rank_correlation': rank_correlation,
            'rank_p_value': rank_p_value,
            'ranking_consistent': rank_correlation > 0.5 and rank_p_value < 0.05,
            'n_common_explainers': len(common_explainers)
        }
    
    def detect_counter_examples(
        self,
        correlation_results: List[ROARCorrelationResult],
        threshold: float = 0.3
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Detect counter-examples where ROAR and faithfulness disagree significantly.
        
        Args:
            correlation_results: Correlation results to analyze
            threshold: Threshold for detecting disagreement
            
        Returns:
            Dictionary of counter-examples by explainer
        """
        counter_examples = {}
        
        for result in correlation_results:
            explainer_counter_examples = []
            
            # Look for points where metrics disagree
            for roar_score, faith_score in zip(result.roar_scores, result.faithfulness_scores):
                # Normalize scores to [0, 1] for comparison
                roar_norm = roar_score  # Already represents accuracy drop
                faith_norm = faith_score  # Already in [0, 1]
                
                # Check for significant disagreement
                disagreement = abs(roar_norm - faith_norm)
                if disagreement > threshold:
                    explainer_counter_examples.append((roar_score, faith_score))
            
            if explainer_counter_examples:
                counter_examples[result.explainer_name] = explainer_counter_examples
        
        return counter_examples
    
    def generate_validation_report(
        self,
        roar_results: Dict[str, List[ROARResult]],
        faithfulness_results: Dict[str, List[FaithfulnessResult]],
        correlation_results: List[ROARCorrelationResult]
    ) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            roar_results: ROAR benchmark results
            faithfulness_results: Faithfulness metric results
            correlation_results: Correlation analysis results
            
        Returns:
            Formatted validation report
        """
        report = "# ROAR-Faithfulness Validation Report\n\n"
        
        # Summary statistics
        report += "## Summary Statistics\n\n"
        
        for result in correlation_results:
            report += f"### {result.explainer_name}\n"
            report += f"- Pearson correlation: r = {result.pearson_r:.4f} (p = {result.p_value:.4f})\n"
            report += f"- Spearman correlation: ρ = {result.spearman_rho:.4f} (p = {result.spearman_p:.4f})\n"
            report += f"- Statistically significant: {'Yes' if result.is_significant else 'No'}\n"
            report += f"- Sample size: {result.n_samples}\n\n"
        
        # Ranking consistency
        ranking_validation = self.validate_explainer_ranking(roar_results, faithfulness_results)
        report += "## Explainer Ranking Consistency\n\n"
        report += f"- ROAR ranking: {ranking_validation['roar_ranking']}\n"
        report += f"- Faithfulness ranking: {ranking_validation['faithfulness_ranking']}\n"
        report += f"- Rank correlation: ρ = {ranking_validation['rank_correlation']:.4f} "
        report += f"(p = {ranking_validation['rank_p_value']:.4f})\n"
        report += f"- Rankings consistent: {'Yes' if ranking_validation['ranking_consistent'] else 'No'}\n\n"
        
        # Counter-examples
        counter_examples = self.detect_counter_examples(correlation_results)
        if counter_examples:
            report += "## Counter-Examples (Significant Disagreements)\n\n"
            for explainer, examples in counter_examples.items():
                report += f"### {explainer}\n"
                report += f"Found {len(examples)} counter-examples:\n"
                for i, (roar_score, faith_score) in enumerate(examples[:5]):  # Show first 5
                    report += f"- Example {i+1}: ROAR = {roar_score:.4f}, Faithfulness = {faith_score:.4f}\n"
                if len(examples) > 5:
                    report += f"- ... and {len(examples) - 5} more\n"
                report += "\n"
        else:
            report += "## Counter-Examples\n\nNo significant counter-examples detected.\n\n"
        
        # Overall assessment
        report += "## Overall Assessment\n\n"
        
        significant_correlations = sum(1 for r in correlation_results if r.is_significant)
        total_correlations = len(correlation_results)
        
        report += f"- {significant_correlations}/{total_correlations} explainers show significant correlation\n"
        report += f"- Ranking consistency: {'Good' if ranking_validation['ranking_consistent'] else 'Poor'}\n"
        
        if significant_correlations >= total_correlations * 0.7:
            report += "- **Conclusion**: Strong agreement between ROAR and faithfulness metrics\n"
        elif significant_correlations >= total_correlations * 0.5:
            report += "- **Conclusion**: Moderate agreement between ROAR and faithfulness metrics\n"
        else:
            report += "- **Conclusion**: Weak agreement between ROAR and faithfulness metrics\n"
        
        return report


# Convenience functions
def run_roar_evaluation(
    model: BaseModelWrapper,
    explainers: Dict[str, ExplainerWrapper],
    samples: List[DatasetSample],
    config: Optional[ROARConfig] = None
) -> Dict[str, List[ROARResult]]:
    """
    Run ROAR evaluation for multiple explainers.
    
    Args:
        model: Model wrapper
        explainers: Dictionary of explainers to evaluate
        samples: Dataset samples
        config: ROAR configuration
        
    Returns:
        ROAR results by explainer name
    """
    if config is None:
        config = ROARConfig()
    
    benchmark = ROARBenchmark(config)
    results = {}
    
    for explainer_name, explainer in explainers.items():
        try:
            explainer_results = benchmark.evaluate_explainer(
                model, explainer, samples, explainer_name
            )
            results[explainer_name] = explainer_results
        except Exception as e:
            warnings.warn(f"ROAR evaluation failed for {explainer_name}: {e}")
            continue
    
    return results


def compute_roar_faithfulness_correlations(
    roar_results: Dict[str, List[ROARResult]],
    faithfulness_results: Dict[str, List[FaithfulnessResult]]
) -> List[ROARCorrelationResult]:
    """
    Compute correlations between ROAR and faithfulness results.
    
    Args:
        roar_results: ROAR results by explainer
        faithfulness_results: Faithfulness results by explainer
        
    Returns:
        List of correlation results
    """
    benchmark = ROARBenchmark(ROARConfig())
    correlation_results = []
    
    for explainer_name in roar_results.keys():
        if explainer_name in faithfulness_results:
            try:
                correlation = benchmark.compute_correlation_with_faithfulness(
                    roar_results[explainer_name],
                    faithfulness_results[explainer_name],
                    explainer_name
                )
                correlation_results.append(correlation)
            except Exception as e:
                warnings.warn(f"Correlation computation failed for {explainer_name}: {e}")
                continue
    
    return correlation_results