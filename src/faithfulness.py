"""
Core faithfulness metric computation framework.
Implements the F(E) formula with Monte-Carlo sampling and statistical analysis.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable, Union, Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from scipy import stats
import warnings

if TYPE_CHECKING:
    from .explainers import Attribution
from .config import get_device, get_batch_size, adjust_batch_size_for_oom, memory_efficient_context, safe_operation
from .robust_computation import (
    safe_divide, stabilize_tensor, execute_with_retries, 
    stream_batches, memory_monitor, ComputationLimits
)
from .masking import FeatureMasker, DataModality, MaskingStrategy
from .baseline import BaselineGenerator, BaselineStrategy


@dataclass
class FaithfulnessConfig:
    """Configuration for faithfulness metric computation."""
    n_samples: int = 1000  # Monte-Carlo samples
    baseline_strategy: str = "random"  # "random", "mean", "zero"
    masking_strategy: str = "pad"  # modality-specific
    confidence_level: float = 0.95
    batch_size: int = 32
    random_seed: int = 42
    device: Optional[torch.device] = None
    numerical_epsilon: float = 1e-8
    computation_limits: Optional[ComputationLimits] = None
    enable_streaming: bool = False  # Enable streaming for large datasets
    
    def __post_init__(self):
        """Set default device and computation limits if not provided."""
        if self.device is None:
            self.device = get_device()
        if self.batch_size is None:
            self.batch_size = get_batch_size()
        if self.computation_limits is None:
            self.computation_limits = ComputationLimits(numerical_epsilon=self.numerical_epsilon)


@dataclass
class FaithfulnessResult:
    """Result of faithfulness metric computation."""
    f_score: float
    confidence_interval: Tuple[float, float]
    n_samples: int
    baseline_performance: float
    explained_performance: float
    statistical_significance: bool
    p_value: float
    computation_metrics: Dict[str, float]
    
    def __str__(self) -> str:
        """String representation of results."""
        return (f"F-score: {self.f_score:.4f} "
                f"[{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}] "
                f"(p={self.p_value:.4f})")


class FaithfulnessMetric:
    """
    Core implementation of the causal-faithfulness metric F(E).
    
    The metric is defined as:
    F(E) = 1 - E[|f(x) - f(x_∖E)|] / E[|f(x) - f(x_rand)|]
    
    Where:
    - x is the original input
    - x_∖E is x with non-explained features masked
    - x_rand is a random baseline
    - f is the model prediction function
    """
    
    def __init__(self, config: FaithfulnessConfig, modality: DataModality = DataModality.TABULAR):
        """Initialize the faithfulness metric with configuration."""
        self.config = config
        self.modality = modality
        self.rng = np.random.RandomState(config.random_seed)
        torch.manual_seed(config.random_seed)
        
        # Initialize feature masker
        self.masker = FeatureMasker(
            modality=modality,
            strategy=MaskingStrategy(config.masking_strategy),
            random_seed=config.random_seed
        )
        
        # Initialize baseline generator
        self.baseline_generator = BaselineGenerator(
            modality=modality,
            strategy=BaselineStrategy(config.baseline_strategy),
            random_seed=config.random_seed
        )
        
    def compute_faithfulness_score(
        self,
        model: Callable,
        explainer: Callable,
        data: Union[torch.Tensor, Dict, np.ndarray],
        target_class: Optional[int] = None
    ) -> FaithfulnessResult:
        """
        Compute the faithfulness score for an explanation method.
        
        Args:
            model: Model prediction function f(x) -> predictions
            explainer: Explanation function that returns feature attributions
            data: Input data (single instance or batch)
            target_class: Target class for explanation (if None, use predicted class)
            
        Returns:
            FaithfulnessResult with F-score and statistical analysis
        """
        # Use memory-efficient context for computation
        with memory_efficient_context():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            # Get original prediction with hardware optimization
            original_pred = self._get_model_prediction(model, data)
            if target_class is None:
                target_class = torch.argmax(original_pred, dim=-1).item()
            
            # Generate explanation with safe operation
            attribution = safe_operation(explainer, model, data)
            explained_features = self._get_top_features(attribution)
            
            # Compute Monte-Carlo samples with robust computation
            if self.config.enable_streaming:
                # Use streaming computation for large datasets
                explained_diffs, baseline_diffs = self._compute_streaming_differences(
                    model, data, explained_features, original_pred, target_class
                )
            else:
                # Use batch computation with adaptive sizing
                explained_diffs = []
                baseline_diffs = []
                current_batch_size = self.config.batch_size
                
                i = 0
                while i < self.config.n_samples:
                    batch_size = min(current_batch_size, self.config.n_samples - i)
                    
                    try:
                        with memory_monitor():
                            # Compute explained feature differences
                            explained_batch = execute_with_retries(
                                self._compute_explained_differences,
                                model, data, explained_features, original_pred, 
                                target_class, batch_size
                            )
                            explained_diffs.extend(explained_batch)
                            
                            # Compute baseline differences
                            baseline_batch = execute_with_retries(
                                self._compute_baseline_differences,
                                model, data, original_pred, target_class, batch_size
                            )
                            baseline_diffs.extend(baseline_batch)
                            
                            i += batch_size
                            
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                            # Reduce batch size and retry
                            current_batch_size = adjust_batch_size_for_oom(current_batch_size)
                            if current_batch_size < 1:
                                raise RuntimeError("Cannot reduce batch size further, computation failed")
                            continue
                        else:
                            raise e
        
        # Convert to numpy arrays
        explained_diffs = np.array(explained_diffs)
        baseline_diffs = np.array(baseline_diffs)
        
        # Compute F-score
        mean_explained = np.mean(explained_diffs)
        mean_baseline = np.mean(baseline_diffs)
        
        # Use safe division to avoid numerical issues
        f_score = 1.0 - safe_divide(mean_explained, mean_baseline)
        
        # Ensure F-score is in [0, 1] bounds
        f_score = np.clip(f_score, 0.0, 1.0)
        
        # Statistical analysis
        confidence_interval = self._compute_confidence_interval(
            explained_diffs, baseline_diffs
        )
        
        # Significance test (paired t-test)
        t_stat, p_value = stats.ttest_rel(baseline_diffs, explained_diffs)
        is_significant = bool(p_value < (1 - self.config.confidence_level))
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            computation_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            computation_time = 0.0
        
        # Prepare computation metrics
        computation_metrics = {
            'computation_time_seconds': computation_time,
            'n_model_queries': self.config.n_samples * 2,  # explained + baseline
            'mean_explained_diff': mean_explained,
            'mean_baseline_diff': mean_baseline,
            'std_explained_diff': np.std(explained_diffs),
            'std_baseline_diff': np.std(baseline_diffs)
        }
        
        return FaithfulnessResult(
            f_score=f_score,
            confidence_interval=confidence_interval,
            n_samples=self.config.n_samples,
            baseline_performance=mean_baseline,
            explained_performance=mean_explained,
            statistical_significance=is_significant,
            p_value=p_value,
            computation_metrics=computation_metrics
        )
    
    def _get_model_prediction(
        self, 
        model: Callable, 
        data: Union[torch.Tensor, Dict, np.ndarray]
    ) -> torch.Tensor:
        """Get model prediction for input data with hardware optimization."""
        with torch.no_grad():
            # Convert data to tensor and move to optimal device
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            elif isinstance(data, torch.Tensor):
                data = data.float()
            
            # Move to device with safe operation
            try:
                data = data.to(self.config.device)
            except Exception as e:
                warnings.warn(f"Failed to move data to {self.config.device}, using CPU: {e}")
                data = data.to('cpu')
                self.config.device = torch.device('cpu')
            
            # Get prediction with safe operation and stabilization
            def _model_forward(model_func, input_data):
                pred = model_func(input_data)
                if isinstance(pred, tuple):
                    pred = pred[0]  # Handle models that return (logits, hidden_states)
                # Stabilize prediction tensor
                if isinstance(pred, torch.Tensor):
                    pred = stabilize_tensor(pred)
                return pred
            
            pred = execute_with_retries(_model_forward, model, data)
            return pred
    
    def _get_top_features(self, attribution: Union[np.ndarray, torch.Tensor, "Attribution"]) -> List[int]:
        """Extract top features from attribution scores."""
        # Handle Attribution object
        if hasattr(attribution, 'feature_scores'):
            attribution = attribution.feature_scores
        
        if isinstance(attribution, torch.Tensor):
            attribution = attribution.cpu().numpy()
        
        # Get absolute importance scores
        abs_scores = np.abs(attribution)
        
        # Return indices sorted by importance (descending)
        return np.argsort(abs_scores)[::-1].tolist()
    
    def _compute_explained_differences(
        self,
        model: Callable,
        data: Union[torch.Tensor, Dict, np.ndarray],
        explained_features: List[int],
        original_pred: torch.Tensor,
        target_class: int,
        batch_size: int
    ) -> List[float]:
        """Compute differences when explained features are masked."""
        differences = []
        
        for _ in range(batch_size):
            # Create masked version (x_∖E) - this will be implemented by FeatureMasker
            # For now, we'll use a placeholder that masks random features
            masked_data = self._mask_features(data, explained_features, mask_explained=False)
            
            # Get prediction for masked data
            masked_pred = self._get_model_prediction(model, masked_data)
            
            # Compute absolute difference for target class
            if len(original_pred.shape) > 1 and original_pred.shape[0] > 1:
                # Batch dimension exists
                diff = torch.abs(
                    original_pred[0, target_class] - masked_pred[0, target_class]
                ).item()
            else:
                # Single sample or flattened
                diff = torch.abs(
                    original_pred.flatten()[target_class] - masked_pred.flatten()[target_class]
                ).item()
            
            differences.append(diff)
        
        return differences
    
    def _compute_baseline_differences(
        self,
        model: Callable,
        data: Union[torch.Tensor, Dict, np.ndarray],
        original_pred: torch.Tensor,
        target_class: int,
        batch_size: int
    ) -> List[float]:
        """Compute differences for random baseline (x_rand)."""
        differences = []
        
        for _ in range(batch_size):
            # Create random baseline - this will be implemented by BaselineGenerator
            # For now, we'll use a placeholder
            baseline_data = self._generate_baseline(data)
            
            # Get prediction for baseline data
            baseline_pred = self._get_model_prediction(model, baseline_data)
            
            # Compute absolute difference for target class
            if len(original_pred.shape) > 1 and original_pred.shape[0] > 1:
                # Batch dimension exists
                diff = torch.abs(
                    original_pred[0, target_class] - baseline_pred[0, target_class]
                ).item()
            else:
                # Single sample or flattened
                diff = torch.abs(
                    original_pred.flatten()[target_class] - baseline_pred.flatten()[target_class]
                ).item()
            
            differences.append(diff)
        
        return differences
    
    def _mask_features(
        self,
        data: Union[torch.Tensor, Dict, np.ndarray],
        features_to_keep: List[int],
        mask_explained: bool = False
    ) -> Union[torch.Tensor, Dict, np.ndarray]:
        """
        Mask features using the configured FeatureMasker.
        
        Args:
            data: Input data
            features_to_keep: Features to keep (if mask_explained=False) or mask (if True)
            mask_explained: Whether to mask the explained features or keep them
        """
        try:
            return self.masker.mask_features(data, features_to_keep, mask_explained)
        except Exception as e:
            warnings.warn(f"Feature masking failed: {e}. Using fallback masking.")
            # Fallback to simple masking for compatibility
            if isinstance(data, torch.Tensor):
                masked_data = data.clone()
                if len(masked_data.shape) > 1:
                    if mask_explained:
                        # Mask the explained features
                        masked_data[..., features_to_keep] = 0
                    else:
                        # Mask all features except explained ones
                        all_features = list(range(masked_data.shape[-1]))
                        features_to_mask = [f for f in all_features if f not in features_to_keep]
                        if features_to_mask:
                            masked_data[..., features_to_mask] = 0
                return masked_data
            else:
                return data
    
    def _generate_baseline(
        self, 
        data: Union[torch.Tensor, Dict, np.ndarray]
    ) -> Union[torch.Tensor, Dict, np.ndarray]:
        """
        Generate baseline data using the configured BaselineGenerator.
        """
        try:
            return self.baseline_generator.generate_baseline(data, batch_size=1)
        except Exception as e:
            warnings.warn(f"Baseline generation failed: {e}. Using fallback baseline.")
            # Fallback to simple baseline generation
            if isinstance(data, torch.Tensor):
                if self.config.baseline_strategy == "random":
                    return torch.randn_like(data)
                elif self.config.baseline_strategy == "zero":
                    return torch.zeros_like(data)
                elif self.config.baseline_strategy == "mean":
                    return torch.mean(data, dim=0, keepdim=True).expand_as(data)
                else:
                    return torch.randn_like(data)
            else:
                return data
    
    def _compute_confidence_interval(
        self,
        explained_diffs: np.ndarray,
        baseline_diffs: np.ndarray
    ) -> Tuple[float, float]:
        """Compute confidence interval for F-score using bootstrap."""
        n_bootstrap = 1000
        bootstrap_f_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = self.rng.choice(len(explained_diffs), size=len(explained_diffs), replace=True)
            
            boot_explained = explained_diffs[indices]
            boot_baseline = baseline_diffs[indices]
            
            # Compute F-score for bootstrap sample
            mean_explained = np.mean(boot_explained)
            mean_baseline = np.mean(boot_baseline)
            
            if mean_baseline < self.config.numerical_epsilon:
                boot_f_score = 0.0
            else:
                boot_f_score = 1.0 - (mean_explained / mean_baseline)
            
            boot_f_score = np.clip(boot_f_score, 0.0, 1.0)
            bootstrap_f_scores.append(boot_f_score)
        
        # Compute confidence interval
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_f_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_f_scores, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def _compute_streaming_differences(
        self,
        model: Callable,
        data: Union[torch.Tensor, Dict, np.ndarray],
        explained_features: List[int],
        original_pred: torch.Tensor,
        target_class: int
    ) -> Tuple[List[float], List[float]]:
        """
        Compute differences using streaming computation for large datasets.
        
        Args:
            model: Model prediction function
            data: Input data
            explained_features: Features identified by explainer
            original_pred: Original model prediction
            target_class: Target class index
            
        Returns:
            Tuple of (explained_diffs, baseline_diffs)
        """
        explained_diffs = []
        baseline_diffs = []
        
        # Create sample indices for streaming
        sample_indices = list(range(self.config.n_samples))
        
        # Stream through batches
        for batch_indices in stream_batches(sample_indices, self.config.batch_size):
            batch_size = len(batch_indices)
            
            with memory_monitor():
                # Compute explained differences for this batch
                explained_batch = execute_with_retries(
                    self._compute_explained_differences,
                    model, data, explained_features, original_pred, 
                    target_class, batch_size
                )
                
                # Stabilize results
                explained_batch = [stabilize_tensor(torch.tensor(x)).item() if isinstance(x, (int, float)) else x 
                                 for x in explained_batch]
                explained_diffs.extend(explained_batch)
                
                # Compute baseline differences for this batch
                baseline_batch = execute_with_retries(
                    self._compute_baseline_differences,
                    model, data, original_pred, target_class, batch_size
                )
                
                # Stabilize results
                baseline_batch = [stabilize_tensor(torch.tensor(x)).item() if isinstance(x, (int, float)) else x 
                                for x in baseline_batch]
                baseline_diffs.extend(baseline_batch)
        
        return explained_diffs, baseline_diffs


def compute_faithfulness_score(
    model: Callable,
    explainer: Callable,
    data: Union[torch.Tensor, Dict, np.ndarray],
    config: Optional[FaithfulnessConfig] = None,
    target_class: Optional[int] = None
) -> FaithfulnessResult:
    """
    Convenience function to compute faithfulness score.
    
    Args:
        model: Model prediction function
        explainer: Explanation method
        data: Input data
        config: Configuration (uses default if None)
        target_class: Target class for explanation
        
    Returns:
        FaithfulnessResult with F-score and analysis
    """
    if config is None:
        config = FaithfulnessConfig()
    
    metric = FaithfulnessMetric(config)
    return metric.compute_faithfulness_score(model, explainer, data, target_class)