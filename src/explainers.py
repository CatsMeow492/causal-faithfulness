"""
Explanation method wrappers for unified interface.
Supports SHAP, Integrated Gradients, LIME, and random baseline explainers.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union, Dict, List, Optional, Any, Tuple
import warnings
import time
from .config import get_device


@dataclass
class Attribution:
    """Unified attribution result from explanation methods."""
    feature_scores: np.ndarray
    feature_indices: List[int]
    method_name: str
    computation_time: float
    metadata: Optional[Dict[str, Any]] = None
    
    def get_top_features(self, k: Optional[int] = None) -> List[int]:
        """Get top k features by absolute importance."""
        abs_scores = np.abs(self.feature_scores)
        sorted_indices = np.argsort(abs_scores)[::-1]
        
        if k is not None:
            return sorted_indices[:k].tolist()
        return sorted_indices.tolist()
    
    def get_feature_importance(self, feature_idx: int) -> float:
        """Get importance score for a specific feature."""
        if feature_idx < len(self.feature_scores):
            return self.feature_scores[feature_idx]
        return 0.0


class ExplainerWrapper(ABC):
    """Abstract base class for explanation method wrappers."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize explainer with random seed for reproducibility."""
        self.random_seed = random_seed
        self.method_name = self.__class__.__name__
        
    @abstractmethod
    def explain(
        self, 
        model: Callable, 
        input_data: Union[torch.Tensor, np.ndarray, Dict],
        target_class: Optional[int] = None,
        **kwargs
    ) -> Attribution:
        """
        Generate explanation for input data.
        
        Args:
            model: Model prediction function
            input_data: Input to explain
            target_class: Target class for explanation (if None, use predicted class)
            **kwargs: Additional method-specific parameters
            
        Returns:
            Attribution object with feature importance scores
        """
        pass
    
    def _get_model_prediction(
        self, 
        model: Callable, 
        data: Union[torch.Tensor, np.ndarray, Dict]
    ) -> torch.Tensor:
        """Get model prediction, handling different input types."""
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            elif isinstance(data, torch.Tensor):
                data = data.float()
            
            # Move to appropriate device
            device = get_device()
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            
            pred = model(data)
            if isinstance(pred, tuple):
                pred = pred[0]  # Handle models that return (logits, hidden_states)
            
            return pred
    
    def _prepare_input(
        self, 
        input_data: Union[torch.Tensor, np.ndarray, Dict]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Prepare input data for explanation method."""
        if isinstance(input_data, torch.Tensor):
            return input_data.cpu().numpy()
        elif isinstance(input_data, np.ndarray):
            return input_data
        elif isinstance(input_data, dict):
            # Handle dictionary inputs (e.g., for transformers)
            if 'input_ids' in input_data:
                return input_data['input_ids'].cpu().numpy() if isinstance(input_data['input_ids'], torch.Tensor) else input_data['input_ids']
            else:
                raise ValueError("Dictionary input must contain 'input_ids' key")
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")


class SHAPWrapper(ExplainerWrapper):
    """
    Wrapper for SHAP explanation methods.
    Supports KernelSHAP, TreeSHAP, and DeepSHAP.
    """
    
    def __init__(
        self, 
        explainer_type: str = "kernel",
        n_samples: int = 1000,
        random_seed: int = 42,
        **shap_kwargs
    ):
        """
        Initialize SHAP wrapper.
        
        Args:
            explainer_type: Type of SHAP explainer ("kernel", "tree", "deep")
            n_samples: Number of samples for KernelSHAP
            random_seed: Random seed for reproducibility
            **shap_kwargs: Additional arguments for SHAP explainer
        """
        super().__init__(random_seed)
        self.explainer_type = explainer_type.lower()
        self.n_samples = n_samples
        self.shap_kwargs = shap_kwargs
        self.method_name = f"SHAP_{explainer_type.upper()}"
        
        # Import SHAP with error handling
        try:
            import shap
            self.shap = shap
        except ImportError:
            raise ImportError(
                "SHAP is required for SHAPWrapper. Install with: pip install shap"
            )
        
        self.explainer = None
        self._model_wrapper = None
    
    def _create_model_wrapper(self, model: Callable) -> Callable:
        """Create a model wrapper compatible with SHAP."""
        def model_wrapper(x):
            """Wrapper function for SHAP compatibility."""
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            
            with torch.no_grad():
                pred = model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                # Return numpy array for SHAP
                if isinstance(pred, torch.Tensor):
                    return pred.cpu().numpy()
                return pred
        
        return model_wrapper
    
    def _initialize_explainer(
        self, 
        model: Callable, 
        background_data: Optional[Union[torch.Tensor, np.ndarray]] = None
    ):
        """Initialize the appropriate SHAP explainer."""
        model_wrapper = self._create_model_wrapper(model)
        
        if self.explainer_type == "kernel":
            if background_data is None:
                raise ValueError("Background data is required for KernelSHAP")
            
            background = self._prepare_input(background_data)
            self.explainer = self.shap.KernelExplainer(
                model_wrapper, 
                background,
                **self.shap_kwargs
            )
            
        elif self.explainer_type == "deep":
            if background_data is None:
                raise ValueError("Background data is required for DeepSHAP")
            
            background = background_data
            if isinstance(background, np.ndarray):
                background = torch.from_numpy(background).float()
            
            self.explainer = self.shap.DeepExplainer(
                model, 
                background,
                **self.shap_kwargs
            )
            
        elif self.explainer_type == "tree":
            # TreeSHAP doesn't need background data
            self.explainer = self.shap.TreeExplainer(
                model,
                **self.shap_kwargs
            )
            
        else:
            raise ValueError(f"Unsupported SHAP explainer type: {self.explainer_type}")
    
    def explain(
        self, 
        model: Callable, 
        input_data: Union[torch.Tensor, np.ndarray, Dict],
        target_class: Optional[int] = None,
        background_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs
    ) -> Attribution:
        """
        Generate SHAP explanation for input data.
        
        Args:
            model: Model prediction function
            input_data: Input to explain
            target_class: Target class for explanation
            background_data: Background data for SHAP (required for some explainers)
            **kwargs: Additional SHAP parameters
            
        Returns:
            Attribution object with SHAP values
        """
        start_time = time.time()
        
        try:
            # Initialize explainer if not already done
            if self.explainer is None:
                self._initialize_explainer(model, background_data)
            
            # Prepare input data
            input_array = self._prepare_input(input_data)
            
            # Handle single instance vs batch
            if len(input_array.shape) == 1:
                input_array = input_array.reshape(1, -1)
            
            # Get target class if not provided
            if target_class is None:
                pred = self._get_model_prediction(model, input_data)
                target_class = torch.argmax(pred, dim=-1).item()
            
            # Compute SHAP values
            if self.explainer_type == "kernel":
                shap_values = self.explainer.shap_values(
                    input_array, 
                    nsamples=self.n_samples,
                    **kwargs
                )
            else:
                shap_values = self.explainer.shap_values(input_array, **kwargs)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # Multi-class case - select target class
                if target_class < len(shap_values):
                    feature_scores = shap_values[target_class][0]  # First instance
                else:
                    warnings.warn(f"Target class {target_class} not found, using class 0")
                    feature_scores = shap_values[0][0]
            else:
                # Single output case
                feature_scores = shap_values[0]  # First instance
            
            # Ensure feature_scores is 1D
            if len(feature_scores.shape) > 1:
                feature_scores = feature_scores.flatten()
            
            # Create feature indices
            feature_indices = list(range(len(feature_scores)))
            
            computation_time = time.time() - start_time
            
            return Attribution(
                feature_scores=feature_scores,
                feature_indices=feature_indices,
                method_name=self.method_name,
                computation_time=computation_time,
                metadata={
                    'target_class': target_class,
                    'explainer_type': self.explainer_type,
                    'n_samples': self.n_samples if self.explainer_type == "kernel" else None,
                    'input_shape': input_array.shape
                }
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            warnings.warn(f"SHAP explanation failed: {str(e)}")
            
            # Return zero attribution as fallback
            input_array = self._prepare_input(input_data)
            if len(input_array.shape) == 1:
                n_features = len(input_array)
            else:
                n_features = input_array.shape[-1]
            
            return Attribution(
                feature_scores=np.zeros(n_features),
                feature_indices=list(range(n_features)),
                method_name=f"{self.method_name}_FAILED",
                computation_time=computation_time,
                metadata={'error': str(e)}
            )


class IntegratedGradientsWrapper(ExplainerWrapper):
    """
    Wrapper for Integrated Gradients explanation method.
    Handles gradient computation with different baseline strategies.
    """
    
    def __init__(
        self,
        n_steps: int = 50,
        baseline_strategy: str = "zero",
        random_seed: int = 42,
        internal_batch_size: int = 32
    ):
        """
        Initialize Integrated Gradients wrapper.
        
        Args:
            n_steps: Number of integration steps
            baseline_strategy: Baseline strategy ("zero", "random", "mean")
            random_seed: Random seed for reproducibility
            internal_batch_size: Batch size for gradient computation
        """
        super().__init__(random_seed)
        self.n_steps = n_steps
        self.baseline_strategy = baseline_strategy
        self.internal_batch_size = internal_batch_size
        self.method_name = "IntegratedGradients"
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
    def _create_baseline(
        self, 
        input_data: torch.Tensor,
        baseline_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Create baseline for Integrated Gradients."""
        if baseline_data is not None:
            return baseline_data
        
        if self.baseline_strategy == "zero":
            return torch.zeros_like(input_data)
        elif self.baseline_strategy == "random":
            return torch.randn_like(input_data)
        elif self.baseline_strategy == "mean":
            # Use mean across all dimensions except batch
            if len(input_data.shape) > 1:
                mean_vals = torch.mean(input_data, dim=tuple(range(1, len(input_data.shape))), keepdim=True)
                return mean_vals.expand_as(input_data)
            else:
                return torch.mean(input_data) * torch.ones_like(input_data)
        else:
            warnings.warn(f"Unknown baseline strategy: {self.baseline_strategy}, using zero")
            return torch.zeros_like(input_data)
    
    def _compute_gradients(
        self,
        model: Callable,
        input_data: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """Compute gradients of model output w.r.t. input."""
        input_data = input_data.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = model(input_data)
        if isinstance(output, tuple):
            output = output[0]
        
        # Get target class output
        if len(output.shape) > 1 and output.shape[1] > 1:
            # Multi-class case
            target_output = output[:, target_class]
        else:
            # Single output case
            target_output = output.squeeze()
        
        # Backward pass
        if target_output.dim() == 0:
            target_output.backward()
        else:
            target_output.sum().backward()
        
        return input_data.grad
    
    def _integrated_gradients_step(
        self,
        model: Callable,
        baseline: torch.Tensor,
        input_data: torch.Tensor,
        target_class: int,
        alpha: float
    ) -> torch.Tensor:
        """Compute gradients at interpolated point."""
        # Interpolate between baseline and input
        interpolated = baseline + alpha * (input_data - baseline)
        
        # Compute gradients
        gradients = self._compute_gradients(model, interpolated, target_class)
        
        return gradients
    
    def explain(
        self,
        model: Callable,
        input_data: Union[torch.Tensor, np.ndarray, Dict],
        target_class: Optional[int] = None,
        baseline_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs
    ) -> Attribution:
        """
        Generate Integrated Gradients explanation.
        
        Args:
            model: Model prediction function
            input_data: Input to explain
            target_class: Target class for explanation
            baseline_data: Custom baseline (if None, uses baseline_strategy)
            **kwargs: Additional parameters
            
        Returns:
            Attribution object with integrated gradients
        """
        start_time = time.time()
        
        try:
            # Convert input to tensor
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).float()
            elif isinstance(input_data, torch.Tensor):
                input_tensor = input_data.float()
            elif isinstance(input_data, dict):
                # Handle dictionary inputs
                if 'input_ids' in input_data:
                    input_tensor = input_data['input_ids']
                    if isinstance(input_tensor, np.ndarray):
                        input_tensor = torch.from_numpy(input_tensor).float()
                else:
                    raise ValueError("Dictionary input must contain 'input_ids' key")
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            # Ensure batch dimension
            if len(input_tensor.shape) == 1:
                input_tensor = input_tensor.unsqueeze(0)
            
            # Move to appropriate device
            device = get_device()
            input_tensor = input_tensor.to(device)

            # Embedding-level path for transformer models
            use_embeddings: bool = bool(kwargs.get('use_embeddings', False))
            hf_model = kwargs.get('hf_model', None)
            attention_mask = kwargs.get('attention_mask', None)

            if use_embeddings and hf_model is not None:
                # Expect integer token ids
                token_ids = input_tensor.long()
                # Ensure we use the HF model's device to avoid cross-device autograd issues
                try:
                    model_device = next(hf_model.parameters()).device
                except Exception:
                    model_device = device
                device = model_device
                if attention_mask is None:
                    attention_mask = torch.ones_like(token_ids, device=device)
                else:
                    attention_mask = attention_mask.to(device)

                embedding_layer = hf_model.get_input_embeddings()
                with torch.no_grad():
                    input_embeds = embedding_layer(token_ids.to(device))  # (B, L, H)

                # Baseline in embedding space
                if baseline_data is not None:
                    baseline = baseline_data.to(device)
                    if baseline.shape != input_embeds.shape:
                        baseline = torch.zeros_like(input_embeds)
                else:
                    baseline = torch.zeros_like(input_embeds)

                # Target class
                if target_class is None:
                    with torch.no_grad():
                        out = hf_model(inputs_embeds=input_embeds, attention_mask=attention_mask)
                        logits = out.logits
                        # If sequence output (LM), use last position
                        if logits.dim() == 3:
                            logits_vec = logits[:, -1, :]
                        else:
                            logits_vec = logits
                        target_class = int(torch.argmax(logits_vec, dim=-1).item())

                # Integrated gradients over embeddings
                alphas = torch.linspace(0.0, 1.0, self.n_steps + 1, device=device)
                total_grad = torch.zeros_like(input_embeds)

                for i in range(0, len(alphas), max(1, self.internal_batch_size)):
                    batch_alphas = alphas[i:i + self.internal_batch_size]
                    grads = []
                    for alpha in batch_alphas:
                        interp = (baseline + alpha * (input_embeds - baseline)).detach()
                        interp.requires_grad_(True)
                        out = hf_model(inputs_embeds=interp, attention_mask=attention_mask)
                        logits = out.logits
                        if logits.dim() == 3:
                            logits_vec = logits[:, -1, :]
                        else:
                            logits_vec = logits
                        target_output = logits_vec[:, target_class].sum()
                        hf_model.zero_grad(set_to_none=True)
                        if interp.grad is not None:
                            interp.grad.zero_()
                        target_output.backward()
                        grads.append(interp.grad.detach())
                    if grads:
                        total_grad = total_grad + torch.stack(grads, dim=0).mean(dim=0)

                avg_grad = total_grad / len(alphas)
                attributions = avg_grad * (input_embeds - baseline)
                # Aggregate over hidden dimension to get per-token attribution
                token_attrib = attributions.sum(dim=-1).squeeze(0)  # (L,)
                feature_scores = token_attrib.detach().cpu().numpy()
                feature_indices = list(range(feature_scores.shape[0]))

                computation_time = time.time() - start_time
                return Attribution(
                    feature_scores=feature_scores,
                    feature_indices=feature_indices,
                    method_name=f"{self.method_name}_EMBED",
                    computation_time=computation_time,
                    metadata={
                        'target_class': target_class,
                        'n_steps': self.n_steps,
                        'baseline_strategy': self.baseline_strategy,
                        'input_shape': input_tensor.shape,
                        'device': str(device),
                        'embedding_dim': int(attributions.shape[-1])
                    }
                )

            # Default token-id path (tabular/continuous features)
            # Get target class if not provided
            if target_class is None:
                with torch.no_grad():
                    pred = model(input_tensor)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    target_class = torch.argmax(pred, dim=-1).item()

            # Create baseline
            if baseline_data is not None:
                if isinstance(baseline_data, np.ndarray):
                    baseline = torch.from_numpy(baseline_data).float().to(device)
                else:
                    baseline = baseline_data.float().to(device)
                if len(baseline.shape) == 1:
                    baseline = baseline.unsqueeze(0)
            else:
                baseline = self._create_baseline(input_tensor)

            if baseline.shape != input_tensor.shape:
                baseline = baseline.expand_as(input_tensor)

            integrated_gradients = torch.zeros_like(input_tensor)
            alphas = torch.linspace(0, 1, self.n_steps + 1, device=device)
            for i in range(0, len(alphas), self.internal_batch_size):
                batch_alphas = alphas[i:i + self.internal_batch_size]
                batch_gradients = []
                for alpha in batch_alphas:
                    try:
                        grad = self._integrated_gradients_step(
                            model, baseline, input_tensor, target_class, alpha.item()
                        )
                        if grad is not None:
                            batch_gradients.append(grad)
                    except Exception as e:
                        warnings.warn(f"Gradient computation failed at alpha={alpha.item()}: {e}")
                        batch_gradients.append(torch.zeros_like(input_tensor))
                if batch_gradients:
                    batch_avg = torch.stack(batch_gradients).mean(dim=0)
                    integrated_gradients += batch_avg

            integrated_gradients = integrated_gradients / len(alphas)
            attributions = integrated_gradients * (input_tensor - baseline)
            feature_scores = attributions.squeeze().cpu().detach().numpy()
            if len(feature_scores.shape) > 1:
                feature_scores = feature_scores.flatten()
            feature_indices = list(range(len(feature_scores)))

            computation_time = time.time() - start_time

            return Attribution(
                feature_scores=feature_scores,
                feature_indices=feature_indices,
                method_name=self.method_name,
                computation_time=computation_time,
                metadata={
                    'target_class': target_class,
                    'n_steps': self.n_steps,
                    'baseline_strategy': self.baseline_strategy,
                    'input_shape': input_tensor.shape,
                    'device': str(device)
                }
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            warnings.warn(f"Integrated Gradients explanation failed: {str(e)}")
            
            # Return zero attribution as fallback
            try:
                input_array = self._prepare_input(input_data)
                if len(input_array.shape) == 1:
                    n_features = len(input_array)
                else:
                    n_features = input_array.shape[-1]
            except:
                n_features = 1  # Fallback
            
            return Attribution(
                feature_scores=np.zeros(n_features),
                feature_indices=list(range(n_features)),
                method_name=f"{self.method_name}_FAILED",
                computation_time=computation_time,
                metadata={'error': str(e)}
            )


class LIMEWrapper(ExplainerWrapper):
    """
    Wrapper for LIME (Local Interpretable Model-agnostic Explanations).
    Supports different data modalities with perturbation-based explanations.
    """
    
    def __init__(
        self,
        n_samples: int = 500,
        random_seed: int = 42,
        modality: str = "tabular",
        **lime_kwargs
    ):
        """
        Initialize LIME wrapper.
        
        Args:
            n_samples: Number of perturbation samples
            random_seed: Random seed for reproducibility
            modality: Data modality ("tabular", "text", "image")
            **lime_kwargs: Additional LIME parameters
        """
        super().__init__(random_seed)
        self.n_samples = n_samples
        self.modality = modality.lower()
        self.lime_kwargs = lime_kwargs
        self.method_name = f"LIME_{modality.upper()}"
        
        # Import LIME with error handling
        try:
            import lime
            import lime.lime_tabular
            import lime.lime_text
            import lime.lime_image
            self.lime = lime
            self.lime_tabular = lime.lime_tabular
            self.lime_text = lime.lime_text
            self.lime_image = lime.lime_image
        except ImportError:
            raise ImportError(
                "LIME is required for LIMEWrapper. Install with: pip install lime"
            )
        
        self.explainer = None
        
    def _create_model_wrapper(self, model: Callable) -> Callable:
        """Create a model wrapper compatible with LIME."""
        def model_wrapper(x):
            """Wrapper function for LIME compatibility."""
            # Handle different input formats
            if isinstance(x, np.ndarray):
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                x = torch.from_numpy(x).float()
            elif isinstance(x, list):
                x = np.array(x)
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                x = torch.from_numpy(x).float()
            
            # Move to appropriate device
            device = get_device()
            x = x.to(device)
            
            with torch.no_grad():
                pred = model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                # Apply softmax for probability output
                if len(pred.shape) > 1 and pred.shape[1] > 1:
                    pred = torch.softmax(pred, dim=1)
                
                # Return numpy array for LIME
                return pred.cpu().numpy()
        
        return model_wrapper
    
    def _initialize_tabular_explainer(
        self, 
        training_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        """Initialize LIME tabular explainer."""
        if training_data is None:
            raise ValueError("Training data is required for LIME tabular explainer")
        
        self.explainer = self.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=class_names,
            random_state=self.random_seed,
            **self.lime_kwargs
        )
    
    def explain(
        self,
        model: Callable,
        input_data: Union[torch.Tensor, np.ndarray, Dict, str],
        target_class: Optional[int] = None,
        training_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        **kwargs
    ) -> Attribution:
        """
        Generate LIME explanation for input data.
        
        Args:
            model: Model prediction function
            input_data: Input to explain
            target_class: Target class for explanation
            training_data: Training data for tabular explainer
            feature_names: Feature names for tabular data
            class_names: Class names for all modalities
            **kwargs: Additional LIME parameters
            
        Returns:
            Attribution object with LIME feature importance
        """
        start_time = time.time()
        
        try:
            # Create model wrapper
            model_wrapper = self._create_model_wrapper(model)
            
            # Initialize explainer based on modality
            if self.explainer is None:
                if self.modality == "tabular":
                    self._initialize_tabular_explainer(training_data, feature_names, class_names)
                else:
                    raise ValueError(f"Only tabular modality is currently supported, got: {self.modality}")
            
            # Prepare input data for tabular modality
            if isinstance(input_data, torch.Tensor):
                input_array = input_data.cpu().numpy()
            elif isinstance(input_data, np.ndarray):
                input_array = input_data
            else:
                input_array = self._prepare_input(input_data)
            
            # Ensure 1D for single instance
            if len(input_array.shape) > 1:
                input_array = input_array.flatten()
            
            # Generate explanation
            explanation = self.explainer.explain_instance(
                input_array,
                model_wrapper,
                num_samples=self.n_samples,
                **kwargs
            )
            
            # Extract feature importance
            feature_importance = explanation.as_list()
            
            # Create feature scores array
            n_features = len(input_array)
            feature_scores = np.zeros(n_features)
            
            for feature_idx, importance in feature_importance:
                if isinstance(feature_idx, int) and feature_idx < n_features:
                    feature_scores[feature_idx] = importance
                elif isinstance(feature_idx, str):
                    # Handle named features
                    try:
                        idx = int(feature_idx.split('_')[-1]) if '_' in feature_idx else int(feature_idx)
                        if idx < n_features:
                            feature_scores[idx] = importance
                    except (ValueError, IndexError):
                        continue
            
            # Create feature indices
            feature_indices = list(range(len(feature_scores)))
            
            # Get target class if not provided
            if target_class is None:
                pred = self._get_model_prediction(model, input_data)
                target_class = torch.argmax(pred, dim=-1).item()
            
            computation_time = time.time() - start_time
            
            return Attribution(
                feature_scores=feature_scores,
                feature_indices=feature_indices,
                method_name=self.method_name,
                computation_time=computation_time,
                metadata={
                    'target_class': target_class,
                    'modality': self.modality,
                    'n_samples': self.n_samples,
                    'input_shape': input_data.shape if hasattr(input_data, 'shape') else None
                }
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            warnings.warn(f"LIME explanation failed: {str(e)}")
            
            # Return zero attribution as fallback
            try:
                input_array = self._prepare_input(input_data)
                if len(input_array.shape) == 1:
                    n_features = len(input_array)
                else:
                    n_features = input_array.shape[-1]
            except:
                n_features = 1  # Final fallback
            
            return Attribution(
                feature_scores=np.zeros(n_features),
                feature_indices=list(range(n_features)),
                method_name=f"{self.method_name}_FAILED",
                computation_time=computation_time,
                metadata={'error': str(e)}
            )


class RandomExplainer(ExplainerWrapper):
    """
    Random explainer baseline for sanity checking and negative control.
    Generates random attribution scores to validate metric discrimination.
    """
    
    def __init__(
        self,
        random_seed: int = 42,
        distribution: str = "uniform",
        scale: float = 1.0
    ):
        """
        Initialize random explainer.
        
        Args:
            random_seed: Random seed for reproducibility
            distribution: Distribution type ("uniform", "normal", "exponential")
            scale: Scale parameter for the distribution
        """
        super().__init__(random_seed)
        self.distribution = distribution.lower()
        self.scale = scale
        self.method_name = f"Random_{distribution.upper()}"
        
        # Set random seed
        self.rng = np.random.RandomState(random_seed)
        
    def _generate_random_scores(self, n_features: int) -> np.ndarray:
        """Generate random attribution scores."""
        if self.distribution == "uniform":
            # Uniform distribution between -scale and scale
            return self.rng.uniform(-self.scale, self.scale, n_features)
        elif self.distribution == "normal":
            # Normal distribution with std = scale
            return self.rng.normal(0, self.scale, n_features)
        elif self.distribution == "exponential":
            # Exponential distribution with scale parameter
            scores = self.rng.exponential(self.scale, n_features)
            # Randomly assign signs
            signs = self.rng.choice([-1, 1], n_features)
            return scores * signs
        else:
            warnings.warn(f"Unknown distribution: {self.distribution}, using uniform")
            return self.rng.uniform(-self.scale, self.scale, n_features)
    
    def explain(
        self,
        model: Callable,
        input_data: Union[torch.Tensor, np.ndarray, Dict],
        target_class: Optional[int] = None,
        **kwargs
    ) -> Attribution:
        """
        Generate random explanation for input data.
        
        Args:
            model: Model prediction function (not used, for interface compatibility)
            input_data: Input to explain
            target_class: Target class (not used, for interface compatibility)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Attribution object with random feature importance scores
        """
        start_time = time.time()
        
        try:
            # Determine number of features from input
            if isinstance(input_data, torch.Tensor):
                input_shape = input_data.shape
                if len(input_shape) == 1:
                    n_features = input_shape[0]
                else:
                    n_features = input_shape[-1]  # Last dimension
            elif isinstance(input_data, np.ndarray):
                input_shape = input_data.shape
                if len(input_shape) == 1:
                    n_features = input_shape[0]
                else:
                    n_features = input_shape[-1]  # Last dimension
            elif isinstance(input_data, dict):
                # Handle dictionary inputs (e.g., for transformers)
                if 'input_ids' in input_data:
                    input_tensor = input_data['input_ids']
                    if isinstance(input_tensor, torch.Tensor):
                        n_features = input_tensor.shape[-1]
                    else:
                        n_features = len(input_tensor) if hasattr(input_tensor, '__len__') else 100
                else:
                    n_features = 100  # Default fallback
            else:
                # Fallback for unknown input types
                n_features = 100
            
            # Generate random attribution scores
            feature_scores = self._generate_random_scores(n_features)
            
            # Create feature indices
            feature_indices = list(range(n_features))
            
            # Get target class for metadata (optional)
            if target_class is None:
                try:
                    pred = self._get_model_prediction(model, input_data)
                    target_class = torch.argmax(pred, dim=-1).item()
                except:
                    target_class = 0  # Fallback
            
            computation_time = time.time() - start_time
            
            return Attribution(
                feature_scores=feature_scores,
                feature_indices=feature_indices,
                method_name=self.method_name,
                computation_time=computation_time,
                metadata={
                    'target_class': target_class,
                    'distribution': self.distribution,
                    'scale': self.scale,
                    'n_features': n_features,
                    'random_seed': self.random_seed
                }
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            warnings.warn(f"Random explainer failed: {str(e)}")
            
            # Return zero attribution as ultimate fallback
            return Attribution(
                feature_scores=np.zeros(10),  # Minimal fallback
                feature_indices=list(range(10)),
                method_name=f"{self.method_name}_FAILED",
                computation_time=computation_time,
                metadata={'error': str(e)}
            )


# Convenience functions
def create_shap_explainer(
    explainer_type: str = "kernel",
    n_samples: int = 1000,
    random_seed: int = 42,
    **kwargs
) -> SHAPWrapper:
    """
    Convenience function to create SHAP explainer.
    
    Args:
        explainer_type: Type of SHAP explainer ("kernel", "tree", "deep")
        n_samples: Number of samples for KernelSHAP
        random_seed: Random seed
        **kwargs: Additional SHAP parameters
        
    Returns:
        Configured SHAPWrapper instance
    """
    return SHAPWrapper(
        explainer_type=explainer_type,
        n_samples=n_samples,
        random_seed=random_seed,
        **kwargs
    )


def create_integrated_gradients_explainer(
    n_steps: int = 50,
    baseline_strategy: str = "zero",
    random_seed: int = 42,
    internal_batch_size: int = 32
) -> IntegratedGradientsWrapper:
    """
    Convenience function to create Integrated Gradients explainer.
    
    Args:
        n_steps: Number of integration steps
        baseline_strategy: Baseline strategy ("zero", "random", "mean")
        random_seed: Random seed
        internal_batch_size: Batch size for gradient computation
        
    Returns:
        Configured IntegratedGradientsWrapper instance
    """
    return IntegratedGradientsWrapper(
        n_steps=n_steps,
        baseline_strategy=baseline_strategy,
        random_seed=random_seed,
        internal_batch_size=internal_batch_size
    )


def create_lime_explainer(
    n_samples: int = 500,
    modality: str = "tabular",
    random_seed: int = 42,
    **kwargs
) -> LIMEWrapper:
    """
    Convenience function to create LIME explainer.
    
    Args:
        n_samples: Number of perturbation samples
        modality: Data modality ("tabular", "text", "image")
        random_seed: Random seed
        **kwargs: Additional LIME parameters
        
    Returns:
        Configured LIMEWrapper instance
    """
    return LIMEWrapper(
        n_samples=n_samples,
        modality=modality,
        random_seed=random_seed,
        **kwargs
    )


def create_random_explainer(
    random_seed: int = 42,
    distribution: str = "uniform",
    scale: float = 1.0
) -> RandomExplainer:
    """
    Convenience function to create random explainer baseline.
    
    Args:
        random_seed: Random seed for reproducibility
        distribution: Distribution type ("uniform", "normal", "exponential")
        scale: Scale parameter for the distribution
        
    Returns:
        Configured RandomExplainer instance
    """
    return RandomExplainer(
        random_seed=random_seed,
        distribution=distribution,
        scale=scale
    )


class OcclusionExplainer(ExplainerWrapper):
    """
    Simple occlusion explainer for tokenized text.
    For each token position, mask (set to pad id 0) and measure target score drop.
    """

    def __init__(self, max_tokens: int = 64, random_seed: int = 42):
        super().__init__(random_seed)
        self.max_tokens = max_tokens
        self.method_name = "Occlusion"

    def explain(
        self,
        model: Callable,
        input_data: Union[torch.Tensor, np.ndarray, Dict],
        target_class: Optional[int] = None,
        **kwargs
    ) -> Attribution:
        start_time = time.time()
        try:
            # Prepare input ids tensor (1, L)
            if isinstance(input_data, np.ndarray):
                ids = torch.from_numpy(input_data).long()
            elif isinstance(input_data, torch.Tensor):
                ids = input_data.long()
            elif isinstance(input_data, dict) and 'input_ids' in input_data:
                ids = input_data['input_ids']
                if isinstance(ids, np.ndarray):
                    ids = torch.from_numpy(ids).long()
            else:
                raise ValueError(f"Unsupported input type for Occlusion: {type(input_data)}")

            if ids.dim() == 1:
                ids = ids.unsqueeze(0)

            device = get_device()
            ids = ids.to(device)

            # Original prediction
            with torch.no_grad():
                orig_pred = model(ids)
                if isinstance(orig_pred, tuple):
                    orig_pred = orig_pred[0]
                if orig_pred.dim() == 2:
                    # Assume logits over classes/vocab
                    base_vec = orig_pred.squeeze(0)
                else:
                    base_vec = orig_pred.squeeze()

            if target_class is None:
                target_class = int(torch.argmax(base_vec).item())

            L = ids.shape[-1]
            limit = min(self.max_tokens, L)
            scores = torch.zeros(L, device=device)

            # Evaluate occlusion per token (up to limit)
            for i in range(limit):
                occluded = ids.clone()
                occluded[0, i] = 0  # pad id assumed 0
                with torch.no_grad():
                    pred = model(occluded)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    vec = pred.squeeze(0) if pred.dim() == 2 else pred.squeeze()
                # Importance: drop in target score
                drop = base_vec[target_class] - vec[target_class]
                scores[i] = drop

            feature_scores = scores.detach().cpu().numpy()
            feature_indices = list(range(L))
            comp_time = time.time() - start_time

            return Attribution(
                feature_scores=feature_scores,
                feature_indices=feature_indices,
                method_name=self.method_name,
                computation_time=comp_time,
                metadata={
                    'target_class': target_class,
                    'max_tokens': self.max_tokens,
                    'sequence_length': L,
                    'device': str(device),
                }
            )

        except Exception as e:
            comp_time = time.time() - start_time
            warnings.warn(f"Occlusion explanation failed: {e}")
            try:
                length = int(input_data.shape[-1]) if hasattr(input_data, 'shape') else 1
            except Exception:
                length = 1
            return Attribution(
                feature_scores=np.zeros(length),
                feature_indices=list(range(length)),
                method_name=f"{self.method_name}_FAILED",
                computation_time=comp_time,
                metadata={'error': str(e)}
            )


def create_all_explainers(
    random_seed: int = 42,
    shap_type: str = "kernel",
    shap_samples: int = 1000,
    ig_steps: int = 50,
    ig_baseline: str = "zero",
    lime_samples: int = 500,
    lime_modality: str = "tabular"
) -> Dict[str, ExplainerWrapper]:
    """
    Create all explanation method wrappers with consistent configuration.
    
    Args:
        random_seed: Random seed for all explainers
        shap_type: SHAP explainer type
        shap_samples: Number of samples for SHAP
        ig_steps: Number of steps for Integrated Gradients
        ig_baseline: Baseline strategy for IG
        lime_samples: Number of samples for LIME
        lime_modality: Modality for LIME
        
    Returns:
        Dictionary of configured explainer instances
    """
    return {
        'shap': create_shap_explainer(
            explainer_type=shap_type,
            n_samples=shap_samples,
            random_seed=random_seed
        ),
        'integrated_gradients': create_integrated_gradients_explainer(
            n_steps=ig_steps,
            baseline_strategy=ig_baseline,
            random_seed=random_seed
        ),
        'lime': create_lime_explainer(
            n_samples=lime_samples,
            modality=lime_modality,
            random_seed=random_seed
        ),
        'random': create_random_explainer(
            random_seed=random_seed,
            distribution="uniform",
            scale=1.0
        )
    }