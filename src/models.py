"""
Model integration wrappers for causal-faithfulness evaluation.
Supports BERT (sentiment analysis) and GPT-2 (language modeling) with hardware-aware loading.
"""

import torch
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    BitsAndBytesConfig
)

from .config import get_device, get_batch_size, DEFAULT_CONFIG
from .datasets import DatasetSample


@dataclass
class ModelPrediction:
    """Unified prediction result structure."""
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_class: Optional[int] = None
    confidence: Optional[float] = None
    perplexity: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    model_name: str
    device: torch.device
    batch_size: int
    max_length: int
    use_quantization: bool = False
    load_in_4bit: bool = False
    torch_dtype: torch.dtype = torch.float32


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config.device
        self._is_loaded = False
        
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def predict(self, inputs: Union[torch.Tensor, List[str], DatasetSample]) -> ModelPrediction:
        """Make predictions on inputs."""
        pass
    
    @abstractmethod
    def predict_batch(self, inputs: List[Union[torch.Tensor, str, DatasetSample]]) -> List[ModelPrediction]:
        """Make batch predictions."""
        pass
    
    def _ensure_loaded(self):
        """Ensure model is loaded before inference."""
        if not self._is_loaded:
            self.load_model()
    
    def _handle_device_fallback(self, error: Exception) -> torch.device:
        """Handle device-specific errors with fallback."""
        if "MPS" in str(error) or "mps" in str(error):
            warnings.warn(f"MPS error: {error}. Falling back to CPU.")
            return torch.device('cpu')
        elif "CUDA" in str(error) or "cuda" in str(error):
            warnings.warn(f"CUDA error: {error}. Falling back to CPU.")
            return torch.device('cpu')
        else:
            raise error


class BERTSentimentWrapper(BaseModelWrapper):
    """
    Wrapper for BERT-based sentiment analysis models.
    Supports bert-base-uncased fine-tuned on SST-2.
    """
    
    def __init__(self, model_name: str = "textattack/bert-base-uncased-SST-2", config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                model_name=model_name,
                device=get_device(),
                batch_size=get_batch_size(),
                max_length=512,
                torch_dtype=torch.float32
            )
        
        super().__init__(config)
        self.num_labels = 2  # Binary sentiment classification
        
    def load_model(self):
        """Load BERT model and tokenizer with hardware-aware configuration."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization if requested and supported
            quantization_config = None
            if self.config.use_quantization and self.config.load_in_4bit:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                except Exception as e:
                    warnings.warn(f"4-bit quantization not available: {e}")
                    quantization_config = None
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.num_labels,
                quantization_config=quantization_config,
                torch_dtype=self.config.torch_dtype,
                device_map="auto" if quantization_config else None
            )
            
            # Move to device if not using device_map
            if quantization_config is None:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._is_loaded = True
            
            print(f"Loaded BERT model '{self.config.model_name}' on {self.device}")
            
        except Exception as e:
            # Try fallback device
            fallback_device = self._handle_device_fallback(e)
            if fallback_device != self.device:
                self.device = fallback_device
                self.config.device = fallback_device
                self.load_model()  # Retry with fallback device
            else:
                raise RuntimeError(f"Failed to load BERT model: {str(e)}")
    
    def predict(self, inputs: Union[torch.Tensor, str, DatasetSample]) -> ModelPrediction:
        """Make prediction on single input."""
        self._ensure_loaded()
        
        # Handle different input types
        if isinstance(inputs, str):
            # Tokenize text input
            encoding = self.tokenizer(
                inputs,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
        elif isinstance(inputs, DatasetSample):
            # Use pre-tokenized input
            input_ids = inputs.tokens.unsqueeze(0).to(self.device)
            attention_mask = inputs.attention_mask.unsqueeze(0).to(self.device)
            
        elif isinstance(inputs, torch.Tensor):
            # Assume input is tokenized
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            input_ids = inputs.to(self.device)
            attention_mask = torch.ones_like(input_ids)
            
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        
        # Make prediction
        with torch.no_grad():
            try:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = torch.max(probabilities, dim=-1)[0].item()
                
                return ModelPrediction(
                    logits=logits.squeeze(0),
                    probabilities=probabilities.squeeze(0),
                    predicted_class=predicted_class,
                    confidence=confidence,
                    metadata={
                        "model_name": self.config.model_name,
                        "device": str(self.device),
                        "input_shape": input_ids.shape
                    }
                )
                
            except Exception as e:
                # Try CPU fallback
                if self.device.type != 'cpu':
                    warnings.warn(f"Prediction failed on {self.device}, trying CPU: {e}")
                    input_ids = input_ids.cpu()
                    attention_mask = attention_mask.cpu()
                    self.model = self.model.cpu()
                    self.device = torch.device('cpu')
                    return self.predict(inputs)
                else:
                    raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, inputs: List[Union[torch.Tensor, str, DatasetSample]]) -> List[ModelPrediction]:
        """Make batch predictions with automatic batching."""
        self._ensure_loaded()
        
        if not inputs:
            return []
        
        predictions = []
        batch_size = self.config.batch_size
        
        # Process in batches
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_predictions = self._predict_batch_internal(batch)
            predictions.extend(batch_predictions)
        
        return predictions
    
    def _predict_batch_internal(self, batch: List[Union[torch.Tensor, str, DatasetSample]]) -> List[ModelPrediction]:
        """Internal batch prediction method."""
        # Prepare batch inputs
        input_ids_list = []
        attention_masks_list = []
        
        for item in batch:
            if isinstance(item, str):
                encoding = self.tokenizer(
                    item,
                    truncation=True,
                    padding="max_length",
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                input_ids_list.append(encoding["input_ids"].squeeze(0))
                attention_masks_list.append(encoding["attention_mask"].squeeze(0))
                
            elif isinstance(item, DatasetSample):
                input_ids_list.append(item.tokens)
                attention_masks_list.append(item.attention_mask)
                
            elif isinstance(item, torch.Tensor):
                if item.dim() == 1:
                    input_ids_list.append(item)
                    attention_masks_list.append(torch.ones_like(item))
                else:
                    input_ids_list.append(item.squeeze(0))
                    attention_masks_list.append(torch.ones_like(item.squeeze(0)))
        
        # Stack into batch tensors
        batch_input_ids = torch.stack(input_ids_list).to(self.device)
        batch_attention_masks = torch.stack(attention_masks_list).to(self.device)
        
        # Make batch prediction
        with torch.no_grad():
            try:
                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(logits, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]
                
                # Convert to individual predictions
                predictions = []
                for i in range(len(batch)):
                    predictions.append(ModelPrediction(
                        logits=logits[i],
                        probabilities=probabilities[i],
                        predicted_class=predicted_classes[i].item(),
                        confidence=confidences[i].item(),
                        metadata={
                            "model_name": self.config.model_name,
                            "device": str(self.device),
                            "batch_index": i
                        }
                    ))
                
                return predictions
                
            except Exception as e:
                # Fallback to individual predictions
                warnings.warn(f"Batch prediction failed, falling back to individual predictions: {e}")
                return [self.predict(item) for item in batch]


class GPT2LanguageModelWrapper(BaseModelWrapper):
    """
    Wrapper for GPT-2 language modeling.
    Supports perplexity computation for WikiText-2 evaluation.
    """
    
    def __init__(self, model_name: str = "gpt2", config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig(
                model_name=model_name,
                device=get_device(),
                batch_size=get_batch_size(8),  # Smaller batch for language modeling
                max_length=512,
                torch_dtype=torch.float32
            )
        
        super().__init__(config)
        
    def load_model(self):
        """Load GPT-2 model and tokenizer with hardware-aware configuration."""
        try:
            # Load tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization if requested
            quantization_config = None
            if self.config.use_quantization and self.config.load_in_4bit:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                except Exception as e:
                    warnings.warn(f"4-bit quantization not available: {e}")
                    quantization_config = None
            
            # Load model
            self.model = GPT2LMHeadModel.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                torch_dtype=self.config.torch_dtype,
                device_map="auto" if quantization_config else None
            )
            
            # Move to device if not using device_map
            if quantization_config is None:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._is_loaded = True
            
            print(f"Loaded GPT-2 model '{self.config.model_name}' on {self.device}")
            
        except Exception as e:
            # Try fallback device
            fallback_device = self._handle_device_fallback(e)
            if fallback_device != self.device:
                self.device = fallback_device
                self.config.device = fallback_device
                self.load_model()  # Retry with fallback device
            else:
                raise RuntimeError(f"Failed to load GPT-2 model: {str(e)}")
    
    def predict(self, inputs: Union[torch.Tensor, str, DatasetSample]) -> ModelPrediction:
        """Make prediction and compute perplexity."""
        self._ensure_loaded()
        
        # Handle different input types
        if isinstance(inputs, str):
            encoding = self.tokenizer(
                inputs,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
        elif isinstance(inputs, DatasetSample):
            input_ids = inputs.tokens.unsqueeze(0).to(self.device)
            attention_mask = inputs.attention_mask.unsqueeze(0).to(self.device)
            
        elif isinstance(inputs, torch.Tensor):
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            input_ids = inputs.to(self.device)
            attention_mask = torch.ones_like(input_ids)
            
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        
        # Make prediction
        with torch.no_grad():
            try:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                logits = outputs.logits
                loss = outputs.loss
                
                # Compute perplexity
                perplexity = torch.exp(loss).item() if loss is not None else None
                
                # Get probabilities for next token prediction
                probabilities = F.softmax(logits, dim=-1)
                
                return ModelPrediction(
                    logits=logits.squeeze(0),
                    probabilities=probabilities.squeeze(0),
                    perplexity=perplexity,
                    metadata={
                        "model_name": self.config.model_name,
                        "device": str(self.device),
                        "loss": loss.item() if loss is not None else None,
                        "input_shape": input_ids.shape
                    }
                )
                
            except Exception as e:
                # Try CPU fallback
                if self.device.type != 'cpu':
                    warnings.warn(f"Prediction failed on {self.device}, trying CPU: {e}")
                    input_ids = input_ids.cpu()
                    attention_mask = attention_mask.cpu()
                    self.model = self.model.cpu()
                    self.device = torch.device('cpu')
                    return self.predict(inputs)
                else:
                    raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, inputs: List[Union[torch.Tensor, str, DatasetSample]]) -> List[ModelPrediction]:
        """Make batch predictions with perplexity computation."""
        self._ensure_loaded()
        
        if not inputs:
            return []
        
        predictions = []
        batch_size = self.config.batch_size
        
        # Process in batches
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_predictions = self._predict_batch_internal(batch)
            predictions.extend(batch_predictions)
        
        return predictions
    
    def _predict_batch_internal(self, batch: List[Union[torch.Tensor, str, DatasetSample]]) -> List[ModelPrediction]:
        """Internal batch prediction method."""
        # For language modeling, we often process individually due to variable lengths
        # and perplexity computation requirements
        return [self.predict(item) for item in batch]


class ModelManager:
    """
    Unified model manager for handling multiple model types and configurations.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.models = {}
        self.configs = {}
        
    def load_bert_sst2(self, model_name: str = "textattack/bert-base-uncased-SST-2", use_quantization: bool = False) -> BERTSentimentWrapper:
        """Load BERT model fine-tuned on SST-2."""
        config = ModelConfig(
            model_name=model_name,
            device=self.device,
            batch_size=get_batch_size(),
            max_length=512,
            use_quantization=use_quantization,
            load_in_4bit=use_quantization,
            torch_dtype=torch.float16 if use_quantization else torch.float32
        )
        
        model = BERTSentimentWrapper(model_name, config)
        model.load_model()
        
        self.models["bert_sst2"] = model
        self.configs["bert_sst2"] = config
        
        return model
    
    def load_gpt2_small(self, model_name: str = "gpt2", use_quantization: bool = False) -> GPT2LanguageModelWrapper:
        """Load GPT-2 small model for language modeling."""
        config = ModelConfig(
            model_name=model_name,
            device=self.device,
            batch_size=get_batch_size(8),  # Smaller batch for LM
            max_length=512,
            use_quantization=use_quantization,
            load_in_4bit=use_quantization,
            torch_dtype=torch.float16 if use_quantization else torch.float32
        )
        
        model = GPT2LanguageModelWrapper(model_name, config)
        model.load_model()
        
        self.models["gpt2_small"] = model
        self.configs["gpt2_small"] = config
        
        return model
    
    def get_model(self, model_name: str) -> BaseModelWrapper:
        """Get a loaded model by name."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available models: {list(self.models.keys())}")
        return self.models[model_name]
    
    def list_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())
    
    def print_model_summary(self):
        """Print summary of all loaded models."""
        print("=== Model Summary ===")
        
        for name, model in self.models.items():
            config = self.configs[name]
            print(f"\n{name}:")
            print(f"  Model: {config.model_name}")
            print(f"  Device: {config.device}")
            print(f"  Batch size: {config.batch_size}")
            print(f"  Max length: {config.max_length}")
            print(f"  Quantization: {config.use_quantization}")
            print(f"  Loaded: {model._is_loaded}")


# Convenience functions for quick model loading
def load_bert_sst2_model(use_quantization: bool = False) -> BERTSentimentWrapper:
    """Quick loader for BERT SST-2 model."""
    manager = ModelManager()
    return manager.load_bert_sst2(use_quantization=use_quantization)


def load_gpt2_model(use_quantization: bool = False) -> GPT2LanguageModelWrapper:
    """Quick loader for GPT-2 model."""
    manager = ModelManager()
    return manager.load_gpt2_small(use_quantization=use_quantization)


def create_model_prediction_function(model: BaseModelWrapper) -> Callable:
    """
    Create a model prediction function compatible with faithfulness evaluation.
    
    Args:
        model: Loaded model wrapper
        
    Returns:
        Function that takes inputs and returns predictions
    """
    def predict_fn(inputs):
        """Model prediction function for faithfulness evaluation."""
        if isinstance(inputs, (list, tuple)):
            return model.predict_batch(inputs)
        else:
            return model.predict(inputs)
    
    return predict_fn