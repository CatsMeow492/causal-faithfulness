"""
Dataset loading and preprocessing for causal-faithfulness evaluation.
Supports SST-2 (sentiment analysis) and WikiText-2 (language modeling) datasets.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from datasets import load_dataset
import warnings
from .config import get_device, get_batch_size, DEFAULT_CONFIG


@dataclass
class DatasetSample:
    """Unified data structure for dataset samples."""
    text: str
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    label: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DatasetInfo:
    """Dataset metadata and licensing information."""
    name: str
    task_type: str  # 'classification' or 'language_modeling'
    license: str
    citation: str
    num_samples: int
    max_length: int


class BaseDatasetLoader:
    """Base class for dataset loaders with common functionality."""
    
    def __init__(self, max_length: int = 512, device: Optional[torch.device] = None):
        self.max_length = max_length
        self.device = device or get_device()
        self.tokenizer = None
        self.dataset_info = None
        
    def _validate_sample(self, sample: DatasetSample) -> bool:
        """Validate a dataset sample."""
        if sample.text is None or len(sample.text.strip()) == 0:
            return False
        if sample.tokens is None or sample.tokens.numel() == 0:
            return False
        if sample.attention_mask is None:
            return False
        return True
    
    def _truncate_sequence(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Truncate sequences to max_length."""
        if tokens.size(-1) > self.max_length:
            tokens = tokens[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
        return tokens, attention_mask
    
    def get_dataset_info(self) -> DatasetInfo:
        """Get dataset information and licensing details."""
        return self.dataset_info


class SST2DatasetLoader(BaseDatasetLoader):
    """
    Stanford Sentiment Treebank v2 dataset loader for BERT sentiment analysis.
    
    License: Custom license allowing research use
    Citation: Socher et al. (2013) "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank"
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512, device: Optional[torch.device] = None):
        super().__init__(max_length, device)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.dataset_info = DatasetInfo(
            name="SST-2",
            task_type="classification",
            license="Custom (research use allowed)",
            citation="Socher et al. (2013) Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
            num_samples=0,  # Will be updated after loading
            max_length=max_length
        )
    
    def load_dataset(self, split: str = "validation", num_samples: Optional[int] = None) -> List[DatasetSample]:
        """
        Load SST-2 dataset with BERT tokenization.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            num_samples: Maximum number of samples to load (None for all)
            
        Returns:
            List of DatasetSample objects
        """
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("glue", "sst2", split=split)
            
            if num_samples is not None:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            samples = []
            
            for i, example in enumerate(dataset):
                text = example["sentence"]
                label = example["label"]
                
                # Tokenize with BERT tokenizer
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                sample = DatasetSample(
                    text=text,
                    tokens=encoding["input_ids"].squeeze(0),
                    attention_mask=encoding["attention_mask"].squeeze(0),
                    label=label,
                    metadata={
                        "dataset": "sst2",
                        "split": split,
                        "index": i,
                        "original_length": len(text.split())
                    }
                )
                
                if self._validate_sample(sample):
                    samples.append(sample)
                else:
                    warnings.warn(f"Invalid sample at index {i}, skipping")
            
            self.dataset_info.num_samples = len(samples)
            print(f"Loaded {len(samples)} valid SST-2 samples from {split} split")
            
            return samples
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SST-2 dataset: {str(e)}")
    
    def preprocess_for_bert(self, samples: List[DatasetSample]) -> Dict[str, torch.Tensor]:
        """
        Preprocess samples for BERT model input.
        
        Args:
            samples: List of DatasetSample objects
            
        Returns:
            Dictionary with batched tensors
        """
        if not samples:
            raise ValueError("No samples provided for preprocessing")
        
        input_ids = torch.stack([sample.tokens for sample in samples])
        attention_masks = torch.stack([sample.attention_mask for sample in samples])
        labels = torch.tensor([sample.label for sample in samples if sample.label is not None])
        
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_masks.to(self.device),
            "labels": labels.to(self.device) if len(labels) > 0 else None
        }


class WikiText2DatasetLoader(BaseDatasetLoader):
    """
    WikiText-2 dataset loader for GPT-2 language modeling experiments.
    
    License: Creative Commons Attribution-ShareAlike License
    Citation: Merity et al. (2016) "Pointer Sentinel Mixture Models"
    """
    
    def __init__(self, model_name: str = "gpt2", max_length: int = 512, device: Optional[torch.device] = None):
        super().__init__(max_length, device)
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Add padding token for GPT-2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.dataset_info = DatasetInfo(
            name="WikiText-2",
            task_type="language_modeling",
            license="Creative Commons Attribution-ShareAlike",
            citation="Merity et al. (2016) Pointer Sentinel Mixture Models",
            num_samples=0,  # Will be updated after loading
            max_length=max_length
        )
    
    def load_dataset(self, split: str = "validation", num_samples: Optional[int] = None) -> List[DatasetSample]:
        """
        Load WikiText-2 dataset with GPT-2 tokenization.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            num_samples: Maximum number of samples to load (None for all)
            
        Returns:
            List of DatasetSample objects
        """
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            
            samples = []
            sample_count = 0
            
            for i, example in enumerate(dataset):
                text = example["text"].strip()
                
                # Skip empty lines and section headers
                if not text or text.startswith("=") or len(text.split()) < 10:
                    continue
                
                # Tokenize with GPT-2 tokenizer
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                sample = DatasetSample(
                    text=text,
                    tokens=encoding["input_ids"].squeeze(0),
                    attention_mask=encoding["attention_mask"].squeeze(0),
                    label=None,  # Language modeling doesn't have explicit labels
                    metadata={
                        "dataset": "wikitext2",
                        "split": split,
                        "index": i,
                        "original_length": len(text.split()),
                        "perplexity_target": True  # Used for perplexity computation
                    }
                )
                
                if self._validate_sample(sample):
                    samples.append(sample)
                    sample_count += 1
                    
                    if num_samples is not None and sample_count >= num_samples:
                        break
                else:
                    warnings.warn(f"Invalid sample at index {i}, skipping")
            
            self.dataset_info.num_samples = len(samples)
            print(f"Loaded {len(samples)} valid WikiText-2 samples from {split} split")
            
            return samples
            
        except Exception as e:
            raise RuntimeError(f"Failed to load WikiText-2 dataset: {str(e)}")
    
    def preprocess_for_gpt2(self, samples: List[DatasetSample]) -> Dict[str, torch.Tensor]:
        """
        Preprocess samples for GPT-2 model input.
        
        Args:
            samples: List of DatasetSample objects
            
        Returns:
            Dictionary with batched tensors
        """
        if not samples:
            raise ValueError("No samples provided for preprocessing")
        
        input_ids = torch.stack([sample.tokens for sample in samples])
        attention_masks = torch.stack([sample.attention_mask for sample in samples])
        
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_masks.to(self.device),
            "labels": input_ids.to(self.device)  # For language modeling, labels = input_ids
        }


class DatasetManager:
    """
    Unified dataset manager for handling multiple datasets and preprocessing pipelines.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.loaders = {}
        self.datasets = {}
        
    def register_loader(self, name: str, loader: BaseDatasetLoader):
        """Register a dataset loader."""
        self.loaders[name] = loader
        
    def load_sst2(self, split: str = "validation", num_samples: int = 200, model_name: str = "bert-base-uncased") -> List[DatasetSample]:
        """Load SST-2 dataset for sentiment analysis experiments."""
        if "sst2" not in self.loaders:
            self.loaders["sst2"] = SST2DatasetLoader(model_name=model_name, device=self.device)
        
        samples = self.loaders["sst2"].load_dataset(split=split, num_samples=num_samples)
        self.datasets["sst2"] = samples
        return samples
    
    def load_wikitext2(self, split: str = "validation", num_samples: int = 200, model_name: str = "gpt2") -> List[DatasetSample]:
        """Load WikiText-2 dataset for language modeling experiments."""
        if "wikitext2" not in self.loaders:
            self.loaders["wikitext2"] = WikiText2DatasetLoader(model_name=model_name, device=self.device)
        
        samples = self.loaders["wikitext2"].load_dataset(split=split, num_samples=num_samples)
        self.datasets["wikitext2"] = samples
        return samples
    
    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """Get information about a loaded dataset."""
        if dataset_name not in self.loaders:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        return self.loaders[dataset_name].get_dataset_info()
    
    def validate_datasets(self) -> Dict[str, bool]:
        """Validate all loaded datasets."""
        validation_results = {}
        
        for name, samples in self.datasets.items():
            try:
                # Basic validation
                if not samples:
                    validation_results[name] = False
                    continue
                
                # Check sample structure
                sample = samples[0]
                if not isinstance(sample, DatasetSample):
                    validation_results[name] = False
                    continue
                
                # Check tensor shapes
                if sample.tokens.dim() != 1 or sample.attention_mask.dim() != 1:
                    validation_results[name] = False
                    continue
                
                validation_results[name] = True
                
            except Exception as e:
                warnings.warn(f"Validation failed for {name}: {str(e)}")
                validation_results[name] = False
        
        return validation_results
    
    def print_dataset_summary(self):
        """Print summary of all loaded datasets."""
        print("=== Dataset Summary ===")
        
        for name, samples in self.datasets.items():
            if name in self.loaders:
                info = self.loaders[name].get_dataset_info()
                print(f"\n{info.name}:")
                print(f"  Task: {info.task_type}")
                print(f"  Samples: {len(samples)}")
                print(f"  Max length: {info.max_length}")
                print(f"  License: {info.license}")
                
                if samples:
                    avg_length = np.mean([len(s.text.split()) for s in samples])
                    print(f"  Avg text length: {avg_length:.1f} words")


def create_license_documentation():
    """Create documentation for dataset licenses and compliance."""
    license_info = {
        "SST-2": {
            "license": "Custom license allowing research use",
            "citation": "Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013). Recursive deep models for semantic compositionality over a sentiment treebank. EMNLP.",
            "url": "https://nlp.stanford.edu/sentiment/",
            "compliance": "Research use only, proper citation required"
        },
        "WikiText-2": {
            "license": "Creative Commons Attribution-ShareAlike",
            "citation": "Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843.",
            "url": "https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/",
            "compliance": "Attribution required, share-alike terms apply"
        }
    }
    
    # Create license documentation file
    license_doc = "# Dataset License Compliance\n\n"
    license_doc += "This document outlines the licensing terms and compliance requirements for datasets used in this project.\n\n"
    
    for dataset, info in license_info.items():
        license_doc += f"## {dataset}\n\n"
        license_doc += f"**License:** {info['license']}\n\n"
        license_doc += f"**Citation:** {info['citation']}\n\n"
        license_doc += f"**URL:** {info['url']}\n\n"
        license_doc += f"**Compliance:** {info['compliance']}\n\n"
    
    # Save to data directory
    os.makedirs("data", exist_ok=True)
    with open("data/LICENSE_COMPLIANCE.md", "w") as f:
        f.write(license_doc)
    
    print("Dataset license documentation created at data/LICENSE_COMPLIANCE.md")


# Convenience functions for quick dataset loading
def load_sst2_validation(num_samples: int = 200) -> List[DatasetSample]:
    """Quick loader for SST-2 validation set."""
    manager = DatasetManager()
    return manager.load_sst2(split="validation", num_samples=num_samples)


def load_wikitext2_validation(num_samples: int = 200) -> List[DatasetSample]:
    """Quick loader for WikiText-2 validation set."""
    manager = DatasetManager()
    return manager.load_wikitext2(split="validation", num_samples=num_samples)