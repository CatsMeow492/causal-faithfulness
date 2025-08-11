#!/usr/bin/env python3
"""
Basic test for SST-2 dataset loading and BERT model functionality.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Test basic imports
try:
    print("Testing basic imports...")
    
    # Test config import
    import config
    print(f"✓ Config imported, device: {config.get_device()}")
    
    # Test datasets import
    import datasets
    print("✓ Datasets imported")
    
    # Test models import  
    import models
    print("✓ Models imported")
    
    print("\nTesting dataset loading...")
    
    # Test SST-2 dataset loading
    dataset_manager = datasets.DatasetManager()
    sst2_samples = dataset_manager.load_sst2(split="validation", num_samples=5)
    print(f"✓ Loaded {len(sst2_samples)} SST-2 samples")
    
    # Print first sample
    if sst2_samples:
        sample = sst2_samples[0]
        print(f"  Sample text: {sample.text[:100]}...")
        print(f"  Sample label: {sample.label}")
        print(f"  Token shape: {sample.tokens.shape}")
    
    print("\nTesting model loading...")
    
    # Test BERT model loading
    model_manager = models.ModelManager()
    bert_model = model_manager.load_bert_sst2()
    print(f"✓ Loaded BERT model on {bert_model.device}")
    
    # Test prediction
    if sst2_samples:
        pred = bert_model.predict(sst2_samples[0])
        print(f"  Prediction class: {pred.predicted_class}")
        print(f"  Confidence: {pred.confidence:.4f}")
    
    print("\n✅ All basic tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()