#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all core module imports."""
    print("Testing imports...")
    
    try:
        # Test config module
        from src.config import get_device, get_batch_size, print_system_info
        print("✓ Config module imported successfully")
        
        # Test faithfulness module
        from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig, compute_faithfulness_score
        print("✓ Faithfulness module imported successfully")
        
        # Test explainers module
        from src.explainers import SHAPWrapper, IntegratedGradientsWrapper, LIMEWrapper, RandomExplainer
        print("✓ Explainers module imported successfully")
        
        # Test datasets module
        from src.datasets import DatasetManager, SST2DatasetLoader, WikiText2DatasetLoader
        print("✓ Datasets module imported successfully")
        
        # Test models module
        from src.models import ModelManager, BERTSentimentWrapper, GPT2LanguageModelWrapper
        print("✓ Models module imported successfully")
        
        # Test masking module
        from src.masking import FeatureMasker, DataModality, MaskingStrategy
        print("✓ Masking module imported successfully")
        
        # Test baseline module
        from src.baseline import BaselineGenerator, BaselineStrategy
        print("✓ Baseline module imported successfully")
        
        # Test robust computation module
        from src.robust_computation import safe_divide, execute_with_retries, memory_monitor
        print("✓ Robust computation module imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test hardware detection
        from src.config import get_device, get_batch_size, print_system_info
        device = get_device()
        batch_size = get_batch_size()
        print(f"✓ Device detection: {device}")
        print(f"✓ Batch size: {batch_size}")
        
        # Test faithfulness config
        from src.faithfulness import FaithfulnessConfig
        config = FaithfulnessConfig()
        print(f"✓ Faithfulness config created: device={config.device}")
        
        # Test masking
        from src.masking import create_masker, DataModality
        masker = create_masker("tabular", "mean")
        print("✓ Masker created successfully")
        
        # Test baseline generation
        from src.baseline import create_baseline_generator
        baseline_gen = create_baseline_generator("tabular", "gaussian")
        print("✓ Baseline generator created successfully")
        
        print("\n✅ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        success = test_basic_functionality()
    
    if success:
        print("\n🎉 All tests passed! Implementation is ready.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed. Check the errors above.")
        sys.exit(1)