#!/usr/bin/env python3
"""
Test script to verify FLAN-T5 finetuning installation and basic functionality
"""

import sys
import os
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required packages can be imported"""
    logger.info("Testing imports...")
    
    try:
        import transformers
        logger.info(f"✓ Transformers version: {transformers.__version__}")
    except ImportError as e:
        logger.error(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import datasets
        logger.info(f"✓ Datasets version: {datasets.__version__}")
    except ImportError as e:
        logger.error(f"✗ Datasets import failed: {e}")
        return False
    
    try:
        import peft
        logger.info(f"✓ PEFT version: {peft.__version__}")
    except ImportError as e:
        logger.error(f"✗ PEFT import failed: {e}")
        return False
    
    try:
        import accelerate
        logger.info(f"✓ Accelerate version: {accelerate.__version__}")
    except ImportError as e:
        logger.error(f"✗ Accelerate import failed: {e}")
        return False
    
    try:
        import wandb
        logger.info(f"✓ WandB version: {wandb.__version__}")
    except ImportError as e:
        logger.error(f"✗ WandB import failed: {e}")
        return False
    
    return True

def test_device_detection():
    """Test device detection"""
    logger.info("Testing device detection...")
    
    # Test CUDA
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("✗ CUDA not available")
    
    # Test MPS
    if torch.backends.mps.is_available():
        logger.info("✓ MPS (Apple Silicon) available")
    else:
        logger.info("✗ MPS not available")
    
    # Test CPU
    logger.info("✓ CPU always available")

def test_model_loading():
    """Test if FLAN-T5 models can be loaded"""
    logger.info("Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Test small model loading
        model_name = "google/flan-t5-small"
        logger.info(f"Testing model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("✓ Tokenizer loaded successfully")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        logger.info("✓ Model loaded successfully")
        
        # Test basic generation
        inputs = tokenizer("Translate this to French: Hello world", return_tensors="pt")
        outputs = model.generate(**inputs, max_length=20)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✓ Basic generation test passed: {generated_text}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False

def test_finetuner_class():
    """Test if the FLANT5Finetuner class can be instantiated"""
    logger.info("Testing FLANT5Finetuner class...")
    
    try:
        from flan_t5_finetune import FLANT5Finetuner
        
        # Test instantiation
        finetuner = FLANT5Finetuner(
            model_size="small",
            device="cpu",
            batch_size=1,
            num_epochs=1
        )
        logger.info("✓ FLANT5Finetuner instantiated successfully")
        
        # Test model sizes
        logger.info(f"✓ Available model sizes: {list(FLANT5Finetuner.MODEL_SIZES.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ FLANT5Finetuner test failed: {e}")
        return False

def test_data_utils():
    """Test data utilities"""
    logger.info("Testing data utilities...")
    
    try:
        from data_utils import DataProcessor
        
        processor = DataProcessor()
        logger.info("✓ DataProcessor instantiated successfully")
        
        # Test sample data creation
        processor.create_sample_dataset("test_data.json", num_samples=5)
        logger.info("✓ Sample data creation successful")
        
        # Clean up
        if os.path.exists("test_data.json"):
            os.remove("test_data.json")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data utilities test failed: {e}")
        return False

def test_runpod_config():
    """Test RunPod configuration"""
    logger.info("Testing RunPod configuration...")
    
    try:
        from runpod_config import RunPodConfig
        
        config = RunPodConfig()
        logger.info("✓ RunPodConfig instantiated successfully")
        
        # Test command args generation
        args = config.get_command_args()
        logger.info(f"✓ Command args generated: {len(args)} arguments")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ RunPod configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting FLAN-T5 finetuning installation test...")
    logger.info("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Device Detection", test_device_detection),
        ("Model Loading", test_model_loading),
        ("Finetuner Class", test_finetuner_class),
        ("Data Utilities", test_data_utils),
        ("RunPod Configuration", test_runpod_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Installation is successful.")
        logger.info("\nYou can now use the FLAN-T5 finetuning codebase:")
        logger.info("• python flan_t5_finetune.py --help")
        logger.info("• python examples/train_translation.py")
        logger.info("• python examples/train_summarization.py")
    else:
        logger.error("❌ Some tests failed. Please check the installation.")
        sys.exit(1)

if __name__ == "__main__":
    main() 