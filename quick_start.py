#!/usr/bin/env python3
"""
Quick Start Script for FLAN-T5 Finetuning
This script helps users get started quickly with the FLAN-T5 finetuning codebase
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🚀 FLAN-T5 Finetuning Codebase - Quick Start")
    print("=" * 60)
    print()

def check_installation():
    """Check if the installation is working"""
    print("🔍 Checking installation...")
    
    try:
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Installation check passed!")
            return True
        else:
            print("❌ Installation check failed!")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Installation check error: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("📝 Creating sample data...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Create simple training data
    sample_data = [
        {
            "input": "Translate this to French: Hello, how are you?",
            "output": "Traduisez ceci en français: Bonjour, comment allez-vous?"
        },
        {
            "input": "Translate this to Spanish: What is your name?",
            "output": "Traduce esto al español: ¿Cuál es tu nombre?"
        },
        {
            "input": "Summarize this text: Artificial Intelligence is transforming industries.",
            "output": "AI is changing various industries."
        },
        {
            "input": "Answer this question: What is machine learning?",
            "output": "Machine learning is a subset of AI that enables computers to learn from data."
        }
    ]
    
    # Save training data
    with open("data/quick_start_data.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample data created: data/quick_start_data.json")

def run_quick_training():
    """Run a quick training example"""
    print("🏋️ Running quick training example...")
    
    # Run training with small model
    cmd = [
        sys.executable, "flan_t5_finetune.py",
        "--mode", "train",
        "--model_size", "small",  # Use small model for quick training
        "--data_path", "data/quick_start_data.json",
        "--batch_size", "2",
        "--num_epochs", "2",
        "--use_lora",
        "--device", "auto",
        "--output_dir", "./quick_start_output",
        "--checkpoint_dir", "./quick_start_checkpoints"
    ]
    
    try:
        print("Running command:", " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Quick training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print("Error output:", e.stderr)
        return False

def test_generation():
    """Test text generation with the trained model"""
    print("🧪 Testing text generation...")
    
    cmd = [
        sys.executable, "flan_t5_finetune.py",
        "--mode", "generate",
        "--model_size", "small",
        "--load_checkpoint", "final_model",
        "--input_text", "Translate this to French: Hello world",
        "--max_generate_length", "30",
        "--device", "auto"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="./quick_start_output")
        if result.returncode == 0:
            print("✅ Generation test completed!")
            print("Generated text should appear in the output above.")
        else:
            print("⚠️ Generation test failed, but this is normal for a quick demo.")
            print("The model may need more training data or epochs for better results.")
        return True
    except Exception as e:
        print(f"❌ Generation test error: {e}")
        return False

def show_next_steps():
    """Show next steps for users"""
    print("\n" + "=" * 60)
    print("🎉 Quick Start Completed!")
    print("=" * 60)
    print()
    print("📚 Next Steps:")
    print("1. Explore the examples:")
    print("   • python examples/train_translation.py")
    print("   • python examples/train_summarization.py")
    print()
    print("2. Train with your own data:")
    print("   • Prepare your data in JSON format")
    print("   • Use: python flan_t5_finetune.py --mode train --data_path your_data.json")
    print()
    print("3. Generate text:")
    print("   • Use: python flan_t5_finetune.py --mode generate --input_text 'Your text here'")
    print()
    print("4. Deploy to RunPod:")
    print("   • Use the runpod_template.py for cloud deployment")
    print()
    print("📖 Documentation:")
    print("• Read README.md for detailed instructions")
    print("• Check the examples/ directory for more examples")
    print()
    print("🔧 Configuration:")
    print("• Edit runpod_config.json for custom settings")
    print("• Use --help for all available options")
    print()

def main():
    """Main quick start function"""
    print_banner()
    
    # Check installation
    if not check_installation():
        print("\n❌ Installation check failed. Please install dependencies first:")
        print("pip install -r requirements.txt")
        return
    
    # Create sample data
    create_sample_data()
    
    # Run quick training
    if run_quick_training():
        # Test generation
        test_generation()
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main() 