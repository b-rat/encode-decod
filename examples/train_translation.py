#!/usr/bin/env python3
"""
Example: Training FLAN-T5 for Translation Task
This script demonstrates how to finetune FLAN-T5 for translation tasks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flan_t5_finetune import FLANT5Finetuner
from data_utils import DataProcessor
import json

def create_translation_data():
    """Create sample translation data"""
    translation_data = [
        {
            "input": "Translate this to French: Hello, how are you?",
            "output": "Traduisez ceci en français: Bonjour, comment allez-vous?"
        },
        {
            "input": "Translate this to Spanish: What is your name?",
            "output": "Traduce esto al español: ¿Cuál es tu nombre?"
        },
        {
            "input": "Translate this to German: I love this movie.",
            "output": "Übersetze dies ins Deutsche: Ich liebe diesen Film."
        },
        {
            "input": "Translate this to Italian: The weather is beautiful today.",
            "output": "Traduci questo in italiano: Il tempo è bello oggi."
        },
        {
            "input": "Translate this to Portuguese: Can you help me?",
            "output": "Traduza isto para português: Você pode me ajudar?"
        },
        {
            "input": "Translate this to French: The restaurant is closed.",
            "output": "Traduisez ceci en français: Le restaurant est fermé."
        },
        {
            "input": "Translate this to Spanish: I need to study for the exam.",
            "output": "Traduce esto al español: Necesito estudiar para el examen."
        },
        {
            "input": "Translate this to German: She is a talented musician.",
            "output": "Übersetze dies ins Deutsche: Sie ist eine talentierte Musikerin."
        },
        {
            "input": "Translate this to Italian: The book is very interesting.",
            "output": "Traduci questo in italiano: Il libro è molto interessante."
        },
        {
            "input": "Translate this to Portuguese: We will meet tomorrow.",
            "output": "Traduza isto para português: Nós nos encontraremos amanhã."
        }
    ]
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Save translation data
    with open("data/translation_data.json", "w", encoding="utf-8") as f:
        json.dump(translation_data, f, indent=2, ensure_ascii=False)
    
    print("Translation data created successfully!")
    return "data/translation_data.json"

def train_translation_model():
    """Train FLAN-T5 for translation task"""
    
    # Create sample data
    data_path = create_translation_data()
    
    # Initialize finetuner for translation
    finetuner = FLANT5Finetuner(
        model_size="base",  # Good balance for translation
        device="auto",      # Automatically detect best device
        max_length=128,     # Shorter sequences for translation
        batch_size=4,       # Conservative batch size
        learning_rate=5e-5, # Standard learning rate
        num_epochs=5,       # More epochs for better translation
        warmup_steps=50,    # Shorter warmup for smaller dataset
        gradient_accumulation_steps=4,
        use_lora=True,      # Use LoRA for efficiency
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        output_dir="./translation_output",
        checkpoint_dir="./translation_checkpoints",
        wandb_project="flan-t5-translation"
    )
    
    print("Loading model and tokenizer...")
    finetuner.load_model_and_tokenizer()
    
    print("Preparing dataset...")
    train_dataset = finetuner.prepare_dataset(data_path)
    
    print("Setting up training...")
    finetuner.setup_training(train_dataset)
    
    print("Starting training...")
    finetuner.train()
    
    print("Training completed!")
    print(f"Model saved to: {finetuner.output_dir}")
    print(f"Checkpoints saved to: {finetuner.checkpoint_dir}")

def test_translation_model():
    """Test the trained translation model"""
    
    # Initialize finetuner
    finetuner = FLANT5Finetuner(
        model_size="base",
        device="auto",
        use_lora=True,
        output_dir="./translation_output",
        checkpoint_dir="./translation_checkpoints"
    )
    
    # Load the trained model
    try:
        finetuner.load_checkpoint("final_model")
        print("Loaded trained model successfully!")
    except:
        print("No trained model found. Loading base model...")
        finetuner.load_model_and_tokenizer()
    
    # Test translations
    test_inputs = [
        "Translate this to French: Hello, how are you?",
        "Translate this to Spanish: What is your name?",
        "Translate this to German: I love this movie.",
        "Translate this to Italian: The weather is beautiful today.",
        "Translate this to Portuguese: Can you help me?"
    ]
    
    print("\nTesting translations:")
    print("=" * 50)
    
    for input_text in test_inputs:
        generated_text = finetuner.generate(
            input_text,
            max_length=50,
            num_beams=4
        )
        print(f"Input: {input_text}")
        print(f"Output: {generated_text}")
        print("-" * 30)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FLAN-T5 for translation")
    parser.add_argument("--mode", choices=["train", "test", "both"], 
                       default="both", help="Mode to run")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        print("Training translation model...")
        train_translation_model()
    
    if args.mode in ["test", "both"]:
        print("\nTesting translation model...")
        test_translation_model() 