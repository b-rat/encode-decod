#!/usr/bin/env python3
"""
FLAN-T5 Finetuning Script
Supports multiple model sizes, training/generation modes, and device options (CPU/GPU/MPS)
Suitable for RunPod deployment
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flan_t5_finetune.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FLANT5Finetuner:
    """FLAN-T5 Finetuning class with support for multiple model sizes and devices"""
    
    # Available FLAN-T5 model sizes
    MODEL_SIZES = {
        'small': 'google/flan-t5-small',      # 80M parameters
        'base': 'google/flan-t5-base',        # 250M parameters
        'large': 'google/flan-t5-large',      # 780M parameters
        'xl': 'google/flan-t5-xl',            # 3B parameters
        'xxl': 'google/flan-t5-xxl',          # 11B parameters
    }
    
    def __init__(
        self,
        model_size: str = 'base',
        device: str = 'auto',
        max_length: int = 512,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        save_total_limit: int = 3,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_8bit: bool = False,
        use_4bit: bool = False,
        output_dir: str = './flan_t5_output',
        checkpoint_dir: str = './checkpoints',
        wandb_project: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize FLAN-T5 finetuner
        
        Args:
            model_size: Size of FLAN-T5 model ('small', 'base', 'large', 'xl', 'xxl')
            device: Device to use ('cpu', 'cuda', 'mps', 'auto')
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for optimizer
            gradient_accumulation_steps: Gradient accumulation steps
            save_steps: Steps between model saves
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
            save_total_limit: Maximum number of checkpoints to save
            use_lora: Whether to use LoRA for efficient finetuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            use_8bit: Whether to use 8-bit quantization
            use_4bit: Whether to use 4-bit quantization
            output_dir: Output directory for model and logs
            checkpoint_dir: Directory for checkpoints
            wandb_project: Weights & Biases project name
            seed: Random seed for reproducibility
        """
        self.model_size = model_size
        self.device = self._setup_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.save_total_limit = save_total_limit
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.wandb_project = wandb_project
        self.seed = seed
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        logger.info(f"Initialized FLAN-T5 finetuner with model size: {model_size}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _setup_device(self, device: str) -> str:
        """Setup device for training/inference"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        logger.info(f"Using device: {device}")
        if device == 'cuda':
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif device == 'mps':
            logger.info("Using MPS (Apple Silicon)")
        
        return device
    
    def load_model_and_tokenizer(self, model_name: Optional[str] = None):
        """Load FLAN-T5 model and tokenizer"""
        if model_name is None:
            model_name = self.MODEL_SIZES[self.model_size]
        
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        model_kwargs = {}
        if self.use_8bit:
            model_kwargs['load_in_8bit'] = True
        elif self.use_4bit:
            model_kwargs['load_in_4bit'] = True
            model_kwargs['bnb_4bit_compute_dtype'] = torch.float16
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
            **model_kwargs
        )
        
        # Apply LoRA if specified
        if self.use_lora:
            logger.info("Applying LoRA configuration")
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q", "v"]
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def prepare_dataset(self, data_path: str, task_type: str = "text2text") -> Dataset:
        """
        Prepare dataset for training
        
        Args:
            data_path: Path to dataset file (JSON, CSV, or HuggingFace dataset)
            task_type: Type of task ('text2text', 'question_answering', 'summarization')
        
        Returns:
            Prepared dataset
        """
        logger.info(f"Loading dataset from: {data_path}")
        
        # Load dataset
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
        elif data_path.endswith('.csv'):
            dataset = Dataset.from_csv(data_path)
        else:
            # Assume it's a HuggingFace dataset
            dataset = load_dataset(data_path, split='train')
        
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Handle different input formats
            if 'input' in examples and 'output' in examples:
                inputs = examples['input']
                targets = examples['output']
            elif 'question' in examples and 'answer' in examples:
                inputs = examples['question']
                targets = examples['answer']
            elif 'text' in examples and 'summary' in examples:
                inputs = examples['text']
                targets = examples['summary']
            else:
                # Assume first column is input, second is target
                columns = list(examples.keys())
                if len(columns) >= 2:
                    inputs = examples[columns[0]]
                    targets = examples[columns[1]]
                else:
                    raise ValueError("Dataset must have at least 2 columns (input and target)")
            
            # Tokenize inputs and targets
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info("Dataset tokenization completed")
        return tokenized_dataset
    
    def setup_training(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Setup training configuration"""
        logger.info("Setting up training configuration")
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.wandb_project else "none",
            run_name=f"flan-t5-{self.model_size}-finetune",
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            fp16=self.device != 'cpu',
            bf16=self.device == 'cuda' and torch.cuda.is_bf16_supported(),
            remove_unused_columns=False,
            push_to_hub=False,
            save_strategy="steps",
            logging_dir=str(self.output_dir / "logs"),
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None,
        )
        
        logger.info("Training configuration setup completed")
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Train the model"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call setup_training() first.")
        
        logger.info("Starting training...")
        
        # Initialize wandb if specified
        if self.wandb_project:
            wandb.init(project=self.wandb_project, name=f"flan-t5-{self.model_size}")
        
        try:
            # Train the model
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save the model
            self.trainer.save_model(str(self.output_dir / "final_model"))
            self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
            
            # Save training metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            
            logger.info(f"Training completed. Final loss: {metrics.get('train_loss', 'N/A')}")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Saving checkpoint...")
            self.trainer.save_model(str(self.output_dir / "interrupted_model"))
            self.tokenizer.save_pretrained(str(self.output_dir / "interrupted_model"))
            logger.info("Checkpoint saved. You can resume training later.")
        
        finally:
            if self.wandb_project:
                wandb.finish()
    
    def generate(self, input_text: str, max_length: int = 100, num_beams: int = 4) -> str:
        """Generate text from input"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            self.model.save_pretrained(str(checkpoint_path))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(checkpoint_path))
        
        # Save configuration
        config = {
            'model_size': self.model_size,
            'device': self.device,
            'max_length': self.max_length,
            'use_lora': self.use_lora,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'use_8bit': self.use_8bit,
            'use_4bit': self.use_4bit,
        }
        
        with open(checkpoint_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_name: str):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint {checkpoint_name} not found")
        
        # Load configuration
        config_path = checkpoint_path / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update configuration
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint_path))
        
        # Move to device
        self.model.to(self.device)
        
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
    
    def get_available_checkpoints(self) -> List[str]:
        """Get list of available checkpoints"""
        checkpoints = []
        if self.checkpoint_dir.exists():
            for item in self.checkpoint_dir.iterdir():
                if item.is_dir() and (item / 'config.json').exists():
                    checkpoints.append(item.name)
        return sorted(checkpoints)


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="FLAN-T5 Finetuning Script")
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="base", 
                       choices=["small", "base", "large", "xl", "xxl"],
                       help="FLAN-T5 model size")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["cpu", "cuda", "mps", "auto"],
                       help="Device to use for training/inference")
    
    # Training configuration
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train", "generate"],
                       help="Mode: train or generate")
    parser.add_argument("--data_path", type=str,
                       help="Path to training data (JSON, CSV, or HuggingFace dataset)")
    parser.add_argument("--eval_data_path", type=str,
                       help="Path to evaluation data")
    parser.add_argument("--output_dir", type=str, default="./flan_t5_output",
                       help="Output directory for model and logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory for checkpoints")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # LoRA configuration
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for efficient finetuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Quantization
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    
    # Generation parameters
    parser.add_argument("--input_text", type=str, help="Input text for generation")
    parser.add_argument("--max_generate_length", type=int, default=100,
                       help="Maximum generation length")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for generation")
    
    # Checkpointing
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint")
    parser.add_argument("--save_checkpoint", type=str, help="Save checkpoint with name")
    parser.add_argument("--load_checkpoint", type=str, help="Load checkpoint for generation")
    
    # Logging
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Initialize finetuner
    finetuner = FLANT5Finetuner(
        model_size=args.model_size,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        seed=args.seed
    )
    
    if args.mode == "train":
        # Training mode
        if not args.data_path:
            raise ValueError("--data_path is required for training mode")
        
        # Load model and tokenizer
        if args.load_checkpoint:
            finetuner.load_checkpoint(args.load_checkpoint)
        else:
            finetuner.load_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset = finetuner.prepare_dataset(args.data_path)
        eval_dataset = None
        if args.eval_data_path:
            eval_dataset = finetuner.prepare_dataset(args.eval_data_path)
        
        # Setup training
        finetuner.setup_training(train_dataset, eval_dataset)
        
        # Train
        finetuner.train(resume_from_checkpoint=args.resume_from)
        
        # Save checkpoint if specified
        if args.save_checkpoint:
            finetuner.save_checkpoint(args.save_checkpoint)
    
    elif args.mode == "generate":
        # Generation mode
        if not args.input_text:
            raise ValueError("--input_text is required for generation mode")
        
        # Load model and tokenizer
        if args.load_checkpoint:
            finetuner.load_checkpoint(args.load_checkpoint)
        else:
            finetuner.load_model_and_tokenizer()
        
        # Generate
        generated_text = finetuner.generate(
            args.input_text,
            max_length=args.max_generate_length,
            num_beams=args.num_beams
        )
        
        print(f"Input: {args.input_text}")
        print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main() 