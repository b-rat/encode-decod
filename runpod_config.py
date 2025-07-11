#!/usr/bin/env python3
"""
RunPod Configuration for FLAN-T5 Finetuning
This file provides configuration and utilities for running FLAN-T5 finetuning on RunPod
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunPodConfig:
    """Configuration class for RunPod deployment"""
    
    def __init__(self, config_path: str = "runpod_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load RunPod configuration"""
        default_config = {
            "model_size": "base",
            "device": "auto",
            "mode": "train",
            "data_path": None,
            "eval_data_path": None,
            "output_dir": "/workspace/flan_t5_output",
            "checkpoint_dir": "/workspace/checkpoints",
            "batch_size": 4,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "max_length": 512,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 4,
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "use_8bit": False,
            "use_4bit": False,
            "wandb_project": None,
            "seed": 42,
            "save_checkpoint": None,
            "resume_from": None,
            "input_text": None,
            "max_generate_length": 100,
            "num_beams": 4
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def save_config(self):
        """Save current configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {self.config_path}")
    
    def get_command_args(self) -> list:
        """Get command line arguments from configuration"""
        args = [
            "python", "flan_t5_finetune.py",
            "--model_size", self.config["model_size"],
            "--device", self.config["device"],
            "--mode", self.config["mode"],
            "--output_dir", self.config["output_dir"],
            "--checkpoint_dir", self.config["checkpoint_dir"],
            "--batch_size", str(self.config["batch_size"]),
            "--learning_rate", str(self.config["learning_rate"]),
            "--num_epochs", str(self.config["num_epochs"]),
            "--max_length", str(self.config["max_length"]),
            "--warmup_steps", str(self.config["warmup_steps"]),
            "--gradient_accumulation_steps", str(self.config["gradient_accumulation_steps"]),
            "--seed", str(self.config["seed"]),
            "--max_generate_length", str(self.config["max_generate_length"]),
            "--num_beams", str(self.config["num_beams"])
        ]
        
        # Add optional arguments
        if self.config["data_path"]:
            args.extend(["--data_path", self.config["data_path"]])
        
        if self.config["eval_data_path"]:
            args.extend(["--eval_data_path", self.config["eval_data_path"]])
        
        if self.config["use_lora"]:
            args.append("--use_lora")
            args.extend(["--lora_r", str(self.config["lora_r"])])
            args.extend(["--lora_alpha", str(self.config["lora_alpha"])])
            args.extend(["--lora_dropout", str(self.config["lora_dropout"])])
        
        if self.config["use_8bit"]:
            args.append("--use_8bit")
        
        if self.config["use_4bit"]:
            args.append("--use_4bit")
        
        if self.config["wandb_project"]:
            args.extend(["--wandb_project", self.config["wandb_project"]])
        
        if self.config["save_checkpoint"]:
            args.extend(["--save_checkpoint", self.config["save_checkpoint"]])
        
        if self.config["resume_from"]:
            args.extend(["--resume_from", self.config["resume_from"]])
        
        if self.config["input_text"]:
            args.extend(["--input_text", self.config["input_text"]])
        
        return args
    
    def run_training(self):
        """Run training with current configuration"""
        if self.config["mode"] != "train":
            raise ValueError("Configuration mode must be 'train' for training")
        
        if not self.config["data_path"]:
            raise ValueError("data_path must be specified for training")
        
        args = self.get_command_args()
        logger.info(f"Running training with command: {' '.join(args)}")
        
        try:
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            logger.info("Training completed successfully")
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e.stderr}")
            raise
    
    def run_generation(self, input_text: str):
        """Run generation with current configuration"""
        if self.config["mode"] != "generate":
            raise ValueError("Configuration mode must be 'generate' for generation")
        
        # Update input text
        self.config["input_text"] = input_text
        args = self.get_command_args()
        logger.info(f"Running generation with command: {' '.join(args)}")
        
        try:
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            logger.info("Generation completed successfully")
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Generation failed: {e.stderr}")
            raise


def create_runpod_handler():
    """Create RunPod handler for API endpoints"""
    
    def handler(job):
        """RunPod job handler"""
        try:
            # Parse job input
            job_input = job["input"]
            
            # Create config
            config = RunPodConfig()
            
            # Update config with job input
            for key, value in job_input.items():
                if key in config.config:
                    config.config[key] = value
            
            # Save updated config
            config.save_config()
            
            # Run based on mode
            if config.config["mode"] == "train":
                output = config.run_training()
                return {"status": "success", "output": output}
            
            elif config.config["mode"] == "generate":
                input_text = job_input.get("input_text", "")
                if not input_text:
                    return {"status": "error", "message": "input_text is required for generation"}
                
                output = config.run_generation(input_text)
                return {"status": "success", "output": output}
            
            else:
                return {"status": "error", "message": f"Invalid mode: {config.config['mode']}"}
        
        except Exception as e:
            logger.error(f"Job failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    return handler


if __name__ == "__main__":
    # Example usage
    config = RunPodConfig()
    
    # Example training configuration
    config.config.update({
        "mode": "train",
        "model_size": "base",
        "data_path": "/workspace/data/training_data.json",
        "eval_data_path": "/workspace/data/eval_data.json",
        "num_epochs": 5,
        "batch_size": 8,
        "use_lora": True,
        "wandb_project": "flan-t5-finetune"
    })
    
    config.save_config()
    print("RunPod configuration created successfully") 