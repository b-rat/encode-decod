#!/usr/bin/env python3
"""
RunPod Template for FLAN-T5 Finetuning
This template can be deployed on RunPod for remote execution
"""

import runpod
import json
import logging
import os
from pathlib import Path
from flan_t5_finetune import FLANT5Finetuner
from runpod_config import RunPodConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handler(job):
    """
    RunPod job handler for FLAN-T5 finetuning
    
    Expected job input format:
    {
        "mode": "train" | "generate",
        "model_size": "small" | "base" | "large" | "xl" | "xxl",
        "device": "cpu" | "cuda" | "mps" | "auto",
        "data_path": "path/to/data.json",  # Required for training
        "eval_data_path": "path/to/eval_data.json",  # Optional for training
        "output_dir": "/workspace/output",
        "checkpoint_dir": "/workspace/checkpoints",
        "batch_size": 4,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "max_length": 512,
        "warmup_steps": 100,
        "gradient_accumulation_steps": 4,
        "use_lora": true,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "use_8bit": false,
        "use_4bit": false,
        "wandb_project": "flan-t5-finetune",
        "save_checkpoint": "checkpoint_name",  # Optional
        "resume_from": "checkpoint_name",  # Optional
        "input_text": "Text to generate from",  # Required for generation
        "max_generate_length": 100,
        "num_beams": 4
    }
    """
    try:
        # Parse job input
        job_input = job["input"]
        logger.info(f"Received job: {job_input}")
        
        # Extract mode
        mode = job_input.get("mode", "train")
        
        if mode == "train":
            return handle_training(job_input)
        elif mode == "generate":
            return handle_generation(job_input)
        else:
            return {
                "status": "error",
                "message": f"Invalid mode: {mode}. Must be 'train' or 'generate'"
            }
    
    except Exception as e:
        logger.error(f"Job failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "traceback": str(e.__traceback__)
        }

def handle_training(job_input):
    """Handle training mode"""
    logger.info("Starting training mode")
    
    # Validate required parameters
    required_params = ["data_path"]
    for param in required_params:
        if param not in job_input:
            return {
                "status": "error",
                "message": f"Missing required parameter: {param}"
            }
    
    # Initialize finetuner with job parameters
    finetuner = FLANT5Finetuner(
        model_size=job_input.get("model_size", "base"),
        device=job_input.get("device", "auto"),
        max_length=job_input.get("max_length", 512),
        batch_size=job_input.get("batch_size", 4),
        learning_rate=job_input.get("learning_rate", 5e-5),
        num_epochs=job_input.get("num_epochs", 3),
        warmup_steps=job_input.get("warmup_steps", 100),
        gradient_accumulation_steps=job_input.get("gradient_accumulation_steps", 4),
        use_lora=job_input.get("use_lora", True),
        lora_r=job_input.get("lora_r", 16),
        lora_alpha=job_input.get("lora_alpha", 32),
        lora_dropout=job_input.get("lora_dropout", 0.1),
        use_8bit=job_input.get("use_8bit", False),
        use_4bit=job_input.get("use_4bit", False),
        output_dir=job_input.get("output_dir", "/workspace/flan_t5_output"),
        checkpoint_dir=job_input.get("checkpoint_dir", "/workspace/checkpoints"),
        wandb_project=job_input.get("wandb_project"),
        seed=job_input.get("seed", 42)
    )
    
    # Load model and tokenizer
    load_checkpoint = job_input.get("load_checkpoint")
    if load_checkpoint:
        finetuner.load_checkpoint(load_checkpoint)
    else:
        finetuner.load_model_and_tokenizer()
    
    # Prepare datasets
    train_dataset = finetuner.prepare_dataset(job_input["data_path"])
    eval_dataset = None
    if job_input.get("eval_data_path"):
        eval_dataset = finetuner.prepare_dataset(job_input["eval_data_path"])
    
    # Setup training
    finetuner.setup_training(train_dataset, eval_dataset)
    
    # Train
    resume_from = job_input.get("resume_from")
    finetuner.train(resume_from_checkpoint=resume_from)
    
    # Save checkpoint if specified
    save_checkpoint = job_input.get("save_checkpoint")
    if save_checkpoint:
        finetuner.save_checkpoint(save_checkpoint)
    
    # Get available checkpoints
    checkpoints = finetuner.get_available_checkpoints()
    
    return {
        "status": "success",
        "message": "Training completed successfully",
        "output_dir": str(finetuner.output_dir),
        "checkpoint_dir": str(finetuner.checkpoint_dir),
        "available_checkpoints": checkpoints
    }

def handle_generation(job_input):
    """Handle generation mode"""
    logger.info("Starting generation mode")
    
    # Validate required parameters
    if "input_text" not in job_input:
        return {
            "status": "error",
            "message": "Missing required parameter: input_text"
        }
    
    # Initialize finetuner
    finetuner = FLANT5Finetuner(
        model_size=job_input.get("model_size", "base"),
        device=job_input.get("device", "auto"),
        max_length=job_input.get("max_length", 512),
        use_lora=job_input.get("use_lora", True),
        lora_r=job_input.get("lora_r", 16),
        lora_alpha=job_input.get("lora_alpha", 32),
        lora_dropout=job_input.get("lora_dropout", 0.1),
        use_8bit=job_input.get("use_8bit", False),
        use_4bit=job_input.get("use_4bit", False),
        output_dir=job_input.get("output_dir", "/workspace/flan_t5_output"),
        checkpoint_dir=job_input.get("checkpoint_dir", "/workspace/checkpoints"),
        seed=job_input.get("seed", 42)
    )
    
    # Load model and tokenizer
    load_checkpoint = job_input.get("load_checkpoint")
    if load_checkpoint:
        finetuner.load_checkpoint(load_checkpoint)
    else:
        finetuner.load_model_and_tokenizer()
    
    # Generate text
    generated_text = finetuner.generate(
        job_input["input_text"],
        max_length=job_input.get("max_generate_length", 100),
        num_beams=job_input.get("num_beams", 4)
    )
    
    return {
        "status": "success",
        "message": "Generation completed successfully",
        "input_text": job_input["input_text"],
        "generated_text": generated_text
    }

def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "FLAN-T5 finetuning service is running",
        "available_models": list(FLANT5Finetuner.MODEL_SIZES.keys())
    }

# Register the handler with RunPod
runpod.serverless.start({
    "handler": handler,
    "health_check": health_check
}) 