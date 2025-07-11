# FLAN-T5 Finetuning Codebase

A comprehensive codebase for finetuning FLAN-T5 encoder-decoder models with support for multiple model sizes, device options (CPU/GPU/MPS), and RunPod deployment.

## Features

- **Multiple Model Sizes**: Support for FLAN-T5 small, base, large, xl, and xxl
- **Device Flexibility**: Run on CPU, GPU (CUDA), or MPS (Apple Silicon)
- **Training & Generation Modes**: Train custom models or generate text with trained models
- **Checkpointing**: Save and resume training progress
- **Progress Monitoring**: Real-time training progress with detailed logging
- **LoRA Support**: Efficient finetuning with Low-Rank Adaptation
- **Quantization**: 8-bit and 4-bit quantization for memory efficiency
- **RunPod Ready**: Easy deployment on RunPod for remote execution
- **Multiple Data Formats**: Support for JSON, CSV, JSONL, and TSV datasets

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd flan-t5-finetune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create sample data (optional):
```bash
python data_utils.py
```

## Quick Start

### Training a Model

```bash
python flan_t5_finetune.py \
    --mode train \
    --model_size base \
    --data_path data/sample_training_data.json \
    --eval_data_path data/sample_eval_data.json \
    --output_dir ./output \
    --batch_size 4 \
    --num_epochs 3 \
    --use_lora \
    --device auto
```

### Generating Text

```bash
python flan_t5_finetune.py \
    --mode generate \
    --model_size base \
    --load_checkpoint checkpoint_name \
    --input_text "Translate this to French: Hello, how are you?" \
    --max_generate_length 50
```

## Model Sizes

| Size | Parameters | Memory (FP16) | Recommended Use |
|------|------------|---------------|-----------------|
| small | 80M | ~0.5GB | Quick experiments, limited resources |
| base | 250M | ~1.5GB | Good balance of performance/speed |
| large | 780M | ~4GB | Better performance, more memory |
| xl | 3B | ~15GB | High performance, requires GPU |
| xxl | 11B | ~50GB | Best performance, requires multiple GPUs |

## Device Support

### CPU
```bash
--device cpu
```
- Slowest but works everywhere
- No GPU memory requirements
- Good for small models and testing

### GPU (CUDA)
```bash
--device cuda
```
- Fastest training and inference
- Requires NVIDIA GPU
- Automatic detection with `--device auto`

### MPS (Apple Silicon)
```bash
--device mps
```
- Good performance on Apple M1/M2/M3
- Automatic detection with `--device auto`
- Limited to Apple Silicon Macs

## Training Configuration

### Basic Training
```bash
python flan_t5_finetune.py \
    --mode train \
    --model_size base \
    --data_path your_data.json \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --use_lora
```

### Advanced Training with Quantization
```bash
python flan_t5_finetune.py \
    --mode train \
    --model_size large \
    --data_path your_data.json \
    --batch_size 4 \
    --use_8bit \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --gradient_accumulation_steps 8
```

### Training with Checkpointing
```bash
# Start training
python flan_t5_finetune.py \
    --mode train \
    --model_size base \
    --data_path your_data.json \
    --save_checkpoint my_model_v1

# Resume training
python flan_t5_finetune.py \
    --mode train \
    --model_size base \
    --data_path your_data.json \
    --resume_from my_model_v1
```

## Data Format

The codebase supports multiple data formats. Your dataset should have input and output columns:

### JSON Format
```json
[
  {
    "input": "Translate this to French: Hello world",
    "output": "Traduisez ceci en français: Bonjour le monde"
  },
  {
    "input": "Summarize this text: Long article content...",
    "output": "This article discusses..."
  }
]
```

### CSV Format
```csv
input,output
"Translate this to French: Hello world","Traduisez ceci en français: Bonjour le monde"
"Summarize this text: Long article content...","This article discusses..."
```

### Supported Column Names
- **Input columns**: `input`, `question`, `text`, `source`, `context`
- **Output columns**: `output`, `answer`, `summary`, `target`, `response`

## RunPod Deployment

### 1. Create RunPod Template

Create a `runpod_template.py` file:

```python
import runpod

def handler(job):
    """RunPod job handler"""
    job_input = job["input"]
    
    # Your training/generation logic here
    # Use the FLANT5Finetuner class
    
    return {"status": "success", "output": "Training completed"}

runpod.serverless.start({"handler": handler})
```

### 2. Deploy to RunPod

1. Upload your code to RunPod
2. Set environment variables
3. Configure GPU requirements
4. Deploy the template

### 3. API Usage

```python
import requests

# Start training
response = requests.post(
    "https://your-runpod-endpoint.runpod.net",
    json={
        "input": {
            "mode": "train",
            "model_size": "base",
            "data_path": "/workspace/data/training_data.json",
            "num_epochs": 5,
            "use_lora": True
        }
    }
)

# Generate text
response = requests.post(
    "https://your-runpod-endpoint.runpod.net",
    json={
        "input": {
            "mode": "generate",
            "load_checkpoint": "my_model_v1",
            "input_text": "Translate this to French: Hello world"
        }
    }
)
```

## Configuration Files

### RunPod Configuration
Create a `runpod_config.json` file:

```json
{
  "model_size": "base",
  "device": "auto",
  "mode": "train",
  "data_path": "/workspace/data/training_data.json",
  "eval_data_path": "/workspace/data/eval_data.json",
  "output_dir": "/workspace/flan_t5_output",
  "checkpoint_dir": "/workspace/checkpoints",
  "batch_size": 8,
  "learning_rate": 5e-5,
  "num_epochs": 5,
  "use_lora": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "wandb_project": "flan-t5-finetune"
}
```

## Advanced Features

### LoRA Configuration
LoRA (Low-Rank Adaptation) enables efficient finetuning:

```bash
--use_lora \
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.1
```

### Quantization
Reduce memory usage with quantization:

```bash
# 8-bit quantization
--use_8bit

# 4-bit quantization (requires bitsandbytes)
--use_4bit
```

### Weights & Biases Integration
Track experiments with W&B:

```bash
--wandb_project "flan-t5-experiments"
```

### Custom Training Arguments
```bash
--warmup_steps 100 \
--gradient_accumulation_steps 4 \
--save_steps 500 \
--eval_steps 500 \
--logging_steps 100
```

## Monitoring and Logging

### Training Progress
The training script provides detailed progress information:
- Loss curves
- Learning rate schedules
- Memory usage
- GPU utilization
- Checkpoint saves

### Log Files
- `flan_t5_finetune.log`: Main training log
- `output/logs/`: TensorBoard logs
- `output/`: Model checkpoints and final model

### Checkpoint Management
```python
from flan_t5_finetune import FLANT5Finetuner

finetuner = FLANT5Finetuner()
checkpoints = finetuner.get_available_checkpoints()
print(f"Available checkpoints: {checkpoints}")
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size: `--batch_size 2`
   - Use quantization: `--use_8bit` or `--use_4bit`
   - Use smaller model: `--model_size small`

2. **Slow Training**
   - Use GPU: `--device cuda`
   - Increase batch size if memory allows
   - Use gradient accumulation: `--gradient_accumulation_steps 8`

3. **Poor Generation Quality**
   - Increase training epochs: `--num_epochs 10`
   - Use larger model: `--model_size large`
   - Improve training data quality

4. **Checkpoint Issues**
   - Ensure sufficient disk space
   - Check file permissions
   - Verify checkpoint directory exists

### Performance Tips

1. **Memory Optimization**
   - Use LoRA for large models
   - Enable quantization
   - Reduce batch size
   - Use gradient accumulation

2. **Speed Optimization**
   - Use GPU when available
   - Increase batch size if memory allows
   - Use mixed precision training (automatic with GPU)

3. **Quality Optimization**
   - Use larger models when possible
   - Increase training epochs
   - Improve data quality and quantity
   - Use validation set for early stopping

## Examples

### Translation Task
```bash
python flan_t5_finetune.py \
    --mode train \
    --model_size base \
    --data_path translation_data.json \
    --num_epochs 5 \
    --use_lora
```

### Summarization Task
```bash
python flan_t5_finetune.py \
    --mode train \
    --model_size large \
    --data_path summarization_data.json \
    --max_length 1024 \
    --batch_size 4 \
    --use_lora
```

### Question Answering
```bash
python flan_t5_finetune.py \
    --mode train \
    --model_size base \
    --data_path qa_data.json \
    --num_epochs 3 \
    --use_lora
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Open an issue on GitHub
4. Check the documentation

## Acknowledgments

- Google for the FLAN-T5 model
- Hugging Face for the Transformers library
- Microsoft for LoRA implementation
- The open-source community for various tools and libraries 