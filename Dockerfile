# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the codebase
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/data \
    /workspace/output \
    /workspace/checkpoints \
    /workspace/logs

# Set environment variables
ENV PYTHONPATH=/workspace
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HF_HOME=/workspace/.cache/huggingface

# Expose port for RunPod (if needed)
EXPOSE 8000

# Create a non-root user
RUN useradd -m -u 1000 -s /bin/bash user
RUN chown -R user:user /workspace
USER user

# Default command
CMD ["python", "flan_t5_finetune.py", "--help"] 