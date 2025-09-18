#!/bin/bash

# Set PyTorch memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Clear GPU memory
nvidia-smi --gpu-reset

# Run training with memory optimized config
python3 train_ete.py --config config/config_ete_memory_optimized.json
