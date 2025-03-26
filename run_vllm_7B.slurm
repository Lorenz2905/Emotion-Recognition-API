#!/bin/bash
#SBATCH --job-name=run_app
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/output_%j.txt
#SBATCH --partition=dev_accelerated
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=125400
#SBATCH --cpus-per-gpu=36

module add devel/cuda/12.4

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8001 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --limit-mm-per-prompt image=10,video=1 \
    --max-model-len 64000 
    
