#!/bin/bash
#SBATCH --job-name=run_app
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/output_%j.txt
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=193000
#SBATCH --cpus-per-gpu=15

module add devel/cuda/12.4

CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --port 8001 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --limit-mm-per-prompt image=10,video=1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 26000 \
    --max-num-seqs 8 \
    --swap-space 40 \
    --disable-log-stats
