#!/bin/bash
#SBATCH --job-name=run_app
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/output_%j.txt
#SBATCH --partition=dev_accelerated
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=125400
#SBATCH --cpus-per-gpu=20
module add devel/cuda

torchrun emotion-recognition-API/main.py