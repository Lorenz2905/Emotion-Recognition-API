#!/bin/bash
#SBATCH --job-name=run_app
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/output_%j.txt
#SBATCH --partition=dev_accelerated
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=125400
#SBATCH --cpus-per-gpu=36
module add devel/cuda

cd /home/hk-project-starter-p0022892/uwlat/Emotion-Recognition-API/emotion_recognition_api
torchrun main.py