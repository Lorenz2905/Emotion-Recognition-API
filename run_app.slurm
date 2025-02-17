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
module add devel/cuda/12.4

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --port 8001 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --limit-mm-per-prompt image=10,video=1 \
    --max-model-len 64000 &

VLLM_PID=$!

sleep 90
echo "⏳ Waiting for vLLM server..."
while ! curl -s http://localhost:8001/v1/models > /dev/null; do
    sleep 10
    echo "🔄 vLLM server is still loading..."
done
echo "✅ vLLM server is ready!"

cd /home/hk-project-starter-p0022892/uwlat/Emotion-Recognition-API/emotion_recognition_api

torchrun main.py &

wait $VLLM_PID
