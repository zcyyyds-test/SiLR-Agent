#!/bin/bash
# Serve Qwen3-14B on AMD WSL GPU 1 for SILR-WISE26 expansion evals.
set -e
cd /mnt/d/zcy/SILR-WISE26
mkdir -p logs
source /root/vllm-env/bin/activate
export CUDA_HOME=/root/vllm-env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1
exec vllm serve /mnt/d/zcy/models/Qwen3-14B \
    --host 0.0.0.0 \
    --port 8004 \
    --served-model-name qwen3-14b \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code
