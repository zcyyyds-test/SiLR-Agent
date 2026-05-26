#!/bin/bash
# Serve Qwen3-32B on GPU 1 / port 8004 (WSL).
# 32B fp16 ~64 GB on B6000 96 GB; util 0.90 leaves KV cache + cudagraph headroom.
set -e
cd /mnt/d/zcy/SILR-WISE26
mkdir -p logs
source /root/vllm-env/bin/activate
export CUDA_HOME=/root/vllm-env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1
exec vllm serve /mnt/d/zcy/models/Qwen3-32B \
    --host 0.0.0.0 \
    --port 8004 \
    --served-model-name qwen3-32b \
    --gpu-memory-utilization 0.78 \
    --max-model-len 4096 \
    --enforce-eager \
    --swap-space 0 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code
