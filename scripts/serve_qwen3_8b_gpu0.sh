#!/bin/bash
# Serve Qwen3-8B on GPU 0 (WSL Ubuntu-24.04) for SILR-WISE26 cross-scale eval.
# Mirrors /root/serve_qwen14b_gpu0.sh but swaps model + GPU 0.
set -e
source /root/vllm-env/bin/activate
export CUDA_HOME=/root/vllm-env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
vllm serve /mnt/d/zcy/models/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8002 \
    --served-model-name qwen3-8b \
    --gpu-memory-utilization 0.75 \
    --max-model-len 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code
