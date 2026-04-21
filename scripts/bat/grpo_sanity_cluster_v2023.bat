@echo off
rem Pre-flight log_prob sanity check — RUN BEFORE iter 1.
rem Exit 0 = go. Non-zero = investigate before training.
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\grpo_sanity_check.py ^
    --base-model D:\zcy\models\Qwen\Qwen3-14B ^
    --sft-adapter outputs\cluster_v2023\sft_adapter\final
