@echo off
rem GRPO iter 3 — uses iter 2 final adapter.
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\train_grpo_cluster_v2023.py ^
    --base-model D:\zcy\models\Qwen3-14B ^
    --sft-adapter outputs\cluster_v2023\grpo_iter2 ^
    --output outputs\cluster_v2023\grpo_iter3 ^
    --iterations 1 ^
    --rollouts-per-scenario 2 ^
    --clip-eps 0.2 ^
    --kl-coeff 0.02 ^
    --lr 1e-6 ^
    --batch-size 2 ^
    --max-steps 15 ^
    --step-cost 0.00
