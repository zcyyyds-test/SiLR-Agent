@echo off
setlocal
set CUDA_VISIBLE_DEVICES=1
set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

cd /d D:\zcy\SILR-Agent

C:\Users\Administrator\miniconda3\envs\pytorch_env\python.exe -u scripts\train_swe_grpo.py ^
    --model-path D:\zcy\models\Qwen3-14B ^
    --sft-adapter D:\zcy\SILR-Agent\outputs\swe_sft_model ^
    --manifest D:\zcy\silr-swe-cache\swe-bench-lite.jsonl ^
    --repo-cache D:\zcy\silr-swe-cache\repos ^
    --output-dir D:\zcy\SILR-Agent\outputs\swe_grpo_model ^
    --iters 3 ^
    --rollouts-per-instance 4 ^
    --lr 1e-6 ^
    --kl-coeff 0.02 ^
    --adv-clip 3.0 ^
    --batch-size 16
