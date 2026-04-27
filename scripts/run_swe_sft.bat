@echo off
setlocal
set CUDA_VISIBLE_DEVICES=1
set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

cd /d D:\zcy\SILR-Agent

C:\Users\Administrator\miniconda3\envs\pytorch_env\python.exe -u scripts\train_swe_sft.py ^
    --model-path D:\zcy\models\Qwen3-14B ^
    --data-path D:\zcy\silr-swe-cache\swe_sft.jsonl ^
    --output-dir D:\zcy\SILR-Agent\outputs\swe_sft_model ^
    --epochs 3 ^
    --batch-size 1 ^
    --grad-accum 8 ^
    --lr 2e-4 ^
    --lora-r 64 ^
    --lora-alpha 128
