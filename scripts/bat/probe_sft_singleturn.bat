@echo off
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\probe_sft_inference.py ^
    --model D:\zcy\models\Qwen\Qwen3-14B ^
    --adapter outputs\cluster_v2023\sft_adapter\final ^
    --sft-jsonl outputs\cluster_v2023\sft_data_v2023_base.jsonl ^
    --n-turns 3 ^
    > outputs\cluster_v2023\probe_sft_singleturn.log 2>&1
