@echo off
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\split_trajectories_to_steps.py ^
    --input outputs\cluster_v2023\sft_data_v2023_v12.jsonl ^
    --output outputs\cluster_v2023\sft_data_v2023_v12_per_step.json ^
    --compress-user ^
    --only-success ^
    > outputs\cluster_v2023\split_sft_v12.log 2>&1
