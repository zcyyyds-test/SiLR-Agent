@echo off
rem Stage 2: split trajectories -> per-step samples, keep only-success so
rem student only learns teacher's winning moves. No more gpu_spec:15x
rem upsample — v10 showed it didn't help (same 0/5 result); rely on the
rem larger natural sample pool from n=60 scenarios instead.
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\split_trajectories_to_steps.py ^
    --input outputs\cluster_v2023\sft_data_v2023_v11.jsonl ^
    --output outputs\cluster_v2023\sft_data_v2023_v11_per_step.json ^
    --compress-user ^
    --only-success ^
    > outputs\cluster_v2023\split_sft_v11.log 2>&1
