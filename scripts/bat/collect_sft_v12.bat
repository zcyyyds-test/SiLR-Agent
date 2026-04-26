@echo off
rem Stage 3: re-collect SFT trajectories on data_v2 with the new
rem aff_run observation field. v11 trajectories are stale (no aff_run).
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\collect_cluster_v2023_sft.py ^
    --scenario-dir domains\cluster_v2023\scenarios\data_v2 ^
    --out outputs\cluster_v2023\sft_data_v2023_v12.jsonl ^
    --seeds 0 1 2 3 4 5 6 7 ^
    > outputs\cluster_v2023\collect_sft_v12.log 2>&1
