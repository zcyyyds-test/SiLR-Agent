@echo off
rem Stage 2: collect trajectories on the 60-scenario set.
rem Runs after build_v2023_scenarios_v2.bat. Keeps seeds [0..7] so each
rem scenario gets 8 Best-fit replay trajectories for Best-fit tiebreak
rem diversity — same schedule as v10 collection.
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\collect_cluster_v2023_sft.py ^
    --scenario-dir domains\cluster_v2023\scenarios\data_v2 ^
    --out outputs\cluster_v2023\sft_data_v2023_v11.jsonl ^
    --seeds 0 1 2 3 4 5 6 7 ^
    > outputs\cluster_v2023\collect_sft_v11.log 2>&1
