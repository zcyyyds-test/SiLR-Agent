@echo off
rem Stage 2: build 60 scenarios (vs original 25) for wider gpu_spec coverage.
rem Writes to a NEW dir scenarios/data_v2/ so original 25 at scenarios/data/
rem remain untouched and eval_{sft,grpo_iter3,grpo_iter4} stay comparable.
rem Keeps n_jobs=[2,2,3,4] distribution from inject_gpu_spec_mismatch (no
rem difficulty reduction — honesty over headline number). Expected:
rem   25->60: gpu_spec scenarios 5->12, teacher-solvable ~7 (vs ~2)
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\build_cluster_v2023_scenarios.py ^
    --raw-dir domains\cluster_v2023\data_pipeline\raw ^
    --out-dir domains\cluster_v2023\scenarios\data_v2 ^
    --n 60 ^
    --seed 42 ^
    > outputs\cluster_v2023\build_v2_scenarios.log 2>&1
