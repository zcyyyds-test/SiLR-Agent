@echo off
rem Stage 2 baseline: eval v10 SFT on the NEW 60-scenario set so SFT v11's
rem numbers have a comparable reference (original 25 scenarios -> new 60
rem changes the scenario distribution, especially gpu_spec 5->12).
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\eval_cluster_v2023.py ^
    --scenario-dir domains\cluster_v2023\scenarios\data_v2 ^
    --model D:\zcy\models\Qwen\Qwen3-14B ^
    --adapter outputs\cluster_v2023\sft_adapter\final ^
    --repeats 1 ^
    --out outputs\cluster_v2023\eval_sft_v10_on_data_v2.json ^
    > outputs\cluster_v2023\eval_sft_v10_on_data_v2.log 2>&1
