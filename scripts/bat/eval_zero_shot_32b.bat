@echo off
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\eval_cluster_v2023.py ^
    --scenario-dir domains\cluster_v2023\scenarios\data ^
    --model D:\zcy\models\Qwen3-32B ^
    --repeats 1 ^
    --out outputs\cluster_v2023\zero_shot_32b.json
