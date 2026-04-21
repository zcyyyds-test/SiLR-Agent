@echo off
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\eval_cluster_v2023.py ^
    --scenario-dir domains\cluster_v2023\scenarios\data ^
    --model D:\zcy\models\Qwen\Qwen3-14B ^
    --adapter outputs\cluster_v2023\grpo_iter3\final ^
    --repeats 3 ^
    --out outputs\cluster_v2023\eval_grpo.json
