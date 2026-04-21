@echo off
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
rem scripts/train_sft.py uses --data-path / --model-path (not --data / --model-name)
rem Input file MUST be JSON array (train_sft.py uses json.load); run
rem enrich_cluster_v2023_sft.py --final-json to convert from JSONL first.
python -u scripts\train_sft.py ^
    --data-path outputs\cluster_v2023\sft_data_v2023.json ^
    --model-path D:\zcy\models\Qwen\Qwen3-14B ^
    --output-dir outputs\cluster_v2023\sft_adapter ^
    --epochs 3 ^
    --lr 2e-4 ^
    --batch-size 1 ^
    --grad-accum 8
