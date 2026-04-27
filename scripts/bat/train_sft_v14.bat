@echo off
rem v14: same anonymized data as v13, 2 epochs (vs v13's 1).
rem v13 1-epoch ended at loss=0.51, token_acc=0.82 — far from convergence
rem (vs v12 raw-id 1 epoch token_acc=0.995). The anonymization made
rem next-token prediction strictly harder, so 1 epoch likely under-trains.
rem Risk: v11 (raw ids, 3 epochs) hit allocator thrash at ~step 700
rem (epoch 1.47); 2 epochs = 954 steps so we may hit the same wall.
rem Mitigation: save_strategy=epoch -> if thrash, fall back to checkpoint-477
rem (which is v13 final by another name).
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\train_sft_cluster_v2023.py ^
    --data-path outputs\cluster_v2023\sft_data_v2023_v13_per_step.json ^
    --model-path D:\zcy\models\Qwen\Qwen3-14B ^
    --output-dir outputs\cluster_v2023\sft_adapter_v14 ^
    --epochs 2 ^
    --max-seq-len 2048 ^
    --lr 2e-4 ^
    --batch-size 2 ^
    --grad-accum 4 ^
    > outputs\cluster_v2023\train_sft_v14.log 2>&1
