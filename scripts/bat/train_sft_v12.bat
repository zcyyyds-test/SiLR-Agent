@echo off
rem Stage 3: SFT v12 with new aff_run observation. ONLY 1 epoch — v11 hit
rem allocator thrashing at epoch 2 (230s/step), and 1-epoch checkpoint
rem already reached loss=0.013/token_acc=0.995. Avoid the thrash.
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\train_sft_cluster_v2023.py ^
    --data-path outputs\cluster_v2023\sft_data_v2023_v12_per_step.json ^
    --model-path D:\zcy\models\Qwen\Qwen3-14B ^
    --output-dir outputs\cluster_v2023\sft_adapter_v12 ^
    --epochs 1 ^
    --max-seq-len 2048 ^
    --lr 2e-4 ^
    --batch-size 2 ^
    --grad-accum 4 ^
    > outputs\cluster_v2023\train_sft_v12.log 2>&1
