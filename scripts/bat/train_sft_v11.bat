@echo off
rem Stage 2: SFT v11 on larger 60-scenario data.
rem Same hyperparams as v10 (the v9/v10 fixes for dtype ghost + reentrant
rem are already baked into scripts/train_sft_cluster_v2023.py).
rem Output dir: outputs/cluster_v2023/sft_adapter_v11/ (does NOT overwrite
rem the v10 baseline at outputs/cluster_v2023/sft_adapter/).
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\train_sft_cluster_v2023.py ^
    --data-path outputs\cluster_v2023\sft_data_v2023_v11_per_step.json ^
    --model-path D:\zcy\models\Qwen\Qwen3-14B ^
    --output-dir outputs\cluster_v2023\sft_adapter_v11 ^
    --epochs 3 ^
    --max-seq-len 2048 ^
    --lr 2e-4 ^
    --batch-size 2 ^
    --grad-accum 4 ^
    > outputs\cluster_v2023\train_sft_v11.log 2>&1
