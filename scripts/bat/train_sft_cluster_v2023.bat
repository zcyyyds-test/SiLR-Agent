@echo off
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
rem scripts/train_sft.py uses --data-path / --model-path (not --data / --model-name)
rem Input file MUST be JSON array (train_sft.py uses json.load).
rem Using per-step split: each record is [system, user_t, assistant_t].
rem Full trajectories (27 msgs, ~100k tokens) could not fit max_seq_len=4096
rem and the first assistant target was truncated away, yielding 0% recovery
rem despite loss=0.07. See decisions-cluster-v2023.md Part 7 P7-4.
rem max_seq_len=16384 covers ~8k/single turn with safe margin.
python -u scripts\train_sft.py ^
    --data-path outputs\cluster_v2023\sft_data_v2023_per_step.json ^
    --model-path D:\zcy\models\Qwen\Qwen3-14B ^
    --output-dir outputs\cluster_v2023\sft_adapter ^
    --epochs 2 ^
    --max-seq-len 16384 ^
    --lr 2e-4 ^
    --batch-size 1 ^
    --grad-accum 8 ^
    > outputs\cluster_v2023\train_sft_v2.log 2>&1
