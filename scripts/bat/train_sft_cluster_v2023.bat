@echo off
set CUDA_VISIBLE_DEVICES=0
rem Third-party diagnosis (Kimi + Codex, 2026-04-22) found the real
rem root cause of the linear +100s/step degradation was:
rem   P1: dtype=torch.bfloat16 alongside quantization_config kept a
rem       pre-quant bf16 copy of Qwen3-14B (~28 GB ghost VRAM)
rem   P2: gradient_checkpointing_kwargs={"use_reentrant": False}
rem       was missing; default use_reentrant=True breaks PEFT LoRA
rem       hooks -- per-step activation leak
rem Both fixes applied in fork scripts/train_sft_cluster_v2023.py
rem (shared train_sft.py left untouched per decisions D11 pattern).
rem Env vars kept from v4 for allocator safety margin.
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
rem Input must be JSON array; observation.py is compact (no cpu_milli/ram_mib).
rem seq_len 10240 is comfortable once the ghost bf16 copy and activation
rem leak are fixed -- single-turn sample is ~7000 tokens + chat template.
python -u scripts\train_sft_cluster_v2023.py ^
    --data-path outputs\cluster_v2023\sft_data_v2023_per_step.json ^
    --model-path D:\zcy\models\Qwen\Qwen3-14B ^
    --output-dir outputs\cluster_v2023\sft_adapter ^
    --epochs 3 ^
    --max-seq-len 2048 ^
    --lr 2e-4 ^
    --batch-size 2 ^
    --grad-accum 4 ^
    > outputs\cluster_v2023\train_sft_v9_noeval.log 2>&1
