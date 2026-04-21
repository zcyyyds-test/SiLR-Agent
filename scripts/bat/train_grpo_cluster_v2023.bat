@echo off
rem GRPO iter 1 — uses SFT adapter as starting point.
rem Run grpo_sanity_cluster_v2023.bat FIRST and confirm ratio ≈ 1.0 before launching.
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
rem Flag names mapped to real upstream train_grpo.py CLI:
rem   --base-model (NOT --model) / --sft-adapter / --output (NOT --output-dir)
rem   --iterations (NOT --iter-count) / --clip-eps (NOT --advantage-clip)
rem   --batch-size 2 (upstream manual per-sample backward OOMs at 16 for 14B+LoRA)
python -u scripts\train_grpo_cluster_v2023.py ^
    --base-model D:\zcy\models\Qwen\Qwen3-14B ^
    --sft-adapter outputs\cluster_v2023\sft_adapter\final ^
    --output outputs\cluster_v2023\grpo_iter1 ^
    --iterations 1 ^
    --rollouts-per-scenario 2 ^
    --clip-eps 0.2 ^
    --kl-coeff 0.02 ^
    --lr 1e-6 ^
    --batch-size 2 ^
    --max-steps 15 ^
    --step-cost 0.00
