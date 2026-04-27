@echo off
rem GRPO iter 5 — init from v13 SFT (anonymized + aff_run obs), runs
rem under capacity-only verifier (domain config.py), reward dead-code +
rem per-token mean log_prob fixes already in place. Goal: push gpu_spec
rem 3/11 -> 5+/11 by exploring scenarios where teacher itself failed.
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\train_grpo_cluster_v2023.py ^
    --base-model D:\zcy\models\Qwen\Qwen3-14B ^
    --sft-adapter outputs\cluster_v2023\sft_adapter_v13\final ^
    --output outputs\cluster_v2023\grpo_iter5 ^
    --iterations 1 ^
    --rollouts-per-scenario 2 ^
    --clip-eps 0.1 ^
    --kl-coeff 0.05 ^
    --lr 1e-6 ^
    --batch-size 2 ^
    --max-steps 15 ^
    --step-cost 0.00 ^
    > outputs\cluster_v2023\train_grpo_iter5.log 2>&1
