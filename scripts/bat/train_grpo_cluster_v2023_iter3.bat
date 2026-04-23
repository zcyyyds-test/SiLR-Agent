@echo off
rem GRPO iter 3 (reward-fixed baseline)
rem Re-initializes from v10 SFT adapter (not grpo_iter2) so the delta is
rem directly comparable against SiLR-SFT 75%. Prior iter1/iter2 used an
rem inline reward with two latent bugs (wrong observation keys 'violations'/'fragmentation_F'
rem instead of 'viol'/'F', and missing gate_passes +1.0 bonus) — see
rem decisions-cluster-v2023.md Part 13. Those bugs are now fixed in
rem scripts/train_grpo_cluster_v2023.py::collect_rollouts.
rem
rem Hyperparameters tuned (relative to iter1/iter2):
rem   lr 1e-6 -> 3e-6  (larger step given reward signal is now real)
rem   kl 0.02 -> 0.01  (slight loosen; policy allowed to update faster)
rem   rollouts 2 kept  (wall-clock budget; can bump to 4 if this iter succeeds)
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\train_grpo_cluster_v2023.py ^
    --base-model D:\zcy\models\Qwen\Qwen3-14B ^
    --sft-adapter outputs\cluster_v2023\sft_adapter\final ^
    --output outputs\cluster_v2023\grpo_iter3 ^
    --iterations 1 ^
    --rollouts-per-scenario 2 ^
    --clip-eps 0.2 ^
    --kl-coeff 0.01 ^
    --lr 3e-6 ^
    --batch-size 2 ^
    --max-steps 15 ^
    --step-cost 0.00 ^
    > outputs\cluster_v2023\train_grpo_iter3.log 2>&1
