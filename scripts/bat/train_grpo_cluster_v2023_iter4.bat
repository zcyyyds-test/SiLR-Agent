@echo off
rem GRPO iter 4 — per-token mean log_prob + system-prompt-in-scoring fixes.
rem iter3 showed log-ratio mean=21 max=35 clamp=414/474 (87%) — symptom of
rem sum-log-prob accumulating 100+ token drift per PPO step. iter4 uses
rem per-token mean so ratio stays in sensible range. Also adds system role
rem to policy scoring messages (prior: scored on [user, assistant] but
rem rollouts generated with [system, user] — context mismatch).
rem
rem Re-initializes from v10 SFT (not iter3 — iter3 policy may be noise-
rem polluted by over-clipped PPO updates). Codex's high-confidence params:
rem   lr 3e-6 -> 1e-6  (pull back after iter3 over-stepped)
rem   kl 0.01 -> 0.05  (stronger penalty against large policy drift)
rem   clip_eps 0.2 -> 0.1  (tighter PPO clip)
rem   rollouts kept at 2 (wall-clock budget)
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\train_grpo_cluster_v2023.py ^
    --base-model D:\zcy\models\Qwen\Qwen3-14B ^
    --sft-adapter outputs\cluster_v2023\sft_adapter\final ^
    --output outputs\cluster_v2023\grpo_iter4 ^
    --iterations 1 ^
    --rollouts-per-scenario 2 ^
    --clip-eps 0.1 ^
    --kl-coeff 0.05 ^
    --lr 1e-6 ^
    --batch-size 2 ^
    --max-steps 15 ^
    --step-cost 0.00 ^
    > outputs\cluster_v2023\train_grpo_iter4.log 2>&1
