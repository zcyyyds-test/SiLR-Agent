@echo off
REM m1 canonical rerun with FIXED react_loop.py (unified APPROVED feedback default)
REM 4 policies x 3 reps = 12 episodes, seeds 1000-1002
REM GPU0/8003 (independent of GPU1 v2 task)
REM Date: 2026-05-26
cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0
set SILR_MAX_TOKENS=2048
set SILR_SCALAR_PROGRESS_RELATIVE_SLACK=0.20
REM Do NOT set SILR_SAFE_PROGRESS_DISTINCT_FEEDBACK — default is unified APPROVED (fixed)

D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 --model qwen3-14b ^
  --scenarios mined_multi_action_1_l0p25g1p0_s5 ^
  --policies terminal progress progress_mag scalar_progress ^
  --reps 3 --rep-start-seed 1000 ^
  --max-steps 6 --max-proposals 3 ^
  --request-timeout-s 240 --max-retries 0 ^
  --output eval_m1_canonical_fixed_gpu0.json ^
  --log-file logs\eval_m1_canonical_fixed_gpu0.log

echo === m1 canonical rerun complete ===
