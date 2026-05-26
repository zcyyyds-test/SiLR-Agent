@echo off
REM Multi-model robustness — Qwen3-32B on GPU1:8004
REM multi_3 x {terminal, progress_mag} x 3 reps = 6 episodes
cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=1
set SILR_MAX_TOKENS=2048
set SILR_SCALAR_PROGRESS_RELATIVE_SLACK=0.20

D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8004/v1 --model qwen3-32b ^
  --scenarios mined_multi_action_3_l0p25g1p0_s12 ^
  --policies terminal progress_mag ^
  --reps 3 --rep-start-seed 1000 ^
  --max-steps 6 --max-proposals 3 ^
  --request-timeout-s 240 --max-retries 0 ^
  --output eval_multimodel_32b_gpu1.json ^
  --log-file logs\eval_multimodel_32b_gpu1.log

echo === 32B multi-model done ===
