@echo off
cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
set SILR_MAX_TOKENS=2048
set SILR_SCALAR_PROGRESS_RELATIVE_SLACK=0.20

D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8004/v1 ^
  --model qwen3-14b ^
  --scenarios mined_multi_action_1_l0p25g1p0_s5 ^
  --policies progress_mag ^
  --reps 1 ^
  --rep-start-seed 1000 ^
  --max-steps 6 ^
  --max-proposals 3 ^
  --request-timeout-s 240 ^
  --max-retries 0 ^
  --output eval_diag_gpu1_multi1_progressmag_tok2048.json ^
  --log-file logs\eval_diag_gpu1_multi1_progressmag_tok2048.log
