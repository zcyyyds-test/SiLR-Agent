@echo off
cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
set SILR_MAX_TOKENS=768
set SILR_SCALAR_PROGRESS_RELATIVE_SLACK=0.20

D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_mpc_from_scalar_plateau.py ^
  --base-url http://127.0.0.1:8003/v1 ^
  --model qwen3-14b ^
  --scenarios mined_multi_action_3_l0p25g1p0_s12 ^
  --seeds 1000 1001 1002 ^
  --max-proposals 3 ^
  --request-timeout-s 240 ^
  --max-retries 0 ^
  --output eval_mpc_from_scalar_plateau_gpu0.json ^
  --log-file logs\eval_mpc_from_scalar_plateau_gpu0.log
