@echo off
cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
set SILR_CODE_COMMIT=624e42274abadbc87c618443a65467e9b058f1ec
set SILR_CODE_DIRTY=1
set SILR_SYNC_ID=trajectory_stepbudget_v1_20260524

D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 ^
  --model qwen3-14b ^
  --scenarios ^
    mined_multi_action_1_l0p25g1p0_s5 ^
    mined_multi_action_3_l0p25g1p0_s12 ^
    mined_mpc_unsolved_2_l2p0g1p0_s20 ^
    mined_mpc_unsolved_3_l3p0g1p0_s24 ^
  --policies progress progress_mag ^
  --reps 3 ^
  --max-steps 6 ^
  --request-timeout-s 240 ^
  --max-retries 0 ^
  --output eval_trajectory_v1_gpu0.json ^
  --log-file logs\eval_trajectory_v1_gpu0.log
if errorlevel 1 exit /b %errorlevel%

D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 ^
  --model qwen3-14b ^
  --scenarios ^
    mined_mpc_unsolved_2_l2p0g1p0_s20 ^
    mined_mpc_unsolved_3_l3p0g1p0_s24 ^
  --policies progress_mag ^
  --reps 3 ^
  --max-steps 12 ^
  --request-timeout-s 240 ^
  --max-retries 0 ^
  --output eval_stepbudget12_v1_gpu0.json ^
  --log-file logs\eval_stepbudget12_v1_gpu0.log
if errorlevel 1 exit /b %errorlevel%
