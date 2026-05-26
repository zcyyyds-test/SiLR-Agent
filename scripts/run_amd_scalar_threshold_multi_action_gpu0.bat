@echo off
setlocal EnableExtensions

cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0
set SILR_SYNC_ID=scalar_threshold_multi_action_20260525
set SILR_MAX_TOKENS=768

call :run_threshold 0p00 0.00
if errorlevel 1 exit /b %ERRORLEVEL%
call :run_threshold 0p05 0.05
if errorlevel 1 exit /b %ERRORLEVEL%
call :run_threshold 0p10 0.10
if errorlevel 1 exit /b %ERRORLEVEL%
call :run_threshold 0p20 0.20
if errorlevel 1 exit /b %ERRORLEVEL%

exit /b 0

:run_threshold
set THRESHOLD_LABEL=%~1
set SILR_SCALAR_PROGRESS_RELATIVE_SLACK=%~2

D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 ^
  --model qwen3-14b ^
  --scenarios ^
    mined_multi_action_1_l0p25g1p0_s5 ^
    mined_multi_action_2_l1p0g1p0_s5 ^
    mined_multi_action_3_l0p25g1p0_s12 ^
  --policies scalar_progress ^
  --reps 3 ^
  --rep-start-seed 1000 ^
  --max-steps 6 ^
  --request-timeout-s 240 ^
  --max-retries 0 ^
  --output eval_scalar_threshold_%THRESHOLD_LABEL%_gpu0.json ^
  --log-file logs\eval_scalar_threshold_%THRESHOLD_LABEL%_gpu0.log

exit /b %ERRORLEVEL%
