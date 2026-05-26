@echo off
cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
set SILR_CODE_COMMIT=624e42274abadbc87c618443a65467e9b058f1ec
set SILR_CODE_DIRTY=1
set SILR_SYNC_ID=multi_action_n5_extra_20260524

D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 ^
  --model qwen3-14b ^
  --scenarios ^
    mined_multi_action_1_l0p25g1p0_s5 ^
    mined_multi_action_2_l1p0g1p0_s5 ^
    mined_multi_action_3_l0p25g1p0_s12 ^
  --policies progress progress_mag ^
  --reps 2 ^
  --rep-start-seed 1003 ^
  --max-steps 6 ^
  --request-timeout-s 240 ^
  --max-retries 0 ^
  --output eval_multi_action_n5_extra_gpu0.json ^
  --log-file logs\eval_multi_action_n5_extra_gpu0.log
if errorlevel 1 exit /b %errorlevel%
