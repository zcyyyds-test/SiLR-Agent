@echo off
cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
set SILR_CODE_COMMIT=624e42274abadbc87c618443a65467e9b058f1ec
set SILR_CODE_DIRTY=1
set SILR_SYNC_ID=security_mini_progmag_20260525

D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_adversarial_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 ^
  --model qwen3-14b ^
  --scenarios ^
    mined_multi_action_3_l0p25g1p0_s12 ^
  --attacks none prompt_injection observation_poison stall stall_rag ^
  --gating-policy progress_mag ^
  --reps 3 ^
  --rep-start-seed 2000 ^
  --max-steps 6 ^
  --request-timeout-s 240 ^
  --max-retries 0 ^
  --output eval_security_mini_progmag_gpu0.json ^
  --log-file logs\eval_security_mini_progmag_gpu0.log
if errorlevel 1 exit /b %errorlevel%
