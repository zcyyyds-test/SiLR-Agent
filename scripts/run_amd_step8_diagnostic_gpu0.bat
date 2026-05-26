@echo off
REM Step-budget diagnostic: same canonical setup but max_steps=8 (vs default 6)
REM Tests whether m1's residual 1.07 plateau is a budget effect (recovers with 8 steps)
REM vs a true mechanism failure (stays at 1.07 even with more budget).
REM Only progress_mag policy (the one we care about for headline).
cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0
set SILR_MAX_TOKENS=2048
set SILR_SCALAR_PROGRESS_RELATIVE_SLACK=0.20

D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 --model qwen3-14b ^
  --scenarios mined_multi_action_1_l0p25g1p0_s5 mined_multi_action_2_l1p0g1p0_s5 mined_multi_action_3_l0p25g1p0_s12 ^
  --policies progress_mag ^
  --reps 3 --rep-start-seed 1000 ^
  --max-steps 8 --max-proposals 3 ^
  --request-timeout-s 240 --max-retries 0 ^
  --output eval_step8_progmag_gpu0.json ^
  --log-file logs\eval_step8_progmag_gpu0.log
echo === step-8 progress_mag diagnostic done ===
