@echo off
cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0
set SILR_MAX_TOKENS=768
set SILR_SCALAR_PROGRESS_RELATIVE_SLACK=0.20

D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 ^
  --model qwen3-14b ^
  --scenarios ^
    mined_multi_action_1_l0p25g1p0_s5 ^
    mined_multi_action_2_l1p0g1p0_s5 ^
    mined_multi_action_3_l0p25g1p0_s12 ^
    mined_multi_action_4_l1p0g1p0_s12 ^
    mined_multi_action_5_l0p25g1p0_s16 ^
    mined_multi_action_6_l1p0g1p0_s16 ^
    mined_multi_action_7_l0p25g1p0_s19 ^
    mined_multi_action_8_l1p0g1p0_s19 ^
  --policies terminal scalar_progress progress progress_mag ^
  --reps 3 ^
  --rep-start-seed 1000 ^
  --max-steps 6 ^
  --max-proposals 3 ^
  --request-timeout-s 240 ^
  --max-retries 0 ^
  --output eval_multi_action_expansion_gpu0_v1.json ^
  --log-file logs\eval_multi_action_expansion_gpu0_v1.log
