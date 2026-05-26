@echo off
cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 ^
  --model qwen3-14b ^
  --scenarios ^
    mined_single_action_1_l1p0g1p0_s18 ^
    mined_single_action_2_l3p0g1p0_s23 ^
    mined_single_action_3_l2p0g0p0_s20 ^
    mined_multi_action_1_l0p25g1p0_s5 ^
    mined_multi_action_2_l1p0g1p0_s5 ^
    mined_multi_action_3_l0p25g1p0_s12 ^
    mined_mpc_unsolved_1_l1p0g1p0_s4 ^
    mined_mpc_unsolved_2_l2p0g1p0_s20 ^
    mined_mpc_unsolved_3_l3p0g1p0_s24 ^
  --policies OFF terminal progress progress_mag ^
  --reps 3 ^
  --max-steps 6 ^
  --request-timeout-s 240 ^
  --max-retries 0 ^
  --output eval_mined_refresh_gpu0_v3.json ^
  --log-file logs\eval_mined_refresh_gpu0_v3.log
