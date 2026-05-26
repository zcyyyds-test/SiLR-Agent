@echo off
REM 4-cell single-episode contrast for multi_1 progress_mag regression
REM Scenario: mined_multi_action_1_l0p25g1p0_s5, policy=progress_mag, seed=1000
REM GPU0 endpoint :8003 (independent of GPU1 v2 task on :8004)
REM Designed by panel synthesis 2026-05-25 22:45

cd /d D:\zcy\SILR-WISE26
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0

REM ---------- Cell A: unset cap + LEGACY feedback (simulates v3 hypothesis) ----------
set "SILR_MAX_TOKENS="
set SILR_LEGACY_SAFE_PROGRESS_FEEDBACK=1
echo === Cell A: unset cap + legacy feedback ===
D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 --model qwen3-14b ^
  --scenarios mined_multi_action_1_l0p25g1p0_s5 ^
  --policies progress_mag ^
  --reps 1 --rep-start-seed 1000 ^
  --max-steps 6 --max-proposals 3 ^
  --request-timeout-s 240 --max-retries 0 ^
  --output eval_multi1_4cell_A_uncap_legacy.json ^
  --log-file logs\eval_multi1_4cell_A_uncap_legacy.log

REM ---------- Cell B: cap=2048 + LEGACY feedback (H1 single variable) ----------
set SILR_MAX_TOKENS=2048
set SILR_LEGACY_SAFE_PROGRESS_FEEDBACK=1
echo === Cell B: cap=2048 + legacy feedback (H1 test) ===
D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 --model qwen3-14b ^
  --scenarios mined_multi_action_1_l0p25g1p0_s5 ^
  --policies progress_mag ^
  --reps 1 --rep-start-seed 1000 ^
  --max-steps 6 --max-proposals 3 ^
  --request-timeout-s 240 --max-retries 0 ^
  --output eval_multi1_4cell_B_cap_legacy.json ^
  --log-file logs\eval_multi1_4cell_B_cap_legacy.log

REM ---------- Cell C: unset cap + ADMITTED feedback (H2 single variable) ----------
set "SILR_MAX_TOKENS="
set "SILR_LEGACY_SAFE_PROGRESS_FEEDBACK="
echo === Cell C: unset cap + current admitted feedback (H2 test) ===
D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 --model qwen3-14b ^
  --scenarios mined_multi_action_1_l0p25g1p0_s5 ^
  --policies progress_mag ^
  --reps 1 --rep-start-seed 1000 ^
  --max-steps 6 --max-proposals 3 ^
  --request-timeout-s 240 --max-retries 0 ^
  --output eval_multi1_4cell_C_uncap_admitted.json ^
  --log-file logs\eval_multi1_4cell_C_uncap_admitted.log

REM ---------- Cell D: cap=2048 + ADMITTED feedback (v2 reproduction) ----------
set SILR_MAX_TOKENS=2048
set "SILR_LEGACY_SAFE_PROGRESS_FEEDBACK="
echo === Cell D: cap=2048 + admitted feedback (v2 repro) ===
D:\miniconda3\envs\pytorch_env\python.exe -u scripts\anm_eval_sweep.py ^
  --base-url http://127.0.0.1:8003/v1 --model qwen3-14b ^
  --scenarios mined_multi_action_1_l0p25g1p0_s5 ^
  --policies progress_mag ^
  --reps 1 --rep-start-seed 1000 ^
  --max-steps 6 --max-proposals 3 ^
  --request-timeout-s 240 --max-retries 0 ^
  --output eval_multi1_4cell_D_cap_admitted.json ^
  --log-file logs\eval_multi1_4cell_D_cap_admitted.log

echo === All 4 cells complete ===
