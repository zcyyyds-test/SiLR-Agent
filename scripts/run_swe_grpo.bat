@echo off
REM Launcher for SWE-bench Lite GRPO post-training.
REM
REM Required env vars:
REM   REPO_ROOT     absolute path to this SILR-Agent checkout
REM   PY            absolute path to a python interpreter with torch+peft+trl
REM   MODEL_PATH    base model directory
REM   SFT_ADAPTER   SFT LoRA adapter directory (starting point for GRPO)
REM   SWE_CACHE     directory holding swe-bench-lite.jsonl + repos/
REM   GRPO_OUT      output directory for the GRPO-trained adapter
REM Optional:
REM   CUDA_DEVICE   defaults to 1

setlocal
if "%CUDA_DEVICE%"=="" set CUDA_DEVICE=1
set CUDA_VISIBLE_DEVICES=%CUDA_DEVICE%

cd /d %REPO_ROOT%

%PY% -u scripts\train_swe_grpo.py ^
    --model-path %MODEL_PATH% ^
    --sft-adapter %SFT_ADAPTER% ^
    --manifest %SWE_CACHE%\swe-bench-lite.jsonl ^
    --repo-cache %SWE_CACHE%\repos ^
    --output-dir %GRPO_OUT% ^
    --iters 3 ^
    --rollouts-per-instance 4 ^
    --lr 1e-6 ^
    --kl-coeff 0.02 ^
    --adv-clip 3.0 ^
    --batch-size 16
