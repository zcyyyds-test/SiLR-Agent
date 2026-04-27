@echo off
REM Launcher for SWE-bench Lite SFT (QLoRA on Qwen3-14B base).
REM
REM Required env vars:
REM   REPO_ROOT     absolute path to this SILR-Agent checkout
REM   PY            absolute path to a python interpreter with torch+peft+trl
REM   MODEL_PATH    base model directory (e.g. ...\Qwen3-14B)
REM   SFT_DATA      JSONL or JSON of {"messages":[...]} records
REM   SFT_OUT       output adapter directory
REM Optional:
REM   CUDA_DEVICE   defaults to 1

setlocal
if "%CUDA_DEVICE%"=="" set CUDA_DEVICE=1
set CUDA_VISIBLE_DEVICES=%CUDA_DEVICE%

cd /d %REPO_ROOT%

%PY% -u scripts\train_swe_sft.py ^
    --model-path %MODEL_PATH% ^
    --data-path %SFT_DATA% ^
    --output-dir %SFT_OUT% ^
    --epochs 3 ^
    --batch-size 1 ^
    --grad-accum 8 ^
    --lr 2e-4 ^
    --lora-r 64 ^
    --lora-alpha 128
