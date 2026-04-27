@echo off
REM Launcher for SWE-bench Lite inference on a single CUDA GPU.
REM
REM Required env vars:
REM   REPO_ROOT      absolute path to this SILR-Agent checkout
REM   PY             absolute path to a python interpreter with torch installed
REM   SWE_CACHE      directory holding swe-bench-lite.jsonl + repos/
REM   SWE_OUT        output predictions JSONL
REM   TRACK          14B-zs / 32B-zs / 14B-sft / 14B-sft-grpo / 14B-fewshot
REM   MODEL_PATH     base model checkpoint
REM   ADAPTER_PATH   LoRA adapter dir (or "NONE" for zero-shot)
REM Optional:
REM   CUDA_DEVICE    defaults to 1
REM   PIP_INDEX_URL  pypi mirror

setlocal
if "%CUDA_DEVICE%"=="" set CUDA_DEVICE=1
set CUDA_VISIBLE_DEVICES=%CUDA_DEVICE%

cd /d %REPO_ROOT%

set ADAPTER_FLAG=
if NOT "%ADAPTER_PATH%"=="NONE" set ADAPTER_FLAG=--adapter-path %ADAPTER_PATH%

%PY% -u scripts\eval_swe_inference.py ^
    --track %TRACK% ^
    --model-path %MODEL_PATH% ^
    %ADAPTER_FLAG% ^
    --manifest %SWE_CACHE%\swe-bench-lite.jsonl ^
    --repo-cache %SWE_CACHE%\repos ^
    --output %SWE_OUT%
