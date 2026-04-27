@echo off
REM Launcher for SWE inference on Intel GPU 1. Call via WMI.
REM Args: %1=track (14B-zs, 32B-zs, 14B-sft, 14B-sft-grpo)
REM       %2=model-path
REM       %3=adapter-path (or NONE)
REM       %4=output

setlocal
set CUDA_VISIBLE_DEVICES=1
set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

cd /d D:\zcy\SILR-Agent

set ADAPTER=
if NOT "%3"=="NONE" set ADAPTER=--adapter-path %3

C:\Users\Administrator\miniconda3\envs\pytorch_env\python.exe -u scripts\eval_swe_inference.py ^
    --track %1 ^
    --model-path %2 ^
    %ADAPTER% ^
    --manifest D:\zcy\silr-swe-cache\swe-bench-lite.jsonl ^
    --repo-cache D:\zcy\silr-swe-cache\repos ^
    --output %4
