@echo off
cd /d D:\zcy\SILR-WISE26
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd /mnt/d/zcy/SILR-WISE26 && bash scripts/serve_qwen3_14b_gpu1_8004.sh"
