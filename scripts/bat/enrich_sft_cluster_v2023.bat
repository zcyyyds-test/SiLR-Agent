@echo off
rem SFT CoT enrichment — runs via LemonAPI relay, takes 30-80 min for 200 trajectories.
rem API key set by caller (WMI launcher) via environment variables.
cd /d D:\zcy\SILR-Agent-cluster-v2023
call C:\Users\Administrator\miniconda3\Scripts\activate.bat pytorch_env
set PYTHONPATH=D:\zcy\SILR-Agent-cluster-v2023
python -u scripts\enrich_cluster_v2023_sft.py ^
    --in outputs\cluster_v2023\sft_data_v2023_base.jsonl ^
    --out outputs\cluster_v2023\sft_data_v2023.enriched.jsonl ^
    --final-json outputs\cluster_v2023\sft_data_v2023.json ^
    > outputs\cluster_v2023\enrich.log 2>&1
