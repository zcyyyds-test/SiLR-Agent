@echo off
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent
mkdir outputs\eval_grpo 2>nul
C:\Users\Administrator\miniconda3\envs\pytorch_env\python.exe -u scripts/eval_sft.py --base-model D:\zcy\models\Qwen\Qwen3-14B --adapter D:\zcy\SILR-Agent\outputs\grpo_model\final --output D:\zcy\SILR-Agent\outputs\eval_grpo --repeats 3 --max-steps 10 2> D:\zcy\SILR-Agent\outputs\eval_grpo\stderr.log
