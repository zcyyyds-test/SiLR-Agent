@echo off
set CUDA_VISIBLE_DEVICES=0
cd /d D:\zcy\SILR-Agent
mkdir outputs\grpo_model 2>nul
C:\Users\Administrator\miniconda3\envs\pytorch_env\python.exe -u scripts/train_grpo.py --base-model D:\zcy\models\Qwen\Qwen3-14B --sft-adapter D:\zcy\SILR-Agent\outputs\sft_model\final --output D:\zcy\SILR-Agent\outputs\grpo_model --iterations 3 --rollouts-per-scenario 2 --lr 5e-6 --kl-coeff 0.02 --batch-size 4 --max-steps 7 2> D:\zcy\SILR-Agent\outputs\grpo_model\stderr.log
