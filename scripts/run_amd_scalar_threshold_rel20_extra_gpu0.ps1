$ErrorActionPreference = "Stop"
Set-Location "D:\zcy\SILR-WISE26"

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = "."
$env:SILR_SCALAR_PROGRESS_RELATIVE_SLACK = "0.20"
$env:SILR_MAX_TOKENS = "768"

& "D:\miniconda3\envs\pytorch_env\python.exe" -u "scripts\anm_eval_sweep.py" `
  --base-url "http://127.0.0.1:8003/v1" `
  --model "qwen3-14b" `
  --scenarios `
    "mined_multi_action_2_l1p0g1p0_s5" `
    "mined_multi_action_3_l0p25g1p0_s12" `
  --policies "scalar_progress" `
  --reps 2 `
  --rep-start-seed 1003 `
  --max-steps 6 `
  --request-timeout-s 240 `
  --max-retries 0 `
  --output "eval_scalar_threshold_rel20_extra_gpu0.json" `
  --log-file "logs\eval_scalar_threshold_rel20_extra_gpu0.log"

exit $LASTEXITCODE
