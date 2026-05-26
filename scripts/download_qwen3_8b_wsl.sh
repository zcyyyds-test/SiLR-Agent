#!/bin/bash
# Run inside AMD WSL Ubuntu-24.04: download Qwen3-8B from HF.
set -e
source /root/vllm-env/bin/activate
export HF_ENDPOINT="https://huggingface.co"
export HTTP_PROXY="http://127.0.0.1:7897"
export HTTPS_PROXY="http://127.0.0.1:7897"
export NO_PROXY="localhost,127.0.0.1,100.0.0.0/8"
export HF_HUB_DOWNLOAD_TIMEOUT=300
cd /mnt/d/zcy/models
python -c "
from huggingface_hub import snapshot_download
import logging, time, sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
t0 = time.time()
try:
    p = snapshot_download(repo_id='Qwen/Qwen3-8B', local_dir='Qwen3-8B', local_dir_use_symlinks=False, max_workers=4)
    logging.info('done %s elapsed=%.1fs', p, time.time()-t0)
except Exception as e:
    logging.exception('FAILED: %s', e)
    sys.exit(1)
"
