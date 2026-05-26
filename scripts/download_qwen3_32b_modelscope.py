"""Qwen3-32B download via modelscope (国内镜像源)
Target: D:/zcy/models/Qwen3-32B
~64 GB safetensors, ~10-20 min via modelscope.
"""
import os
import sys
import time
import logging

DEST = "D:/zcy/models/Qwen3-32B"
LOG_PATH = "D:/zcy/models/download_qwen3_32b_modelscope.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)

try:
    from modelscope import snapshot_download
except ImportError:
    logging.error("modelscope not installed; trying pip install via tuna mirror...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
        "modelscope",
    ])
    from modelscope import snapshot_download

t0 = time.perf_counter()
logging.info("Qwen3-32B download starting via modelscope.cn -> %s", DEST)
try:
    path = snapshot_download(
        model_id="Qwen/Qwen3-32B",
        local_dir=DEST,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.py", "*.md", "*.jinja", "tokenizer*"],
    )
    elapsed = time.perf_counter() - t0
    logging.info("DONE in %.1fs -> %s", elapsed, path)
except Exception as e:
    logging.exception("Download failed")
    sys.exit(1)
