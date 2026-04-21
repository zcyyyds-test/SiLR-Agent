"""Fetch Alibaba cluster-trace-GPU-v2023 files.

Priority order:
  1. Local scp-ed path (recommended for Intel server — local network only)
  2. GitHub raw / proxy mirror fallback
  3. Fail with clear instructions.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

EXPECTED_FILES = {
    "openb_node_list_gpu_node.csv": (
        "https://raw.githubusercontent.com/alibaba/clusterdata/master/"
        "cluster-trace-gpu-v2023/openb_node_list_gpu_node.csv"
    ),
    "openb_pod_list_default.csv": (
        "https://raw.githubusercontent.com/alibaba/clusterdata/master/"
        "cluster-trace-gpu-v2023/openb_pod_list_default.csv"
    ),
}

MIRROR_PREFIXES = [
    "https://ghproxy.com/",
    "https://gh.api.99988866.xyz/",
]


def verify(raw_dir: Path) -> bool:
    """Return True iff all EXPECTED_FILES are present and non-empty."""
    for fname in EXPECTED_FILES:
        p = Path(raw_dir) / fname
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def fetch(raw_dir: Path, *, allow_mirror: bool = True) -> None:
    """Populate raw_dir with EXPECTED_FILES. Raises RuntimeError on failure."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for fname, url in EXPECTED_FILES.items():
        target = raw_dir / fname
        if target.is_file() and target.stat().st_size > 0:
            logger.info("skip existing %s", fname)
            continue

        urls = [url] + ([m + url for m in MIRROR_PREFIXES] if allow_mirror else [])
        last_err: Exception | None = None
        for u in urls:
            try:
                logger.info("download %s → %s", u, target)
                with urllib.request.urlopen(u, timeout=60) as r, open(target, "wb") as f:
                    shutil.copyfileobj(r, f)
                break
            except Exception as e:
                last_err = e
                logger.warning("failed %s: %s", u, e)
        else:
            raise RuntimeError(
                f"All download URLs failed for {fname}. "
                f"Last error: {last_err}. "
                f"scp from local to {target} manually as workaround."
            )

    if not verify(raw_dir):
        raise RuntimeError(f"Post-fetch verification failed at {raw_dir}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
