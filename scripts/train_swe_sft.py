"""Thin entry-point that calls train_sft.main with SWE defaults.

Kept as a separate file so train_sft.py stays domain-agnostic (it's already
used by cluster + finance). SWE-specific default: max-seq-len=4096
(code trajectories are longer than finance's 2048 default).
"""
from __future__ import annotations

import sys

from scripts.train_sft import main as _train_sft_main


if __name__ == "__main__":
    # Inject SWE defaults unless the caller overrode them.
    injected = ["--max-seq-len", "4096"]
    sys.argv = [sys.argv[0]] + injected + sys.argv[1:]
    _train_sft_main()
