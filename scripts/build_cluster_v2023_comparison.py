"""Collate all four eval JSON files into a single markdown comparison table.

Writes `outputs/cluster_v2023/comparison_table.md` and prints to stdout.
Missing JSONs are rendered as em-dash placeholders — safe to run before
all eval tracks have completed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TABLE_TEMPLATE = """| Method | Recovery | F_normalized | Reject Rate |
|---|---|---|---|
| Best-fit expert | 100.0% (construction) | 1.000 | 0.0% |
| Qwen3-14B zero-shot | {zs14_rec} | {zs14_f} | {zs14_rej} |
| Qwen3-32B zero-shot | {zs32_rec} | {zs32_f} | {zs32_rej} |
| SiLR-SFT (14B) | {sft_rec} | {sft_f} | {sft_rej} |
| **SiLR-SFT+GRPO (14B)** | **{grpo_rec}** | **{grpo_f}** | **{grpo_rej}** |
"""


def _agg(path: Path) -> dict:
    if not path.is_file():
        return {"recovery_rate": None, "mean_F_normalized": None,
                "reject_rate": None}
    data = json.loads(path.read_text())
    return data.get("aggregate", data)


def _fmt_pct(x):
    return "—" if x is None else f"{x*100:.1f}%"


def _fmt_f(x):
    return "—" if x is None else f"{x:.3f}"


def build_table(out_dir: Path) -> str:
    zs14 = _agg(out_dir / "zero_shot_14b.json")
    zs32 = _agg(out_dir / "zero_shot_32b.json")
    sft = _agg(out_dir / "eval_sft.json")
    grpo = _agg(out_dir / "eval_grpo.json")
    return TABLE_TEMPLATE.format(
        zs14_rec=_fmt_pct(zs14["recovery_rate"]),
        zs14_f=_fmt_f(zs14["mean_F_normalized"]),
        zs14_rej=_fmt_pct(zs14["reject_rate"]),
        zs32_rec=_fmt_pct(zs32["recovery_rate"]),
        zs32_f=_fmt_f(zs32["mean_F_normalized"]),
        zs32_rej=_fmt_pct(zs32["reject_rate"]),
        sft_rec=_fmt_pct(sft["recovery_rate"]),
        sft_f=_fmt_f(sft["mean_F_normalized"]),
        sft_rej=_fmt_pct(sft["reject_rate"]),
        grpo_rec=_fmt_pct(grpo["recovery_rate"]),
        grpo_f=_fmt_f(grpo["mean_F_normalized"]),
        grpo_rej=_fmt_pct(grpo["reject_rate"]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="outputs/cluster_v2023")
    args = p.parse_args()
    d = Path(args.out_dir)
    table = build_table(d)
    out_path = d / "comparison_table.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(table)


if __name__ == "__main__":
    main()
