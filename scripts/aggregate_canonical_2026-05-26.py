"""Aggregate canonical fixed-code results + multi-model robustness.

Inputs (AMD-side paths via local pull):
- eval_m1_canonical_fixed_gpu0.json   (m1 x 4 policies x 3 reps)
- eval_m23_canonical_fixed_gpu0.json  (m2,m3 x 4 policies x 3 reps)
- eval_multimodel_8b_gpu1.json        (multi_3 x {terminal, progress_mag} x 3)
- eval_multimodel_32b_gpu1.json       (multi_3 x {terminal, progress_mag} x 3)
- eval_multi_action_expansion_gpu1_v2.json  (v2 broken-code reference)

Outputs:
- experiments/canonical_table_rq1_2026-05-26.md
- experiments/multimodel_robustness_2026-05-26.md
- experiments/canonical_vs_v2_diff_2026-05-26.md
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
EXP.mkdir(exist_ok=True)


def load(name):
    p = ROOT / name
    if not p.exists():
        print(f"MISSING: {name}", file=sys.stderr)
        return None
    return json.load(open(p))


def filter_episodes(data, scenarios=None, policies=None, seeds=None):
    out = []
    for e in data.get("episodes", []):
        if scenarios and e.get("scenario") not in scenarios:
            continue
        if policies and e.get("policy") not in policies:
            continue
        if seeds and e.get("rep_seed") not in seeds:
            continue
        out.append(e)
    return out


def policy_summary(eps):
    by_pol = defaultdict(list)
    for e in eps:
        by_pol[e["policy"]].append(e)
    rows = []
    for pol, runs in sorted(by_pol.items()):
        n = len(runs)
        rec = sum(1 for e in runs if e["recovered"])
        fp_mean = sum(e["final_penalty"] for e in runs) / max(1, n)
        worse = sum(1 for e in runs if e["final_penalty"] > e.get("default_penalty", 0))
        # reject/prop rate
        total_prop = sum(e.get("total_proposals", 0) for e in runs)
        total_rej = sum(e.get("total_rejections", 0) for e in runs)
        rp = total_rej / total_prop if total_prop else 0.0
        rows.append((pol, rec, n, fp_mean, worse, rp))
    return rows


# ============================================================
# RQ1 canonical: combine m1 + m23 canonical (fixed code) at matched N=3
# ============================================================
def build_rq1_canonical():
    m1 = load("eval_m1_canonical_fixed_gpu0.json")
    m23 = load("eval_m23_canonical_fixed_gpu0.json")
    if not m1 or not m23:
        return "(canonical m1 or m23 missing — fall back to v2 partial)"
    multi = ("mined_multi_action_1_l0p25g1p0_s5",
             "mined_multi_action_2_l1p0g1p0_s5",
             "mined_multi_action_3_l0p25g1p0_s12")
    pols = ("terminal", "progress", "progress_mag", "scalar_progress")
    seeds = (1000, 1001, 1002)
    eps = filter_episodes(m1, scenarios=multi, policies=pols, seeds=seeds) + \
          filter_episodes(m23, scenarios=multi, policies=pols, seeds=seeds)
    rows = policy_summary(eps)
    lines = [
        "## RQ1 canonical (fixed code, matched N=3)",
        "",
        "| Policy | Recovery | Final pen. mean | Worsening | Reject/prop |",
        "|---|---|---|---|---|",
    ]
    for pol, rec, n, fp, worse, rp in rows:
        lines.append(f"| `{pol}` | {rec}/{n} | {fp:.3f} | {worse}/{n} | {rp:.3f} |")
    return "\n".join(lines)


# ============================================================
# Per-scenario breakdown (m1 vs m2 vs m3 under canonical fix)
# ============================================================
def build_per_scenario():
    m1 = load("eval_m1_canonical_fixed_gpu0.json") or {"episodes": []}
    m23 = load("eval_m23_canonical_fixed_gpu0.json") or {"episodes": []}
    all_eps = m1["episodes"] + m23["episodes"]
    by = defaultdict(lambda: defaultdict(list))
    for e in all_eps:
        sc = e["scenario"].replace("mined_multi_action_", "m").split("_l")[0]
        by[sc][e["policy"]].append(e)
    lines = [
        "",
        "## Per-scenario (canonical fix)",
        "",
        "| Scenario | Policy | Recovery | Final pen mean | Reject/prop |",
        "|---|---|---|---|---|",
    ]
    for sc in sorted(by):
        for pol in ("terminal", "progress", "progress_mag", "scalar_progress"):
            runs = by[sc].get(pol, [])
            if not runs:
                continue
            rec = sum(1 for e in runs if e["recovered"])
            fp = sum(e["final_penalty"] for e in runs) / len(runs)
            tp = sum(e.get("total_proposals", 0) for e in runs)
            tr = sum(e.get("total_rejections", 0) for e in runs)
            rp = tr / tp if tp else 0.0
            lines.append(f"| {sc} | `{pol}` | {rec}/{len(runs)} | {fp:.3f} | {rp:.3f} |")
    return "\n".join(lines)


# ============================================================
# Multi-model robustness on multi_3
# ============================================================
def build_multimodel():
    sources = {
        "Qwen3-8B":  ("eval_multimodel_8b_gpu1.json",  ),
        "Qwen3-14B": ("eval_m23_canonical_fixed_gpu0.json",  ),  # m3 piece
        "Qwen3-32B": ("eval_multimodel_32b_gpu1.json", ),
    }
    lines = [
        "# Multi-Model Robustness on `mined_multi_action_3`",
        "",
        "| Model | Policy | Recovery | Final pen mean | Reject/prop |",
        "|---|---|---|---|---|",
    ]
    for model, files in sources.items():
        for fname in files:
            d = load(fname)
            if not d:
                lines.append(f"| {model} | — | (missing {fname}) | | |")
                continue
            eps = filter_episodes(d, scenarios=("mined_multi_action_3_l0p25g1p0_s12",),
                                  policies=("terminal", "progress_mag"),
                                  seeds=(1000, 1001, 1002))
            for pol in ("terminal", "progress_mag"):
                runs = [e for e in eps if e["policy"] == pol]
                if not runs:
                    lines.append(f"| {model} | `{pol}` | (no data) | | |")
                    continue
                rec = sum(1 for e in runs if e["recovered"])
                fp = sum(e["final_penalty"] for e in runs) / len(runs)
                tp = sum(e.get("total_proposals", 0) for e in runs)
                tr = sum(e.get("total_rejections", 0) for e in runs)
                rp = tr / tp if tp else 0.0
                lines.append(f"| {model} | `{pol}` | {rec}/{len(runs)} | {fp:.3f} | {rp:.3f} |")
    return "\n".join(lines)


# ============================================================
# Comparison: v2 (broken ADMITTED feedback) vs canonical (fixed)
# ============================================================
def build_v2_diff():
    v2 = load("eval_multi_action_expansion_gpu1_v2.json")
    m1 = load("eval_m1_canonical_fixed_gpu0.json")
    m23 = load("eval_m23_canonical_fixed_gpu0.json")
    if not v2:
        return "(v2 missing)"
    multi_n3 = ("mined_multi_action_1_l0p25g1p0_s5",
                "mined_multi_action_2_l1p0g1p0_s5",
                "mined_multi_action_3_l0p25g1p0_s12")
    pols = ("terminal", "progress", "progress_mag", "scalar_progress")
    seeds = (1000, 1001, 1002)
    v2_eps = filter_episodes(v2, scenarios=multi_n3, policies=pols, seeds=seeds)
    canon_eps = []
    if m1: canon_eps += filter_episodes(m1, scenarios=multi_n3, policies=pols, seeds=seeds)
    if m23: canon_eps += filter_episodes(m23, scenarios=multi_n3, policies=pols, seeds=seeds)
    v2_rows = {r[0]: r for r in policy_summary(v2_eps)}
    canon_rows = {r[0]: r for r in policy_summary(canon_eps)}
    lines = [
        "# v2 (broken ADMITTED) vs canonical (fixed APPROVED) at matched N=3",
        "",
        "| Policy | v2 Recovery | v2 fp mean | Canonical Recovery | Canonical fp mean | Δ recovery |",
        "|---|---|---|---|---|---|",
    ]
    for pol in pols:
        v2r = v2_rows.get(pol)
        cr = canon_rows.get(pol)
        if v2r and cr:
            dr = cr[1] - v2r[1]
            lines.append(f"| `{pol}` | {v2r[1]}/{v2r[2]} | {v2r[3]:.3f} | {cr[1]}/{cr[2]} | {cr[3]:.3f} | {dr:+d} |")
        elif v2r:
            lines.append(f"| `{pol}` | {v2r[1]}/{v2r[2]} | {v2r[3]:.3f} | — | — | — |")
        elif cr:
            lines.append(f"| `{pol}` | — | — | {cr[1]}/{cr[2]} | {cr[3]:.3f} | — |")
    return "\n".join(lines)


def build_step8_diagnostic():
    d = load("eval_step8_progmag_gpu0.json")
    if not d:
        return "(step-8 diagnostic missing)"
    eps = d.get("episodes", [])
    by = defaultdict(list)
    for e in eps:
        sc = e["scenario"].replace("mined_multi_action_", "m").split("_l")[0]
        by[sc].append(e)
    lines = [
        "## Step-budget diagnostic — progress_mag at max_steps=8 (vs 6 default)",
        "",
        "| Scenario | Recovery at max_steps=6 | Recovery at max_steps=8 | Δ |",
        "|---|---|---|---|",
    ]
    # Reference: matched N=3 from canonical fix
    m1 = load("eval_m1_canonical_fixed_gpu0.json")
    m23 = load("eval_m23_canonical_fixed_gpu0.json")
    canon_pm_by_sc = defaultdict(list)
    for src in (m1, m23):
        if not src: continue
        for e in src.get("episodes", []):
            if e.get("policy") == "progress_mag":
                sc = e["scenario"].replace("mined_multi_action_", "m").split("_l")[0]
                canon_pm_by_sc[sc].append(e)
    for sc in sorted(by):
        s8_recs = sum(1 for e in by[sc] if e["recovered"])
        s8_n = len(by[sc])
        s6 = canon_pm_by_sc.get(sc, [])
        s6_recs = sum(1 for e in s6 if e["recovered"])
        s6_n = len(s6)
        delta = s8_recs - s6_recs
        lines.append(f"| {sc} | {s6_recs}/{s6_n} | {s8_recs}/{s8_n} | {delta:+d} |")
    total_s8 = sum(sum(1 for e in v if e["recovered"]) for v in by.values())
    total_s8_n = sum(len(v) for v in by.values())
    total_s6 = sum(sum(1 for e in v if e["recovered"]) for v in canon_pm_by_sc.values())
    total_s6_n = sum(len(v) for v in canon_pm_by_sc.values())
    lines.append(f"| **TOTAL** | **{total_s6}/{total_s6_n}** | **{total_s8}/{total_s8_n}** | **{total_s8-total_s6:+d}** |")
    return "\n".join(lines)


def main():
    rq1 = build_rq1_canonical()
    per = build_per_scenario()
    mm = build_multimodel()
    diff = build_v2_diff()
    step8 = build_step8_diagnostic()

    (EXP / "canonical_table_rq1_2026-05-26.md").write_text(
        "# Canonical RQ1 (fixed code, 2026-05-26)\n\n" + rq1 + "\n" + per + "\n\n" + step8 + "\n"
    )
    (EXP / "multimodel_robustness_2026-05-26.md").write_text(mm + "\n")
    (EXP / "canonical_vs_v2_diff_2026-05-26.md").write_text(diff + "\n")
    (EXP / "step8_diagnostic_2026-05-26.md").write_text(step8 + "\n")

    print("=" * 60)
    print(rq1)
    print(per)
    print("=" * 60)
    print(mm)
    print("=" * 60)
    print(diff)
    print("=" * 60)
    print("Wrote:")
    print("  experiments/canonical_table_rq1_2026-05-26.md")
    print("  experiments/multimodel_robustness_2026-05-26.md")
    print("  experiments/canonical_vs_v2_diff_2026-05-26.md")


if __name__ == "__main__":
    main()
