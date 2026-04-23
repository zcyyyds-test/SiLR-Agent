# cluster_v2023 — Alibaba OpenB GPU cluster domain

Reference domain built on the **Alibaba cluster-trace-GPU-v2023** ([OpenB][openb])
production trace released with FGD (ATC'23). Sister to the synthetic
`domains/cluster/` — same SiLR framework, now driven by real workload
and the paper's native fragmentation metric.

## Constraints (5 checkers)

| # | Checker | Trace field | Role |
|---|---|---|---|
| 1 | `resource_capacity` | cpu_milli / memory_mib / gpu | **per-action gate** |
| 2 | `affinity` | gpu_spec ↔ node.model | **per-action gate** |
| 3 | `priority` | qos (LS/Burstable/BE) | observer-only |
| 4 | `queue` | job status | observer-only |
| 5 | `fragmentation` (FGD formula) | derived from `p(g)` | observer-only |

`Priority / Queue / Fragmentation` are observer-only — per-action
gating of queue emptiness / preemption-of-lower-priority would
reject 100 % of intermediate actions, because draining happens
gradually across multiple assigns. The signals still feed the GRPO
dense reward. See inline docstrings in `checkers.py`.

## Fragmentation formula (FGD ATC'23)

```
F(cluster) = Σ_node Σ_g∈G  p(g) · 𝟙[0 < remaining(node) < g] · remaining(node)
```

- `G = {1, 2, 4, 8}` — common job GPU sizes
- `p(g)` — **empirical distribution precomputed from v2023 trace**
  (`data_pipeline/compute_job_size_dist.py → job_size_dist.json`).
  Hardcoded fallback exists for smoke tests but is NOT FGD-comparable.

## Scenarios

25 recovery scenarios on 40 stratified-sampled nodes, ~400 pending jobs
drawn from a 60-minute window of the OpenB trace. Distribution per
Philly ATC'19 fault ratios:

| Fault type | Ratio | Trigger |
|---|---|---|
| `node_failure` | 50 % | 1–3 nodes set to Down; their running jobs preempted |
| `gpu_spec_mismatch` | 20 % | Inject `gpu_spec_required` mismatching the node's model |
| `qos_pressure` | 20 % | Force LS jobs into Queued while BE is Running |
| `fragmentation_surge` | 10 % | Scatter small BE jobs + queue a large LS job |

15 training / 10 held-out. All scenarios solvability-checked by the
Best-fit expert (rerolled if unsolvable ≤15 steps).

## Results

28 scenarios × 1 repeat, greedy decoding (temperature=0), Qwen3-14B +
LoRA (r=64, α=128), single RTX PRO 6000 Blackwell 96 GB.

| Method | Recovery | F_normalized | Reject Rate |
|---|---|---|---|
| Best-fit expert (teacher) | 84.0% (construction) | 1.000 | 0.0% |
| **Qwen3-14B zero-shot** | **0.0%** | 0.048 | **100.0%** |
| SiLR-SFT (14B) | **75.0%** | 0.055 | 29.7% |
| SiLR-SFT+GRPO (14B, 2 iter) | 75.0% | 0.055 | 29.7% |

Per fault-type (SiLR-SFT):

| Fault type | Recovery | Teacher cap | Note |
|---|---|---|---|
| `fragmentation_surge` | 3/3 (100%) | 100% | 13-step trajectories, longest horizon |
| `qos_pressure` | 6/6 (100%) | 100% | 15 steps, preempt-then-assign |
| `node_failure` | 12/14 (86%) | 100% | 2 unrecovered are edge cases |
| `gpu_spec_mismatch` | **0/5 (0%)** | 20% | Teacher baseline also low; see Limitations |

Headline: zero-shot → SFT = **+75 pp recovery** and reject rate
100 % → 30 %. Fragmentation is ~18× below the Best-fit teacher baseline
(F_normalized 0.055 means we actually fragment less than the expert
does on average, because the verifier gate rejects high-F proposals).

GRPO (2 iterations × rollouts_per_scenario=2) produced byte-identical
greedy evaluation to SFT despite `adapter_model.safetensors` hash
diverging. Small `lr=1e-6 × clip_eps=0.2 × kl_coeff=0.02` keeps the
policy distribution within ε of SFT, and there are no positive reward
signals for `gpu_spec_mismatch` to bootstrap the bad-coverage fault
type. Consistent with GridAgent's decisions log note _"任务的正确操作近乎
确定性，动作空间不可随机探索，binary reward 无梯度"_.

## Limitations

### gpu_spec_mismatch 0% — training-time spec coverage gap

Diagnostic (Codex + Kimi review plus data-driven analysis):

- 8 teacher-success trajectories (out of 40 gpu_spec_mismatch seeds)
  used only **2 unique target node_ids**: `openb-node-0835` (G3 spec)
  and `openb-node-0024` (V100M32 spec).
- 15× upsample of these trajectories made the model emit `migrate_job`
  with correct format (reject reason shifted from "missing node_id"
  to "affinity violation") but picking a wrong target node.
- Test scenarios demand `V100M16` in 4/5 cases. Training never showed
  the model a successful migration to V100M16 nodes (`0271`, `1137`),
  because Best-fit expert fails those cases at 0% → filtered out by
  `--only-success`. Hence the model learned "migrate to 0835 or 0024"
  as a policy, not "match queued spec with free node model".
- Fix would require regenerating 15–30 gpu_spec_mismatch scenarios
  with enforced spec diversity (A10 / T4 / P100 / V100M16 / V100M32 /
  G3), plus making Best-fit solvable on V100M16 demand (currently 2
  nodes of V100M16 are occupied in all 5 scenarios). Estimated cost:
  10–12 h re-collect + retrain. Deferred.

See `decisions-cluster-v2023.md` Part 10–11 for full diagnostic trail.

## Reproduce

```bash
# 1. Trace (from local → scp to Intel server)
scp domains/cluster_v2023/data_pipeline/raw/openb_*.csv \
    administrator@intel:/d/zcy/SILR-Agent-cluster-v2023/domains/cluster_v2023/data_pipeline/raw/

# 2. Precompute p(g)
python -c "from pathlib import Path; from domains.cluster_v2023.data_pipeline.compute_job_size_dist import compute_dist, save_dist; \
save_dist(compute_dist(Path('domains/cluster_v2023/data_pipeline/raw/openb_pod_list_default.csv')), \
          Path('domains/cluster_v2023/data_pipeline/job_size_dist.json'))"

# 3. Build 25 scenarios (default window is tuned to OpenB creation_time
#    distribution; see --help for details)
python scripts/build_cluster_v2023_scenarios.py \
    --raw-dir domains/cluster_v2023/data_pipeline/raw \
    --out-dir domains/cluster_v2023/scenarios/data --n 25

# 4. SFT data (Best-fit expert + GPT-5.4 CoT)
python scripts/collect_cluster_v2023_sft.py \
    --scenario-dir domains/cluster_v2023/scenarios/data \
    --out outputs/cluster_v2023/sft_data_v2023_base.jsonl \
    --seeds 0 1 2 3 4 5 6 7
LEMON_API_KEY=... python scripts/enrich_cluster_v2023_sft.py \
    --in outputs/cluster_v2023/sft_data_v2023_base.jsonl \
    --out outputs/cluster_v2023/sft_data_v2023.enriched.jsonl \
    --final-json outputs/cluster_v2023/sft_data_v2023.json

# 5. Train (Intel GPU 0 only — see /remote-train skill)
scripts/bat/eval_zero_shot_14b.bat      # baseline 1
scripts/bat/eval_zero_shot_32b.bat      # baseline 2
scripts/bat/train_sft_cluster_v2023.bat
scripts/bat/eval_sft_cluster_v2023.bat  # gate: recovery ≥ 80%
scripts/bat/grpo_sanity_cluster_v2023.bat   # gate: log_prob ratio ≈ 1.0
scripts/bat/train_grpo_cluster_v2023.bat         # iter 1
scripts/bat/train_grpo_cluster_v2023_iter2.bat   # iter 2
scripts/bat/train_grpo_cluster_v2023_iter3.bat   # iter 3
scripts/bat/eval_grpo_cluster_v2023.bat

# 6. Comparison table
python scripts/build_cluster_v2023_comparison.py
```

## Citations

- Weng et al., _Beware of Fragmentation: Scheduling GPU-Sharing
  Workloads with Fragmentation Gradient Descent_, USENIX ATC'23
  (trace + F formula).
- Jeon et al., _Analysis of Large-Scale Multi-Tenant GPU Clusters for
  DNN Training Workloads_, USENIX ATC'19 (fault-distribution baseline).
- Trace repo: <https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023>

[openb]: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023
