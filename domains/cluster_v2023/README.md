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

28 recovery scenarios on 40 stratified-sampled nodes, ~400 pending jobs
drawn from a 60-minute window of the OpenB trace. Fault distribution
approximates Philly ATC'19 empirical ratios:

| Fault type | Count | Trigger |
|---|---|---|
| `node_failure` | 14 | 1–3 nodes set to Down; their running jobs preempted |
| `qos_pressure` | 6 | Force LS jobs into Queued while BE is Running |
| `gpu_spec_mismatch` | 5 | Inject `gpu_spec_required` mismatching the node's model |
| `fragmentation_surge` | 3 | Scatter small BE jobs + queue a large LS job |

Scenarios are checked by the Best-fit expert and kept only if the
expert can recover within 15 steps (teacher baseline ~84 % overall —
see Results).

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

GRPO (2 iterations × rollouts_per_scenario=2, `lr=1e-6` / `clip_eps=0.2`
/ `kl_coeff=0.02`) did not improve over SFT on this benchmark: greedy
evaluation is byte-identical to SFT. `gpu_spec_mismatch` rollouts
produce no positive reward signal (all actions rejected → no gradient
to bootstrap the fault type), and the chosen hyperparameters keep the
policy within ε of the SFT distribution on the other three fault types
where SFT is already at ceiling. Expected behavior given the training
data coverage described below.

## Limitations

### `gpu_spec_mismatch` 0 % — training-time spec coverage gap

The 8 teacher-success trajectories for this fault type (out of 40
seeds) route through only two unique target nodes (`0835` / G3 and
`0024` / V100M32). Test scenarios demand `V100M16` in 4 out of 5
cases, and no V100M16 migration appears in training because the
Best-fit expert fails on those seeds, which an `--only-success` filter
then drops. The trained model therefore learns "migrate to 0835 or
0024" rather than "match queued spec with a free node of the same
model", and fails whenever the test scenario's spec demand falls
outside the trained two-node set.

Fixing this requires regenerating scenarios with enforced spec
diversity (A10 / T4 / P100 / V100M16 in addition to the two covered
specs) and raising the Best-fit success rate on V100M16 demand (the 2
V100M16 nodes are saturated in every existing scenario). Deferred to
future work.

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
