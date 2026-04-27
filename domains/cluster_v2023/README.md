# cluster_v2023 — Alibaba OpenB GPU cluster domain

Reference domain built on the **Alibaba cluster-trace-GPU-v2023** ([OpenB][openb])
production trace released with FGD (ATC'23). Sister to the synthetic
`domains/cluster/` — same SiLR framework, now driven by real workload
and the paper's native fragmentation metric.

## Constraints (5 checkers)

| # | Checker | Trace field | Role |
|---|---|---|---|
| 1 | `resource_capacity` | cpu_milli / memory_mib / gpu | **per-action gate** |
| 2 | `affinity` | gpu_spec ↔ node.model | observer-only |
| 3 | `priority` | qos (LS/Burstable/BE) | observer-only |
| 4 | `queue` | job status | observer-only |
| 5 | `fragmentation` (FGD formula) | derived from `p(g)` | observer-only |

`Affinity / Priority / Queue / Fragmentation` are observer-only — per-action
gating of any of these would reject 100 % of intermediate actions when
the scenario starts with multiple violations and a single migrate can
only resolve one at a time. Capacity is the single hard structural
constraint that prevents proposing physically impossible migrates;
the others are episode-level success criteria fed to `Observation.is_stable`.

## Fragmentation formula (FGD ATC'23)

```
F(cluster) = Σ_node Σ_g∈G  p(g) · 𝟙[0 < remaining(node) < g] · remaining(node)
```

- `G = {1, 2, 4, 8}` — common job GPU sizes
- `p(g)` — **empirical distribution precomputed from v2023 trace**
  (`data_pipeline/compute_job_size_dist.py → job_size_dist.json`).
  Hardcoded fallback exists for smoke tests but is NOT FGD-comparable.

## Scenarios

60 recovery scenarios on 40 stratified-sampled nodes, ~400 pending jobs
drawn from a 60-minute window of the OpenB trace. Fault distribution
approximates Philly ATC'19 empirical ratios:

| Fault type | Count | Trigger |
|---|---|---|
| `node_failure` | 31 | 5 nodes set to Down; their running jobs preempted |
| `qos_pressure` | 12 | Force LS jobs into Queued while BE is Running |
| `gpu_spec_mismatch` | 11 | Inject `gpu_spec_required` mismatching the node's model on N=2 or 3 jobs |
| `fragmentation_surge` | 6 | Scatter small BE jobs + queue a large LS job |

Scenarios are checked by the Best-fit expert and kept only if the
expert can produce a non-empty plan. The expert itself recovers ~84 %
overall — see Results.

## Observation

Compact JSON (≤1 KB on 40-node clusters) with the fields the policy
needs to plan a recovery:

| Field | Content |
|---|---|
| `sum` | counts (ready/down/free_gpu/queued/running) |
| `down` | node ids in Down status |
| `free` | `[node_id, model, free_gpu]` for ready nodes with spare GPU |
| `q` | queued jobs `[id, qos, gpu, gpu_spec_required]` |
| `strand` | running jobs on Down nodes (need migration) |
| `be_run` | preemptable BE jobs (capped at 12) |
| `aff_run` | running jobs whose `gpu_spec_required` ≠ current node's `model` (need migration) |
| `F` / `F_th` | fragmentation index and threshold |
| `viol` | constraint types currently violated |

The `aff_run` field is critical for `gpu_spec_mismatch` recovery: the
policy needs to know which jobs are misplaced and what model they
require so it can pick a matching node from `free`.

## Results

60 scenarios × 1 repeat, greedy decoding (temperature=0), Qwen3-14B +
LoRA (r=64, α=128), single RTX PRO 6000 Blackwell 96 GB.

| Method | Recovery | F_normalized | Reject Rate |
|---|---|---|---|
| Best-fit expert (teacher) | 84.0% (construction) | 1.000 | 0.0% |
| **Qwen3-14B zero-shot** | **0.0%** | 0.048 | **100.0%** |
| SiLR-SFT (14B) | **91.7%** | 0.065 | 10.6% |

Per fault-type (SiLR-SFT):

| Fault type | Recovery | Note |
|---|---|---|
| `fragmentation_surge` | 6/6 (100%) | 13-step trajectories, longest horizon |
| `qos_pressure` | 12/12 (100%) | preempt-then-assign LS jobs |
| `node_failure` | 30/31 (97%) | 1 unrecovered is an edge case where Best-fit also stalls |
| `gpu_spec_mismatch` | 7/11 (64%) | matches teacher solvability — 4 unrecovered are scenarios the Best-fit teacher also fails |

Headline: zero-shot → SFT = **+91.7 pp recovery** and reject rate
100 % → 11 %. Fragmentation index F_normalized 0.065 means the policy
keeps the cluster ~15× less fragmented than the Best-fit teacher's
F=1.0 baseline (the verifier rejects high-F migrate candidates,
nudging the policy toward consolidation).

GRPO was attempted on top of the SFT base in multiple iterations; in
the final iteration the PPO objective converged cleanly (log-ratio
mean 0.05, clamp 0/528) but eval recovery degraded by 10 pp because
the `fragmentation_surge` bucket reached 100% rollout success → group-
relative advantage of 0 → no protective gradient. Other-bucket policy
updates spilled over via the shared backbone and broke the precise
fragmentation-recovery sequence. Frag-specific reward shaping or an
SFT-anchor BC term would address it but are deferred — the data-side
fixes that drove SFT to 91.7% already saturate the teacher coverage.

## Limitations

### `gpu_spec_mismatch` 7/11 — capped by teacher solvability

The 4 unrecovered `gpu_spec_mismatch` scenarios collide with the
cluster topology: the trace has only 2 V100M16 nodes, so when the
fault demands placing 3+ jobs on V100M16 the Best-fit (non-preempting)
teacher cannot satisfy them, leaving the SFT student with no positive
trajectory to imitate. Lifting this requires either ShadowExpert-style
preempt-then-migrate planning to expand teacher coverage, or sampling
node populations that include more V100M16 capacity. Out of scope here.

### `gpu_spec_mismatch` injection difficulty

`scripts/build_cluster_v2023_scenarios.py` samples `n_jobs ∈ {2,2,3}`
when injecting affinity violations (down from `{2,2,3,4}` in the
original spec). The dropped `n=4` arm is essentially infeasible on
40-node clusters with the V100M16 topology described above; keeping it
just bloated eval with hard-impossible scenarios without giving SFT
any positive trajectory to learn from. The reported gpu_spec recovery
should therefore be read as "on the difficulty distribution where the
teacher itself can solve roughly half".

### Single-GPU constraint

Training and evaluation are pinned to one 96 GB GPU. Multi-GPU LoRA
under 4-bit quantization is not validated in this fork; the training
scripts assume `CUDA_VISIBLE_DEVICES=0`.

## Reproduce

```bash
# 1. Trace (from local → server)
scp domains/cluster_v2023/data_pipeline/raw/openb_*.csv \
    administrator@server:/path/to/SILR-Agent-cluster-v2023/domains/cluster_v2023/data_pipeline/raw/

# 2. Precompute p(g)
python -c "from pathlib import Path; from domains.cluster_v2023.data_pipeline.compute_job_size_dist import compute_dist, save_dist; \
save_dist(compute_dist(Path('domains/cluster_v2023/data_pipeline/raw/openb_pod_list_default.csv')), \
          Path('domains/cluster_v2023/data_pipeline/job_size_dist.json'))"

# 3. Build 60 scenarios
python scripts/build_cluster_v2023_scenarios.py \
    --raw-dir domains/cluster_v2023/data_pipeline/raw \
    --out-dir domains/cluster_v2023/scenarios/data --n 60

# 4. SFT data: collect → split per-step → ID anonymize
python scripts/collect_cluster_v2023_sft.py \
    --scenario-dir domains/cluster_v2023/scenarios/data \
    --out outputs/cluster_v2023/sft_data.jsonl \
    --seeds 0 1 2 3 4 5 6 7
python scripts/split_trajectories_to_steps.py \
    --input outputs/cluster_v2023/sft_data.jsonl \
    --output outputs/cluster_v2023/sft_data_per_step.json \
    --compress-user --only-success
python scripts/anonymize_sft_ids.py \
    --input outputs/cluster_v2023/sft_data_per_step.json \
    --output outputs/cluster_v2023/sft_data_per_step_anon.json

# 5. Train (single GPU 0)
scripts/bat/eval_zero_shot_14b.bat
scripts/bat/train_sft_cluster_v2023.bat
scripts/bat/eval_sft_cluster_v2023.bat

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
