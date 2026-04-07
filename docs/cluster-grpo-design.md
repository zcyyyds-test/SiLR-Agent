# GPU Cluster Job Scheduling Domain + SFT-to-GRPO Pipeline Design

## Goal

Add a GPU cluster job scheduling domain to SILR-Agent and implement a step-level GRPO post-training pipeline, demonstrating that verifier-derived reward signals can improve LLM agent performance beyond SFT alone.

Target: SFT baseline ~80-85% recovery → GRPO ~90-95% on 50+ fault scenarios.

## Domain: GPU Cluster Job Scheduling

Inspired by HPC environments like TSUBAME (Institute of Science Tokyo). An LLM agent manages job scheduling on a multi-node GPU cluster, handling node failures, job surges, and resource contention while respecting priority, affinity, and capacity constraints.

### Cluster Topology

- **15 GPU nodes** across 3 racks (rack-a, rack-b, rack-c), connected via InfiniBand:
  - 6 standard nodes: 4 GPU (80 GB each), 64 CPU cores, 256 GB RAM
  - 6 high-memory nodes: 4 GPU (80 GB each), 64 CPU cores, 512 GB RAM
  - 3 fat nodes: 8 GPU (80 GB each), 128 CPU cores, 1 TB RAM
- **60-80 jobs** from 10-15 job groups (experiments/projects)
  - Resource requests: gpu_count, gpu_memory, cpu_cores, ram
  - 3 priority classes: urgent (non-preemptible, e.g., deadline experiments), normal, preemptible (e.g., hyperparameter sweeps)
  - Multi-node jobs require nodes within the same rack (InfiniBand locality)
  - Some jobs are latency-sensitive (need fat nodes or dedicated GPUs)

### Constraint Checkers (5)

| Checker | Validates | Margin |
|---------|-----------|--------|
| ResourceCapacityChecker | No node exceeds GPU/CPU/RAM capacity | `1 - max_utilization` |
| AffinityChecker | Multi-node jobs co-located in same rack | violation count |
| RackSpreadChecker | Fault-tolerant job groups span 2+ racks | covered_racks / required |
| PriorityChecker | No urgent job queued while preemptible job running | queued urgent count |
| QueueChecker | All jobs scheduled (recovery condition) | `1 - queued / total` |

### Agent Tools (6)

| Tool | Parameters | Effect |
|------|-----------|--------|
| assign_job | job_id, node_id | Place a Queued job on a node |
| migrate_job | job_id, target_node | Move a Running job (checkpoint + restart) |
| preempt_job | job_id | Suspend job, free its resources (re-queued) |
| scale_job | job_id, gpu_count | Adjust GPU allocation (elastic training) |
| drain_node | node_id | Mark node for maintenance (no new jobs) |
| restore_node | node_id | Bring node back online |

### Solver

Recompute derived state after each action: per-node GPU/CPU/RAM utilization, job queue, affinity status. Pure dict operations, O(N), <0.1ms.

### Shadow Copy

`copy.deepcopy(cluster_state)` — cluster state is pure Python dict/dataclass, no external dependencies.

### Scenarios (6 types, parameterized to 50+)

1. **Single node failure** — all jobs on failed node re-queued, need rescheduling
2. **Rack failure** — all nodes in one rack go down (switch/power failure)
3. **Job surge** — batch of high-priority jobs submitted, need immediate placement
4. **Resource fragmentation** — total GPU capacity sufficient but scattered across nodes; multi-GPU jobs can't fit without migration
5. **Priority conflict** — urgent jobs queued but all GPUs occupied by preemptible jobs
6. **Compound** — node failure + job surge simultaneously

Difficulty controlled by: number of affected nodes, job count, GPU pressure level, constraint overlap.

## Training Pipeline

### Phase 1: SFT Data Collection

- Teacher model: Gemini 3 Flash (via SSH tunnel if needed)
- Run all scenarios through SILR agent with teacher model
- TrajectoryRecorder exports successful episodes as SFT data
- Target: 150-300 SFT samples
- Quality gate: filter empty thoughts, low-efficiency trajectories (reuse GridAgent lessons)

### Phase 2: SFT Training

- Model: Qwen3-14B
- Method: QLoRA 4-bit (LoRA r=64, alpha=128, dropout=0.05)
- Config: batch=2, grad_accum=4, lr=2e-4, 3 epochs, max_seq=4096
- Expected baseline: ~80-85% recovery rate

### Phase 3: Step-Level GRPO

#### Why step-level, not episode-level

- Each step yields an independent (observation, action, reward) sample
- Reward is immediately available from SiLR verifier (no sparse episode-level credit assignment)
- 50 scenarios x 8 rollouts x ~5 steps = ~2000 samples per GRPO iteration
- Episode-level GRPO has sparse reward and long sequence issues; can be added as ablation

#### GRPO Loop

```
for iteration in 1..10:
    # 1. Rollout: run current policy on all scenarios, N=8 rollouts each
    samples = []
    for scenario in scenarios:
        for _ in range(8):
            episode = agent.run_episode(scenario)
            for step in episode.steps:
                samples.append(obs, action, reward, group_key=(scenario.id, step.number))

    # 2. Group-relative advantage
    for group in group_by(samples, "group_key"):
        mean_r, std_r = stats(group.rewards)
        for s in group: s.advantage = (s.reward - mean_r) / (std_r + eps)

    # 3. GRPO policy update
    for batch in dataloader(samples):
        ratio = exp(log_prob_current - log_prob_old)
        clipped = clip(ratio, 1-0.2, 1+0.2)
        loss = -min(ratio * adv, clipped * adv) + 0.1 * KL(current || sft_ref)
        update(loss)

    # 4. Evaluate
    eval_metrics = run_eval(current_policy)
    log(iteration, eval_metrics)
```

#### Reward Function

```python
def compute_scheduling_reward(verification_result, step_cost=0.05):
    if verdict == PASS:
        margin = mean(resource_margin, affinity_margin, rack_margin, priority_margin)
        return 1.0 + margin * 0.5 - step_cost
    elif verdict == FAIL:
        severity = violation_count / total_constraints
        return -0.3 - 0.7 * severity
    else:  # ERROR (parse failure, invalid action)
        return -1.0
```

- `step_cost` incentivizes fewer recovery steps
- Margin bonus rewards headroom beyond bare constraint satisfaction
- Severity scaling gives informative gradient for different failure modes

#### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N rollouts/scenario | 8 | GRPO paper recommends 4-16 |
| GRPO iterations | 5-10 | Eval each iteration, early stop |
| Clip epsilon | 0.2 | Standard PPO/GRPO |
| KL coefficient | 0.1 | Prevent DPO-style distribution collapse |
| Learning rate | 1e-5 | Lower than SFT (2e-4) for stability |
| Reference model | SFT checkpoint (frozen LoRA) | Shared base weights, separate adapters |

#### Memory Budget (single 96GB GPU)

| Phase | VRAM | Notes |
|-------|------|-------|
| Rollout (inference) | ~25 GB | QLoRA 4-bit + KV cache |
| Training | ~35 GB | Policy + ref (shared base) + optimizer |
| Peak | ~45-50 GB | Alternating, not concurrent |

## Evaluation

### Metrics

- Recovery rate (primary)
- Average steps
- Rejection rate
- Average margin (GRPO-specific)
- Unsafe action rate (NoVerify ablation)

### Comparison Matrix

| Experiment | Model | Training | Expected Recovery |
|-----------|-------|----------|-------------------|
| Zero-shot | GPT-5.4 | None | ~60-70% |
| Few-shot | GPT-5.4 | None | ~85-90% |
| SFT | Qwen3-14B | SFT only | ~80-85% |
| **SFT + GRPO** | **Qwen3-14B** | **SFT + GRPO** | **~90-95%** |

### Ablations (time permitting)

1. Binary reward (PASS=+1/FAIL=-1) vs margin reward — value of continuous signal
2. N=4 vs 8 vs 16 rollouts — sample efficiency
3. KL coefficient sweep — exploration vs stability

## File Structure

```
domains/cluster/
├── __init__.py
├── config.py                   # build_cluster_domain_config()
├── manager.py                  # ClusterManager(BaseSystemManager)
├── checkers.py                 # 5 constraint checkers
├── observation.py              # ClusterObserver
├── tools.py                    # 6 tools
├── scenarios/
│   ├── __init__.py
│   └── loader.py               # ClusterScenarioLoader + parameterized generation
├── failsafe.py                 # Priority-first scheduling fallback
└── prompts/
    ├── system_prompt.py
    └── tool_schemas.py

silr/training/
├── grpo_trainer.py             # NEW: Step-level GRPO loop
└── (existing: sft_trainer.py, dpo_trainer.py, reward.py, data_loader.py)

examples/cluster_scheduling.py  # NEW: runnable demo
tests/test_cluster_domain.py    # NEW
tests/test_cluster_scenarios.py # NEW
tests/test_grpo_trainer.py      # NEW
```

## Timeline (4 weeks)

| Week | Deliverable | Commit scope |
|------|------------|-------------|
| W1 | Cluster domain: manager, checkers, tools, scenarios, tests | `feat: add GPU cluster scheduling domain` |
| W2 | SFT data collection + SFT training + baseline eval | `feat: cluster scenario variants` + training artifacts |
| W3 | GRPO trainer implementation + first training run | `feat: add step-level GRPO trainer` |
| W4 | Hyperparameter tuning, ablations, final eval, docs | `docs: cluster domain and GRPO results` |
