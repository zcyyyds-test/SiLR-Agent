# GPU Cluster Scheduling Domain

A reference SiLR domain for GPU cluster job scheduling, with a complete SFT → GRPO post-training pipeline that uses the SiLR verifier as the reward signal.

## Overview

This domain models a multi-rack GPU cluster where an LLM agent makes job placement, preemption, migration, and node lifecycle decisions. Every proposed action is verified on a shadow copy of the cluster state before execution — actions that would violate resource limits, affinity constraints, priority ordering, or queue-clearance rules are rejected before reaching the real system.

The same verifier that gates inference also drives reinforcement learning: during GRPO training, action acceptance/rejection produces a reward signal, allowing the model to learn from its own rollouts without supervised exemplars.

## Topology and Constraints

- **15 GPU nodes** across 3 racks (`rack-a`, `rack-b`, `rack-c`)
- **Heterogeneous hardware**: standard (4 GPU / 64 CPU / 256 GB), highmem (4 GPU / 64 CPU / 512 GB), fat (8 GPU / 128 CPU / 1 TB)
- **6 tools**: `assign_job`, `preempt_job`, `migrate_job`, `restore_node`, `drain_node`, `scale_job`
- **5 constraint checkers**: ResourceCapacity, Affinity, RackSpread, Priority, QueueClearance

The verifier uses ResourceCapacity and Affinity for per-action safety; RackSpread, Priority, and QueueClearance are episode-level objectives surfaced via the observer.

## Failure Scenarios

17 scenarios across 6 categories:

| Category | Examples |
|----------|----------|
| Single-node hardware failures | Random node down, requires migration of running jobs |
| Rack-level outages | Full rack drain, cascade across affinity-constrained jobs |
| Workload surges | Urgent job burst that exceeds queue clearance budget |
| Resource fragmentation | Jobs sized mismatched to free node capacities |
| Priority and affinity conflicts | Urgent jobs blocked by lower-priority preemptible jobs on the wrong rack |
| Compound failures | Multiple modes simultaneously (rack outage + urgent surge) |

Scenario specs live in [`scenarios/loader.py`](scenarios/loader.py).

## Training Pipeline

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Teacher model  │ →  │ SFT data     │ →  │  SFT training   │
│  (GPT-5.4)      │    │  cleaning    │    │  (QLoRA 4-bit)  │
└─────────────────┘    └──────────────┘    └────────┬────────┘
                                                     │
                       ┌──────────────┐              ▼
                       │ GRPO update  │ ←  ┌─────────────────┐
                       │  (PPO loss + │    │ Online rollout  │
                       │   GRPO adv)  │    │ + SiLR reward   │
                       └──────────────┘    └─────────────────┘
```

**Stage 1 — SFT data collection**: Teacher model (GPT-5.4 via OpenAI-compatible API) is run on each scenario; trajectories are recorded as `(observation, thought, action)` triples. See `scripts/collect_sft_data.py`.

**Stage 2 — Data cleaning**: Multiple collection runs are merged, deduplicated, and the observation format is replayed against the current observer to ensure schema consistency. Missing chain-of-thought is back-filled by the teacher model. See `scripts/clean_sft_data.py`.

**Stage 3 — SFT training**: Qwen3-14B + LoRA (r=64, α=128) trained for 3 epochs on the cleaned dataset using QLoRA 4-bit quantization. See `scripts/train_sft.py`.

**Stage 4 — GRPO post-training**: Step-level Group Relative Policy Optimization. For each scenario, multiple rollouts are collected; per-step rewards come from SiLR verification (`+0.45` for accepted action, `−0.50` for rejected, `+1.00` recovery bonus). Advantages are computed within each scenario group, then a clipped PPO objective updates the LoRA weights. See `scripts/train_grpo.py`.

## Results

| Model | Recovery Rate (51 episodes) |
|-------|----------------------------|
| GPT-5.4 (teacher) | 67% (34/51) |
| Qwen3-14B + SFT | 88.2% (45/51) |
| **Qwen3-14B + SFT + GRPO** | **94.1% (48/51)** |

Eval protocol: 3 repeats × 17 scenarios, greedy decoding (temperature=0), max 10 steps per episode.

**Key improvements**:
- The hardest scenario (`compound_failure_surge`) went from **0% → 100%** recovery after GRPO post-training
- All 15 already-solved scenarios maintained 100% (no regression)
- GRPO converged in 3 iterations (~5h on a single H100-class GPU)

## Application Context

The cluster topology, failure modes, and constraint model are inspired by GPU cluster operation patterns at **TSUBAME 4.0**, the H100-based supercomputer at Institute of Science Tokyo.

**Future work**:
- Validate the trained agent on a 4-8 GPU TSUBAME 4.0 allocation against real workload traces
- Integrate as a verifier-gated *advisor* alongside PBS Professional (TSUBAME's production scheduler), where the LLM proposes scheduling decisions and the SiLR verifier checks safety before execution

The "advisor mode" deployment pattern is the standard safe path for LLM agents entering production: the LLM proposes, the verifier vetoes, and existing rule-based systems retain final execution authority.
