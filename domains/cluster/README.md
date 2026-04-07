# GPU Cluster Scheduling Domain

A reference SiLR domain for GPU cluster job scheduling, with a complete SFT → GRPO post-training pipeline that uses the SiLR verifier as the reward signal.

## Overview

This domain models a multi-rack GPU cluster where an LLM agent makes job placement, preemption, migration, and node lifecycle decisions. Every proposed action is verified on a shadow copy of the cluster state before execution — actions that would violate resource limits, affinity constraints, priority ordering, or queue-clearance rules are rejected before reaching the real system.

The same verifier that gates inference also drives reinforcement learning: during GRPO training, action acceptance/rejection produces a reward signal, allowing the model to learn from its own rollouts without supervised exemplars.

## Topology and Constraints

- **15 GPU nodes** across 3 racks (`rack-a`, `rack-b`, `rack-c`)
- **Heterogeneous hardware**: standard, high-memory, and fat nodes with varying GPU counts and memory budgets
- **6 tools**: `assign_job`, `preempt_job`, `migrate_job`, `restore_node`, `drain_node`, `scale_job`
- **5 constraint checkers**: ResourceCapacity, Affinity, RackSpread, Priority, Queue

The verifier uses ResourceCapacity and Affinity for per-action safety; RackSpread, Priority, and Queue are episode-level objectives surfaced via the observer.

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

The agent is post-trained in two stages:

**SFT stage** — A teacher model (GPT-5.4) is run on each scenario; the resulting `(observation, thought, action)` trajectories are merged, deduplicated, and replayed against the current observer to ensure schema consistency. Missing chain-of-thought is back-filled by the teacher. The cleaned dataset trains a Qwen3-14B + LoRA student via QLoRA. See `scripts/collect_sft_data.py`, `clean_sft_data.py`, and `train_sft.py`.

**GRPO stage** — Step-level Group Relative Policy Optimization. For each scenario, multiple rollouts are collected; per-step rewards come from SiLR verification (`+0.45` for accepted action, `−0.50` for rejected, `+1.00` recovery bonus). Advantages are normalized within each scenario group, then a clipped PPO objective updates the LoRA weights. See `scripts/train_grpo.py`.

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

## Application Context

The cluster topology, failure modes, and constraint model are inspired by GPU cluster operation patterns at **TSUBAME 4.0**, the H100-based supercomputer at Institute of Science Tokyo.

**Future work**:
- Validate the trained agent on an 8 GPU TSUBAME 4.0 allocation against real workload traces
- Integrate as a verifier-gated *advisor* alongside PBS Professional (TSUBAME's production scheduler), where the LLM proposes scheduling decisions and the SiLR verifier checks safety before execution
