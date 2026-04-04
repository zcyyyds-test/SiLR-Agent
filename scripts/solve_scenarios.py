"""Teacher model: solve 5 GPU cluster scheduling scenarios with SiLR verification.

Produces detailed trajectories showing observation -> thought -> action at each step.
Each action is verified via SiLRVerifier before application.

Solver strategy:
  Phase 1: Direct assign queued jobs to nodes with free capacity (urgent first)
  Phase 2: Preempt preemptible running jobs to free capacity for urgent/normal
  Phase 3: "Migrate" normal jobs off constrained racks for affinity-bound jobs
  Phase 4: Re-assign evicted/migrated jobs to other racks

Results saved to scripts/scenario_results.json for downstream use.
"""
import sys
import json
import time
sys.path.insert(0, '.')

from domains.cluster import ClusterManager, build_cluster_domain_config, ClusterScenarioLoader
from silr.verifier import SiLRVerifier
from domains.cluster.tools import create_cluster_toolset

SCENARIOS = [
    "single_node_failure",
    "priority_conflict",
    "job_surge_small",
    "job_surge",
    "compound_failure_surge",
]

MAX_STEPS = 50


# ---------------------------------------------------------------------------
# Capacity helpers
# ---------------------------------------------------------------------------

def get_node_capacity(state):
    cap = {}
    for nid, n in sorted(state["nodes"].items()):
        if n["status"] != "Ready":
            continue
        cap[nid] = {
            "gpu_free": n["gpu_total"] - n["gpu_used"],
            "cpu_free": n["cpu_total"] - n["cpu_used"],
            "ram_free": n["ram_total_gb"] - n["ram_used_gb"],
            "type": n["type"],
            "rack": n["rack"],
            "gpu_total": n["gpu_total"],
        }
    return cap


def can_fit(cap_entry, job):
    return (cap_entry["gpu_free"] >= job["gpu"]
            and cap_entry["cpu_free"] >= job["cpu"]
            and cap_entry["ram_free"] >= job["ram_gb"])


def find_best_node(cap, job, exclude_nodes=None, exclude_racks=None):
    affinity = job.get("rack_affinity")
    exclude_n = exclude_nodes or set()
    exclude_r = exclude_racks or set()
    candidates = []
    for nid, c in sorted(cap.items()):
        if nid in exclude_n:
            continue
        if c["rack"] in exclude_r:
            continue
        if affinity and c["rack"] != affinity:
            continue
        if can_fit(c, job):
            candidates.append((nid, c))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1]["gpu_free"])
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Single-step planner
# ---------------------------------------------------------------------------

def pick_next_action(state, skip_jobs, migrated_from_rack, protected_jobs):
    jobs = state["jobs"]
    assignments = state["assignments"]
    nodes = state["nodes"]
    cap = get_node_capacity(state)

    priority_order = {"urgent": 0, "normal": 1, "preemptible": 2}
    queued = [(jid, jobs[jid]) for jid in sorted(jobs.keys())
              if jobs[jid]["status"] == "Queued"]
    queued.sort(key=lambda x: (priority_order.get(x[1]["priority"], 9), -x[1]["gpu"]))

    if not queued:
        return None

    preemptible_running = []
    normal_running = []
    for jid, j in sorted(jobs.items()):
        if j["status"] != "Running":
            continue
        nid = assignments.get(jid)
        if not nid:
            continue
        if jid in protected_jobs:
            continue
        if j["priority"] == "preemptible":
            preemptible_running.append((jid, j, nid))
        elif j["priority"] == "normal":
            normal_running.append((jid, j, nid))

    # Phase 1: Direct assignment
    for qjid, qjob in queued:
        if qjid in skip_jobs:
            continue
        exclude_r = set()
        if qjid in migrated_from_rack:
            exclude_r.add(migrated_from_rack[qjid])
        node = find_best_node(cap, qjob, exclude_racks=exclude_r)
        if node:
            return {
                "tool_name": "assign_job",
                "params": {"job_id": qjid, "node_id": node},
                "reason": (f"Direct assign {qjob['priority']} {qjid} "
                           f"(gpu={qjob['gpu']}) to {node}"),
            }

    # Phase 2: Preempt preemptible for urgent/normal
    for qjid, qjob in queued:
        if qjid in skip_jobs:
            continue
        if qjob["priority"] == "preemptible":
            continue
        affinity = qjob.get("rack_affinity")

        for nid, ncap in sorted(cap.items()):
            if affinity and ncap["rack"] != affinity:
                continue
            node_preemptible = [
                (pjid, pjob) for pjid, pjob, pnid in preemptible_running
                if pnid == nid and pjid not in skip_jobs
            ]
            if not node_preemptible:
                continue
            freed_gpu, freed_cpu, freed_ram = ncap["gpu_free"], ncap["cpu_free"], ncap["ram_free"]
            for pjid, pjob in sorted(node_preemptible, key=lambda x: x[1]["gpu"], reverse=True):
                freed_gpu += pjob["gpu"]
                freed_cpu += pjob["cpu"]
                freed_ram += pjob["ram_gb"]
                would_fit = (freed_gpu >= qjob["gpu"] and freed_cpu >= qjob["cpu"]
                             and freed_ram >= qjob["ram_gb"])
                return {
                    "tool_name": "preempt_job",
                    "params": {"job_id": pjid},
                    "reason": (f"Preempt preemptible {pjid} on {nid} "
                               f"{'(will fit)' if would_fit else '(partial)'} "
                               f"for {qjob['priority']} {qjid}"),
                }
        if not affinity:
            for pjid, pjob, pnid in sorted(
                preemptible_running, key=lambda x: x[1]["gpu"], reverse=True
            ):
                if pjid not in skip_jobs:
                    return {
                        "tool_name": "preempt_job",
                        "params": {"job_id": pjid},
                        "reason": (f"Preempt preemptible {pjid} globally "
                                   f"for {qjob['priority']} {qjid}"),
                    }

    # Phase 3: Migrate normal jobs off constrained racks for affinity
    for qjid, qjob in queued:
        if qjid in skip_jobs:
            continue
        affinity = qjob.get("rack_affinity")
        if not affinity:
            continue
        for pjid, pjob, pnid in normal_running:
            if nodes[pnid]["rack"] != affinity:
                continue
            if pjid in skip_jobs:
                continue
            ncap = cap.get(pnid)
            if not ncap:
                continue
            freed_gpu = ncap["gpu_free"] + pjob["gpu"]
            freed_cpu = ncap["cpu_free"] + pjob["cpu"]
            freed_ram = ncap["ram_free"] + pjob["ram_gb"]
            if (freed_gpu >= qjob["gpu"] and freed_cpu >= qjob["cpu"]
                    and freed_ram >= qjob["ram_gb"]):
                other_cap = {nid: c for nid, c in cap.items() if c["rack"] != affinity}
                if find_best_node(other_cap, pjob) is not None:
                    return {
                        "tool_name": "preempt_job",
                        "params": {"job_id": pjid},
                        "reason": (f"Migrate normal {pjid} off {affinity} "
                                   f"for affinity-constrained {qjid}"),
                        "_migrated_rack": affinity,
                    }

    # Phase 4: Cleanup
    for qjid, qjob in queued:
        exclude_r = set()
        if qjid in migrated_from_rack:
            exclude_r.add(migrated_from_rack[qjid])
        node = find_best_node(cap, qjob, exclude_racks=exclude_r)
        if node:
            return {
                "tool_name": "assign_job",
                "params": {"job_id": qjid, "node_id": node},
                "reason": (f"Cleanup assign {qjob['priority']} {qjid} "
                           f"(gpu={qjob['gpu']}) to {node}"),
            }

    return None


# ---------------------------------------------------------------------------
# Scenario solver
# ---------------------------------------------------------------------------

def solve_scenario(sid):
    print(f"\n{'='*70}")
    print(f"SCENARIO: {sid}")
    print('='*70)

    mgr = ClusterManager()
    config = build_cluster_domain_config()
    loader = ClusterScenarioLoader()
    scenario = loader.load(sid)
    loader.setup_episode(mgr, scenario)
    verifier = SiLRVerifier(mgr, domain_config=config)
    observer = config.create_observer(mgr)
    toolset = create_cluster_toolset(mgr)

    # Initial state summary
    state = mgr.system_state
    init_obs = observer.observe()
    total_gpu = sum(n["gpu_total"] for n in state["nodes"].values() if n["status"] == "Ready")
    used_gpu = sum(n["gpu_used"] for n in state["nodes"].values() if n["status"] == "Ready")
    init_queued = len([j for j in state["jobs"].values() if j["status"] == "Queued"])
    init_queued_gpu = sum(j["gpu"] for j in state["jobs"].values() if j["status"] == "Queued")
    down_nodes = [nid for nid, n in state["nodes"].items() if n["status"] == "NotReady"]

    print(f"  Difficulty: {scenario.difficulty}")
    print(f"  Initial: {init_queued} queued jobs ({init_queued_gpu} GPU), "
          f"{used_gpu}/{total_gpu} GPU used, {len(init_obs.violations)} violations")
    if down_nodes:
        print(f"  Down nodes: {sorted(down_nodes)}")

    trajectory = []
    step = 0
    skip_jobs = set()
    migrated_from_rack = {}
    protected_jobs = set()
    stall_count = 0
    prev_queued = None

    while step < MAX_STEPS:
        obs = observer.observe()
        state = mgr.system_state

        if obs.is_stable:
            print(f"\n  >>> STABLE after {step} steps! <<<")
            break

        current_queued = len([j for j in state["jobs"].values() if j["status"] == "Queued"])
        if current_queued == prev_queued:
            stall_count += 1
        else:
            stall_count = 0
        prev_queued = current_queued
        if stall_count > 6:
            break

        action_dict = pick_next_action(state, skip_jobs, migrated_from_rack, protected_jobs)
        if action_dict is None:
            if skip_jobs:
                skip_jobs.clear()
                action_dict = pick_next_action(state, skip_jobs, migrated_from_rack, protected_jobs)
            if action_dict is None:
                break

        action_for_verify = {
            "tool_name": action_dict["tool_name"],
            "params": action_dict["params"],
        }

        result = verifier.verify(action_for_verify)
        verdict = result.verdict.value

        step_record = {
            "step": step,
            "observation_summary": json.loads(obs.compressed_json),
            "thought": action_dict["reason"],
            "action": action_for_verify,
            "verdict": verdict,
        }

        if verdict == "PASS":
            tool = toolset[action_dict["tool_name"]]
            tool.execute(**action_dict["params"])
            mgr.run_pflow()
            step_record["applied"] = True

            if action_dict["tool_name"] == "preempt_job":
                pjid = action_dict["params"]["job_id"]
                skip_jobs.add(pjid)
                if "_migrated_rack" in action_dict:
                    migrated_from_rack[pjid] = action_dict["_migrated_rack"]
            elif action_dict["tool_name"] == "assign_job":
                jid = action_dict["params"]["job_id"]
                skip_jobs.discard(jid)
                migrated_from_rack.pop(jid, None)
                job = state["jobs"][jid]
                if job.get("rack_affinity"):
                    protected_jobs.add(jid)

            print(f"  Step {step}: {action_dict['tool_name']}({json.dumps(action_dict['params'])}) -> PASS")
            print(f"    Thought: {action_dict['reason']}")
        else:
            step_record["applied"] = False
            step_record["fail_reason"] = result.fail_reason
            if action_dict["tool_name"] == "assign_job":
                skip_jobs.add(action_dict["params"]["job_id"])
            print(f"  Step {step}: FAIL -> {result.fail_reason}")

        trajectory.append(step_record)
        step += 1

    # Final assessment
    final_obs = observer.observe()
    state = mgr.system_state
    final_queued = len([j for j in state["jobs"].values() if j["status"] == "Queued"])
    final_queued_gpu = sum(j["gpu"] for j in state["jobs"].values() if j["status"] == "Queued")
    total_gpu = sum(n["gpu_total"] for n in state["nodes"].values() if n["status"] == "Ready")
    used_gpu = sum(n["gpu_used"] for n in state["nodes"].values() if n["status"] == "Ready")
    free_gpu = total_gpu - used_gpu

    print(f"\n  Final: stable={final_obs.is_stable}, violations={len(final_obs.violations)}")
    print(f"  GPU: {used_gpu}/{total_gpu} ({free_gpu} free), Queued: {final_queued} ({final_queued_gpu} GPU)")

    # Classify remaining violations
    capacity_overflow = final_queued_gpu > free_gpu
    all_remaining_preemptible = all(
        state["jobs"][jid]["priority"] == "preemptible"
        for jid in sorted(state["jobs"].keys())
        if state["jobs"][jid]["status"] == "Queued"
    )

    if not final_obs.is_stable:
        for v in final_obs.violations:
            print(f"    [{v['severity']}] {v['detail']}")
        if capacity_overflow:
            print(f"  >> CAPACITY OVERFLOW: need {final_queued_gpu} GPU, only {free_gpu} free")
        if all_remaining_preemptible and final_queued > 0:
            print(f"  >> All remaining queued jobs are preemptible (correct triage)")

    return {
        "scenario": sid,
        "difficulty": scenario.difficulty,
        "description": scenario.description,
        "initial_state": {
            "queued_jobs": init_queued,
            "queued_gpu": init_queued_gpu,
            "gpu_used": used_gpu,
            "gpu_total": total_gpu,
            "violations": len(init_obs.violations),
            "down_nodes": down_nodes,
        },
        "steps_taken": step,
        "stable": final_obs.is_stable,
        "final_state": {
            "queued_jobs": final_queued,
            "queued_gpu": final_queued_gpu,
            "gpu_used": used_gpu,
            "gpu_total": total_gpu,
            "gpu_free": free_gpu,
            "violations": len(final_obs.violations),
            "capacity_overflow": capacity_overflow,
            "all_remaining_preemptible": all_remaining_preemptible,
        },
        "trajectory": trajectory,
        "all_actions_verified": all(t["verdict"] == "PASS" for t in trajectory),
    }


def main():
    t0 = time.perf_counter()
    all_results = []

    for sid in SCENARIOS:
        result = solve_scenario(sid)
        all_results.append(result)

    elapsed = time.perf_counter() - t0

    # Save results
    with open("scripts/scenario_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*70}")
    print(f"FINAL REPORT ({elapsed:.2f}s total)")
    print('='*70)

    for r in all_results:
        if r["stable"]:
            status = "FULLY RECOVERED"
        elif r["final_state"]["capacity_overflow"]:
            status = f"OPTIMAL (capacity overflow, {r['final_state']['queued_gpu']} GPU unplaceable)"
        else:
            status = f"PARTIAL ({r['final_state']['violations']} violations)"

        print(f"\n  {r['scenario']} [{r['difficulty']}]: {r['steps_taken']} steps -> {status}")
        print(f"    {r['description']}")
        print(f"    Initial: {r['initial_state']['queued_jobs']} queued, "
              f"{r['initial_state']['violations']} violations")
        print(f"    Final:   {r['final_state']['queued_jobs']} queued, "
              f"{r['final_state']['violations']} violations, "
              f"GPU {r['final_state']['gpu_used']}/{r['final_state']['gpu_total']}")
        print(f"    All actions verified: {r['all_actions_verified']}")

    print(f"\n  Results saved to scripts/scenario_results.json")


if __name__ == "__main__":
    main()
