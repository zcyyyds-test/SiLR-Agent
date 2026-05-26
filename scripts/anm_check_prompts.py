"""Verify the ANM domain's ReAct-loop wiring without requiring an LLM.

Exercises the now-populated DomainConfig fields end to end:
  - build_tool_schemas → well-formed OpenAI tool definitions with real device
    bounds embedded.
  - get_valid_device_ids → action-parser-friendly id maps.
  - build_system_prompt → all expected sections, references real device ids
    and constraint windows.
  - create_observer (with_observer=True) → ANMObserver.observe() emits a
    correctly-typed Observation with raw={"simulator": ...} (Panel#3 P3).

Run from repo root:
    PYTHONPATH=. python scripts/anm_check_prompts.py
"""

from __future__ import annotations

import json

from domains.anm import (
    ANMScenarioLoader,
    GymANMManager,
    build_anm_domain_config,
)


def main() -> None:
    cfg = build_anm_domain_config(with_observer=True)
    mgr = GymANMManager(seed=42)
    # apply a known stressed scenario so the observation has interesting content
    ANMScenarioLoader().setup_episode(
        mgr, ANMScenarioLoader().load("medium_seed42_default")
    )

    # --- 1. tool schemas ---
    schemas = cfg.build_tool_schemas(mgr)
    assert len(schemas) == 2, f"expected 2 action schemas, got {len(schemas)}"
    names = [s["function"]["name"] for s in schemas]
    assert set(names) == {"set_generator_setpoint", "set_storage_setpoint"}, names
    for s in schemas:
        func = s["function"]
        assert "description" in func and func["description"]
        assert "parameters" in func and "properties" in func["parameters"]
        assert func["parameters"].get("required"), "must declare required params"
    # device bounds should be embedded in descriptions (MW unit)
    gen_desc = next(
        s for s in schemas if s["function"]["name"] == "set_generator_setpoint"
    )["function"]["parameters"]["properties"]["gen_id"]["description"]
    assert "MW" in gen_desc and "P∈" in gen_desc, gen_desc
    storage_desc = next(
        s for s in schemas if s["function"]["name"] == "set_storage_setpoint"
    )["function"]["parameters"]["properties"]["storage_id"]["description"]
    assert "SoC∈" in storage_desc, storage_desc
    print("OK tool schemas (2 schemas, device bounds in description)")

    # --- 2. valid device ids ---
    ids = cfg.get_valid_device_ids(mgr)
    assert set(ids.keys()) == {"gen_id", "storage_id"}, ids
    assert ids["gen_id"] == mgr.get_generator_ids()
    assert ids["storage_id"] == mgr.get_storage_ids()
    print(f"OK valid device ids: {ids}")

    # --- 3. system prompt ---
    prompt = cfg.build_system_prompt(mgr, schemas)
    for section in ("## Role", "## Available Action Tools", "## Safety Constraints",
                    "## Network Topology", "## Protocol"):
        assert section in prompt, f"system prompt missing section: {section}"
    # references real device ids
    for g in mgr.get_generator_ids():
        assert f"{g}" in prompt
    assert f"base MVA: {mgr.base_mva}" in prompt
    print(f"OK system prompt ({len(prompt)} chars, 5 sections)")

    # --- 4. observer ---
    observer = cfg.create_observer(mgr)
    obs = observer.observe()
    # P3 fix: raw is a dict, not the Simulator directly
    assert isinstance(obs.raw, dict), f"Observation.raw must be dict, got {type(obs.raw)}"
    assert "simulator" in obs.raw, "raw must contain 'simulator' key"
    # compressed JSON parses + has the new device-bounds sections
    cj = json.loads(obs.compressed_json)
    assert "generators" in cj and cj["generators"], "observer must expose generator bounds"
    assert "storage" in cj and cj["storage"], "observer must expose storage bounds"
    assert "checkers" in cj, "observer must include checker summaries"
    # medium_seed42_default is a known stress scenario → expect violations + not stable
    assert obs.is_stable is False, "stressed scenario must be flagged not stable"
    assert obs.violations, "stressed scenario must report violations"
    print(f"OK observer: {len(obs.violations)} violations, is_stable={obs.is_stable}")

    print("\n--- sample system prompt (first 60 lines) ---")
    print("\n".join(prompt.splitlines()[:60]))
    print("--- sample compressed observation ---")
    print(json.dumps(cj, indent=2)[:1200])
    print("\nPROMPTS+OBSERVER WIRING OK")


if __name__ == "__main__":
    main()
