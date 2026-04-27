"""System prompt for the SWE Agentless two-stage agent."""
from __future__ import annotations


def build_swe_system_prompt(manager, tool_schemas: list) -> str:
    inst = manager.instance
    return (
        "You are a code-repair agent. You will be given a Python repository "
        "and a bug description. Fix the bug by following these two steps:\n\n"
        "Step 1 — call `localize` with a list of 'file_path:line_no' strings "
        "naming the lines most likely responsible for the bug. You can list "
        "multiple candidates.\n\n"
        "Step 2 — call `patch` with a unified diff that fixes the bug. The "
        "diff must apply cleanly at the repository root with `git apply`. "
        "After this call the verifier will run the repository's test suite.\n\n"
        f"Problem statement:\n{inst.problem_statement}\n"
    )


def build_swe_tool_schemas(manager) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "localize",
                "description": "Store a ranked list of suspect 'file:line' locations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "locations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 'path/to/file.py:line_no' strings.",
                        },
                    },
                    "required": ["locations"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "patch",
                "description": "Submit a unified diff that fixes the bug. Terminal action.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch": {
                            "type": "string",
                            "description": "Unified diff (git-apply compatible).",
                        },
                    },
                    "required": ["patch"],
                },
            },
        },
    ]
