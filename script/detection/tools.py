"""Provider-neutral definitions of the three tools the agent can call.

Each spec is {name, description, parameters (JSON schema)}; the LLM
backends translate this into their own wire format (OpenAI wraps it in
{"type": "function", "function": ...}, Anthropic renames parameters to
input_schema).
"""
from __future__ import annotations

from typing import Any, Dict, List

INSPECT = "inspect_event_log"
RUN_SCRIPT = "run_detection_script"
FINISH = "finish"

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": INSPECT,
        "description": (
            "Evaluate a read-only pandas expression against the input event log "
            "(bound to `df`) to explore its structure before writing detection code. "
            "Returns the repr of the result, truncated to ~4000 characters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A single pandas expression using `df`, e.g. \"df['Activity'].value_counts().head(20)\"",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": RUN_SCRIPT,
        "description": (
            "Write the given Python script to disk and execute it against the real "
            "input CSV. The script must read the input path from sys.argv[1] and "
            "write its output (original columns plus pattern_detected / "
            "detected_pattern_types) to sys.argv[2] — never hardcode paths. Returns "
            "whether it succeeded, stdout/stderr on failure, and on success the "
            "detection rate and per-type breakdown (never ground truth)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "python_code": {"type": "string", "description": "Complete, runnable Python script."},
            },
            "required": ["python_code"],
        },
    },
    {
        "name": FINISH,
        "description": "Call this once you are satisfied with a detection run and want to stop iterating.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Brief explanation of why you're stopping here."},
                "rate_reasonable": {
                    "type": "boolean",
                    "description": "Whether the final detection rate looks reasonable (not too aggressive/conservative).",
                },
                "selected_iteration": {
                    "type": "integer",
                    "description": "Which run_detection_script call (by its 1-based order) to use as the final output. Omit to use the most recent successful run.",
                },
            },
            "required": ["reason"],
        },
    },
]


def tool_specs(include_inspect: bool = True) -> List[Dict[str, Any]]:
    if include_inspect:
        return TOOL_SPECS
    return [t for t in TOOL_SPECS if t["name"] != INSPECT]
