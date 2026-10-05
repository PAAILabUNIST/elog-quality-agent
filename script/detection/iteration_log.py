"""Append-only JSON log of every run_detection_script iteration (<dataset>.iterations.json)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Fields kept from an iteration record; everything else (e.g. the generated
# code itself) is dropped - the code is already saved as a .py file.
_LOGGED_FIELDS = {
    "iteration", "step", "dataset", "error",
    "rate_metrics",
    "metrics_strict", "metrics_moderate", "metrics_generous",
    "execution_time_seconds", "script_path", "iter_output_path",
    "llm_stop_decision", "llm_stop_reason", "llm_rate_reasonable",
    "llm_best_iteration_selected", "llm_best_iteration_reason",
}


def strip_to_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in record.items() if k in _LOGGED_FIELDS}


class IterationMetricsLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("[]", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        clean_record = strip_to_metrics(record)
        clean_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        data.append(clean_record)

        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
