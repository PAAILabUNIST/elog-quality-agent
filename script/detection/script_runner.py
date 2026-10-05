"""run_detection_script tool: save the model's script, execute it on the real
input CSV, and score the output (rate metrics for the model, GT metrics
logged silently)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .config import REQUIRED_OUTPUT_COLUMNS
from .context import DetectionContext
from .metrics import compute_metrics_from_output, compute_rate_metrics


class ScriptRunner:
    def __init__(self, scripts_dir: Path, timeout_seconds: int = 240):
        self.scripts_dir = scripts_dir
        self.timeout_seconds = timeout_seconds

    def run(
        self, context: DetectionContext, python_code: str, iteration: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Returns (text fed back to the model, iteration record for the log)."""
        record = self._new_record(context, python_code, iteration)
        iter_output = iteration_output_csv(context, iteration)

        if not python_code.strip():
            record["error"] = "Empty python_code"
            return "Error: python_code was empty.", record

        try:
            script_path = self._write_script(context.dataset_stem, python_code, iteration)
            record["script_path"] = str(script_path)

            t0 = time.time()
            result = self._execute(script_path, context.input_csv, iter_output)
            elapsed = time.time() - t0
            record["execution_time_seconds"] = round(elapsed, 2)

            if result.returncode != 0:
                err_text = format_stderr(result.stderr)
                record["error"] = err_text[:500]
                print(f"  Iteration {iteration}: FAILED (exit {result.returncode}), {elapsed:.1f}s")
                return f"Script failed (exit code {result.returncode}).\n\nSTDERR:\n{err_text[:2000]}", record

            validate_output(iter_output)

            rm = compute_rate_metrics(iter_output)
            record["rate_metrics"] = rm
            record["iter_output_path"] = str(iter_output)

            # GT metrics per run (silent - never fed back to the model)
            gt_result = compute_metrics_from_output(iter_output, context.ground_truth_csv)
            if "error" not in gt_result:
                record["metrics_strict"] = gt_result["metrics_strict"]
                record["metrics_moderate"] = gt_result["metrics_moderate"]
                record["metrics_generous"] = gt_result["metrics_generous"]

            f1_mod = record["metrics_moderate"]["metrics"]["f1_score"] if record["metrics_moderate"] else "n/a"
            print(
                f"  Iteration {iteration}: "
                f"flagged={rm['errors_detected']} ({rm['error_rate']:.1%}), "
                f"mod_F1={f1_mod}, time={elapsed:.1f}s"
            )
            return rate_summary_for_model(iteration, elapsed, rm), record

        except subprocess.TimeoutExpired:
            record["execution_time_seconds"] = self.timeout_seconds
            record["error"] = f"Script execution exceeded timeout of {self.timeout_seconds}s."
            print(f"  Iteration {iteration}: TIMEOUT after {self.timeout_seconds:.1f}s")
            return f"Script timed out after {self.timeout_seconds}s. Optimize it to run faster.", record

        except Exception as e:
            record["error"] = str(e)
            return f"Error: {e}", record

    # ------------------------------------------------------------------

    @staticmethod
    def _new_record(context: DetectionContext, python_code: str, iteration: int) -> Dict[str, Any]:
        return {
            "iteration":              iteration,
            "step":                   "detector",
            "dataset":                context.dataset_stem,
            "error":                  None,
            "rate_metrics":           None,
            "script_path":            None,
            "iter_output_path":       None,
            "metrics_strict":         None,
            "metrics_moderate":       None,
            "metrics_generous":       None,
            "execution_time_seconds": None,
            "generated_code":         python_code,
            "llm_stop_decision":      None,
            "llm_stop_reason":        None,
            "llm_rate_reasonable":    None,
        }

    def _write_script(self, dataset_stem: str, python_code: str, iteration: int) -> Path:
        dataset_scripts_dir = self.scripts_dir / dataset_stem
        dataset_scripts_dir.mkdir(parents=True, exist_ok=True)
        out_path = dataset_scripts_dir / f"detect_{dataset_stem}_v{iteration}.py"
        out_path.write_text(python_code, encoding="utf-8")
        return out_path

    def _execute(self, script_path: Path, input_csv: Path, output_csv: Path) -> subprocess.CompletedProcess:
        # KMP_DUPLICATE_LIB_OK works around a Windows-specific hard
        # crash (Intel OpenMP "Error #15: duplicate runtime
        # initialized") that happens when a process loads both an
        # MKL-linked NumPy/SciPy and PyTorch's own bundled OpenMP
        # runtime - which every generated script that imports
        # sentence_transformers does. Without it, the whole subprocess
        # can hard-crash with no Python traceback at all, silently
        # killing the run rather than raising a catchable error.
        script_env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
        return subprocess.run(
            [sys.executable, str(script_path), str(input_csv), str(output_csv)],
            capture_output=True, text=True, timeout=self.timeout_seconds, env=script_env,
        )


def iteration_output_csv(context: DetectionContext, iteration: int) -> Path:
    """<dataset>.detected.csv -> <dataset>.detected.v<N>.csv"""
    base = context.output_csv
    return base.with_name(f"{base.stem}.v{iteration}{base.suffix}")


def validate_output(out_csv: Path) -> None:
    df = pd.read_csv(out_csv)
    missing = [c for c in REQUIRED_OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Output missing columns: {missing}")


def format_stderr(stderr: str) -> str:
    stderr = stderr or ""
    if "Traceback" in stderr:
        text = "\n".join(stderr.strip().split("\n")[-12:])
    else:
        text = stderr
    if "iteritems" in stderr:
        text += (
            "\n\nNOTE: df.iteritems() is removed in Pandas 2.x. "
            "Replace ALL .iteritems() calls with .items()."
        )
    return text


def rate_summary_for_model(iteration: int, elapsed: float, rm: Dict[str, Any]) -> str:
    summary = (
        f"Run {iteration} succeeded in {elapsed:.1f}s.\n"
        f"Rows flagged: {rm['errors_detected']} / {rm['total_rows']} ({rm['error_rate']:.1%})\n"
    )
    if rm.get("pattern_type_distribution"):
        summary += "Pattern type breakdown:\n"
        for t, c in sorted(rm["pattern_type_distribution"].items(), key=lambda x: -x[1]):
            summary += f"  - {t}: {c}\n"
    summary += (
        "\n(Ground truth is intentionally not shown to you - judge reasonableness "
        "from the rate and distribution alone, and how stable it is if you re-run "
        "with adjustments.)"
    )
    return summary
