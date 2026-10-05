"""Console output: per-dataset metrics and per-condition averages."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .context import DetectionContext

STYLES = ["strict", "moderate", "generous"]


def summarize_context(context: DetectionContext, input_csv: Path) -> Dict[str, Any]:
    final_rate: Dict[str, Any] = {}
    last_success = next(
        (r for r in reversed(context.improvement_history) if r.get("rate_metrics")), None,
    )
    if last_success:
        final_rate = last_success["rate_metrics"]

    result: Dict[str, Any] = {
        "dataset":         input_csv.name,
        "errors_detected": final_rate.get("errors_detected", 0),
        "total_rows":      final_rate.get("total_rows", 0),
    }
    for style in STYLES:
        result[f"detection_metrics_{style}"] = (
            context.evaluation_metrics.get(f"metrics_{style}", {}).get("metrics", {})
        )
    return result


def print_dataset_result(context: DetectionContext) -> None:
    print("\nOUTPUTS:")
    print(f"  - Detected:      {context.output_csv}")
    if context.evaluation_path:
        print(f"  - Evaluation:    {context.evaluation_path}")
    if context.iter_log_path:
        print(f"  - Iteration Log: {context.iter_log_path}")

    if not context.evaluation_metrics:
        return
    print("\nDETECTION METRICS (ALL STYLES):")
    for style in STYLES:
        style_metrics = context.evaluation_metrics.get(f"metrics_{style}", {})
        m = style_metrics.get("metrics", {})
        cm = style_metrics.get("confusion_matrix", {})
        c = style_metrics.get("counts", {})
        print(f"\n  {style.upper()}:")
        print(f"    Precision: {m.get('precision', 0):.4f}")
        print(f"    Recall:    {m.get('recall', 0):.4f}")
        print(f"    F1 Score:  {m.get('f1_score', 0):.4f}")
        print(f"    Accuracy:  {m.get('accuracy', 0):.4f}")
        total_rows = c.get("total_rows", 0) or 1
        pred_patterns = c.get("predicted_patterns", 0)
        print(f"    Patterns: {pred_patterns} / {total_rows} ({pred_patterns / total_rows * 100:.1f}%)")
        print(f"    TP: {cm.get('TP', 0)} | FP: {cm.get('FP', 0)} | "
              f"FN: {cm.get('FN', 0)} | TN: {cm.get('TN', 0)}")


def print_condition_summary(condition: str, results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    n = len(results)
    print("\n" + "=" * 70)
    print(f"SUMMARY - {condition} - {n} DATASETS")
    print("=" * 70)
    for style in STYLES:
        key = f"detection_metrics_{style}"
        avg_p = sum(r[key].get("precision", 0) for r in results) / n
        avg_r = sum(r[key].get("recall",    0) for r in results) / n
        avg_f1 = sum(r[key].get("f1_score", 0) for r in results) / n
        print(f"\n{style.upper()} METRICS:")
        print(f"  Average Detection:")
        print(f"    Precision: {avg_p:.4f}")
        print(f"    Recall:    {avg_r:.4f}")
        print(f"    F1 Score:  {avg_f1:.4f}")
