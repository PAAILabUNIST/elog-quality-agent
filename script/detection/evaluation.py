"""Write <dataset>.evaluation.json for the iteration the agent selected."""
from __future__ import annotations

import json

from .context import DetectionContext
from .iteration_log import strip_to_metrics
from .metrics import compute_metrics_from_output


def evaluate(context: DetectionContext) -> DetectionContext:
    print("\n[STEP 2/2] Evaluation")

    if not context.evaluation_metrics:
        metrics_result = compute_metrics_from_output(context.output_csv, context.ground_truth_csv)
        if "error" not in metrics_result:
            context.evaluation_metrics = metrics_result

    if context.evaluation_metrics:
        save_evaluation_report(context)
    return context


def save_evaluation_report(context: DetectionContext) -> None:
    best_record = next(
        (r for r in context.improvement_history
         if r.get("llm_best_iteration_selected") is not None),
        None,
    )
    report = {
        "dataset":         context.dataset_stem,
        "evaluation_type": "detection",
        "llm_best_selection": {
            "selected_iteration": best_record.get("llm_best_iteration_selected") if best_record else None,
            "reason":             best_record.get("llm_best_iteration_reason")   if best_record else None,
        },
        "metrics":             context.evaluation_metrics,
        "improvement_history": [strip_to_metrics(r) for r in context.improvement_history],
    }
    eval_path = context.output_csv.parent / f"{context.input_csv.stem}.evaluation.json"
    eval_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    context.evaluation_path = eval_path
