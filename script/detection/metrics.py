"""Scoring a detected CSV.

- compute_rate_metrics: label-free stats (flag rate, per-type counts) - the
  only feedback the agent ever sees.
- compute_metrics_from_output: ground-truth precision/recall/F1 under three
  matching styles - computed silently, never shown to the agent.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd


# ============================================================
# RATE-ONLY METRICS (shown to the agent)
# ============================================================

def compute_rate_metrics(output_csv: Path) -> Dict[str, Any]:
    df = pd.read_csv(output_csv)
    total = len(df)
    predicted = int(df["pattern_detected"].fillna(False).astype(bool).sum())
    error_rate = predicted / total if total > 0 else 0.0

    type_dist: Dict[str, int] = {}
    for cell in df.loc[df["pattern_detected"] == True, "detected_pattern_types"].dropna():
        for t in str(cell).split("|"):
            t = t.strip()
            if t:
                type_dist[t] = type_dist.get(t, 0) + 1

    return {
        "total_rows":                total,
        "errors_detected":           predicted,
        "error_rate":                error_rate,
        "pattern_type_distribution": type_dist,
    }


# ============================================================
# GROUND TRUTH METRICS (hidden from the agent)
# ============================================================

def parse_ground_truth_label(label_str) -> set:
    """Extract pattern types from a GT label like "HOMONYMOUS(...) | POLLUTED(...)"."""
    if pd.isna(label_str) or str(label_str).strip() == "":
        return set()

    pattern_types = set()
    matches = re.findall(r'([A-Z]+)\s*\(', str(label_str))

    CANONICAL_MAP = {
        "FORMBASED":     "form-based",
        "AUTOFORMBASED": "form-based",
        "POLLUTED":      "polluted",
        "DISTORTED":     "distorted",
        "SYNONYMOUS":    "synonymous",
        "COLLATERAL":    "collateral",
        "HOMONYMOUS":    "homonymous",
        "EMPTY":         "empty",
    }
    for match in matches:
        canonical = CANONICAL_MAP.get(match.upper())
        if canonical:
            pattern_types.add(canonical)
    return pattern_types


def compute_metrics_from_output(output_csv: Path, ground_truth_csv: Path) -> Dict[str, Any]:
    """A flagged row counts as a true positive only if its predicted types match GT:
    - STRICT:   predicted types == GT types
    - MODERATE: predicted types are a subset of GT types
    - GENEROUS: predicted and GT types share at least one type
    """
    df_gt = pd.read_csv(ground_truth_csv)
    if 'label' not in df_gt.columns:
        return {"error": "Ground truth CSV missing 'label' column",
                "metrics_strict": None, "metrics_moderate": None, "metrics_generous": None}

    df_out = pd.read_csv(output_csv)
    if 'pattern_detected' not in df_out.columns or 'detected_pattern_types' not in df_out.columns:
        return {"error": "Output CSV missing required columns",
                "metrics_strict": None, "metrics_moderate": None, "metrics_generous": None}

    if len(df_gt) != len(df_out):
        return {"error": f"Row count mismatch: GT={len(df_gt)}, Output={len(df_out)}",
                "metrics_strict": None, "metrics_moderate": None, "metrics_generous": None}

    def normalize_pattern_type(t: str) -> str:
        if not t:
            return ""
        return t.lower().strip().replace("_", "-").replace("formbased", "form-based")

    df_gt['gt_pattern_types'] = df_gt['label'].apply(
        lambda x: {normalize_pattern_type(t) for t in parse_ground_truth_label(x)})
    df_gt['gt_has_pattern'] = df_gt['gt_pattern_types'].apply(lambda x: len(x) > 0)

    y_true = df_gt['gt_has_pattern']
    y_pred = df_out['pattern_detected'].fillna(False).astype(bool)

    CANONICAL_MAP = {
        "form_based": "form-based", "formbased": "form-based",
        "form-based": "form-based", "autoformbased": "form-based",
        "distorted": "distorted",   "polluted":  "polluted",
        "homonymous": "homonymous", "synonymous": "synonymous",
        "collateral": "collateral", "empty": "empty",
    }

    def canonicalize(t: str) -> str:
        if not t or pd.isna(t):
            return ""
        return CANONICAL_MAP.get(t.lower().strip().replace("_", "-"), t.lower().strip())

    def get_predicted_types(idx):
        pred_types = set()
        all_types = str(df_out.iloc[idx].get("detected_pattern_types", ""))
        for t in all_types.split("|"):
            c = canonicalize(t)
            if c:
                pred_types.add(c)
        return pred_types

    def get_gt_types(idx):
        gt = {canonicalize(t) for t in df_gt.iloc[idx]["gt_pattern_types"]}
        gt.discard("")
        return gt

    def is_correct_strict(idx):
        if not y_pred.iloc[idx] or not y_true.iloc[idx]:
            return False
        return get_gt_types(idx) == get_predicted_types(idx)

    def is_correct_moderate(idx):
        if not y_pred.iloc[idx] or not y_true.iloc[idx]:
            return False
        return get_predicted_types(idx).issubset(get_gt_types(idx))

    def is_correct_generous(idx):
        if not y_pred.iloc[idx] or not y_true.iloc[idx]:
            return False
        return len(get_gt_types(idx) & get_predicted_types(idx)) > 0

    correct_strict = [is_correct_strict(i) for i in range(len(df_gt))]
    correct_moderate = [is_correct_moderate(i) for i in range(len(df_gt))]
    correct_generous = [is_correct_generous(i) for i in range(len(df_gt))]

    def calc(correct_detections, style_name):
        correct = pd.Series(correct_detections, index=df_gt.index)
        tp = int((y_pred & y_true & correct).sum())
        fp = int((y_pred & ~y_true).sum())
        fn = int((y_true & (~y_pred | (y_pred & ~correct))).sum())
        tn = int((~y_pred & ~y_true).sum())
        total = len(df_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        pattern_df = df_out[df_out["pattern_detected"] == True]
        type_dist: Dict[str, int] = {}
        if not pattern_df.empty:
            for _, row in pattern_df.iterrows():
                for t in get_predicted_types(row.name):
                    type_dist[t] = type_dist.get(t, 0) + 1

        return {
            "style": style_name,
            "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
            "metrics": {
                "precision":           round(precision, 4),
                "recall":              round(recall, 4),
                "f1_score":            round(f1, 4),
                "accuracy":            round(accuracy, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
            },
            "counts": {
                "total_rows":            total,
                "ground_truth_patterns": int(y_true.sum()),
                "predicted_patterns":    int(y_pred.sum()),
                "correctly_detected":    tp,
            },
            "pattern_type_distribution": type_dist,
        }

    return {
        "metrics_strict":   calc(correct_strict,   "STRICT"),
        "metrics_moderate": calc(correct_moderate, "MODERATE"),
        "metrics_generous": calc(correct_generous, "GENEROUS"),
    }
