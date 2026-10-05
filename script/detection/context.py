"""Per-dataset state passed through the pipeline, plus dataset-naming helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import KNOWN_PATTERN_TYPES
from .iteration_log import IterationMetricsLogger


def anonymize_stem(stem: str) -> str:
    """Hide the injected pattern type from the dataset name before the LLM
    sees it (e.g. "pub_..._homonymous_noLabel" -> "pub_..._dataset_nolabel"),
    so the file name doesn't leak the answer."""
    result = stem.lower()
    for pattern_type in KNOWN_PATTERN_TYPES:
        result = re.sub(re.escape(pattern_type), "dataset", result, flags=re.IGNORECASE)
    return result


def slugify(stem: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)


def resolve_ground_truth(no_label_csv: Path) -> Path:
    """<name>_noLabel.csv -> sibling <name>_withLabel.csv."""
    if "_noLabel" not in no_label_csv.stem:
        raise ValueError(f"Expected *_noLabel.csv, got {no_label_csv.name}")
    gt_path = no_label_csv.parent / no_label_csv.name.replace("_noLabel.csv", "_withLabel.csv")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground truth file: {gt_path}")
    return gt_path


@dataclass
class DetectionContext:
    dataset_stem:     str
    input_csv:        Path
    ground_truth_csv: Path
    output_csv:       Path

    improvement_history: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_metrics:  Dict[str, Any] = field(default_factory=dict)
    evaluation_path:     Optional[Path] = None
    iter_log_path:       Optional[Path] = None
    iter_logger:         Optional[IterationMetricsLogger] = None

    @property
    def llm_safe_stem(self) -> str:
        return anonymize_stem(self.dataset_stem)

    @property
    def llm_safe_input_path(self) -> str:
        return str(self.input_csv.parent / anonymize_stem(self.input_csv.stem)) + ".csv"
