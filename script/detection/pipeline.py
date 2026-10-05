"""Run the agent + evaluation over every dataset of one condition
(domain x model x prompt)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from . import config
from .agent import AgenticDetector
from .context import DetectionContext, resolve_ground_truth, slugify
from .evaluation import evaluate
from .iteration_log import IterationMetricsLogger
from .llm.base import LLMBackend
from .reporting import print_dataset_result, summarize_context


def process_dataset(detector: AgenticDetector, input_csv: Path, output_root: Path) -> DetectionContext:
    dataset_stem = slugify(input_csv.stem)
    dataset_output_dir = output_root / dataset_stem
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    iter_log_path = dataset_output_dir / f"{input_csv.stem}.iterations.json"
    context = DetectionContext(
        dataset_stem=dataset_stem,
        input_csv=input_csv,
        ground_truth_csv=resolve_ground_truth(input_csv),
        output_csv=dataset_output_dir / f"{input_csv.stem}.detected.csv",
        iter_log_path=iter_log_path,
        iter_logger=IterationMetricsLogger(iter_log_path),
    )

    context = detector.execute(context)

    if not context.output_csv.exists():
        print("  Skipping evaluation — no output file produced after all iterations.")
        return context

    return evaluate(context)


def run_condition(
    llm: LLMBackend, domain: str, prompt: str, dataset_glob: str = config.DEFAULT_DATASET_GLOB,
) -> List[Dict[str, Any]]:
    """Detect on every <domain> dataset with <prompt>; returns one result row per dataset."""
    data_dir = config.REPO_ROOT / config.DOMAINS[domain]
    out_dir = config.output_dir(domain, llm.tag, prompt)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = AgenticDetector(
        llm=llm,
        prompt_template=config.load_prompt(prompt),
        scripts_dir=config.generated_scripts_dir(domain, llm.tag, prompt),
    )

    csv_files = sorted(data_dir.rglob(dataset_glob))
    print("\n" + "#" * 70)
    print(f"CONDITION: {config.condition_name(domain, llm.tag, prompt)}")
    print(f"Model: {llm.model} | Datasets: {len(csv_files)} | Output: {out_dir}")
    print("#" * 70)
    if not csv_files:
        print(f"No CSV files matching {dataset_glob!r} in {data_dir}")
        return []

    results: List[Dict[str, Any]] = []
    for in_csv in csv_files:
        try:
            print("\n" + "=" * 70)
            print(f"DATASET: {in_csv.name}")
            print("=" * 70)
            context = process_dataset(detector, in_csv, out_dir)
            results.append(summarize_context(context, in_csv))
            print_dataset_result(context)
        except Exception as e:
            print(f"\nERROR processing {in_csv.name}: {e}")
            continue
    return results
