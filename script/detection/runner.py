"""Command-line entry shared by detect_gpt.py and detect_claude.py: loops over
every (prompt, domain) pair and runs the pipeline for the given LLM backend."""
from __future__ import annotations

import argparse
from typing import Type

from . import config
from .llm.base import LLMBackend
from .pipeline import run_condition
from .reporting import print_condition_summary


def parse_args(model_label: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run {model_label} detection over all domains and prompt conditions.",
    )
    parser.add_argument(
        "--domain", nargs="+", choices=list(config.DOMAINS), default=list(config.DOMAINS),
        help="Which dataset domains to run (default: all).",
    )
    parser.add_argument(
        "--prompt", nargs="+", choices=list(config.PROMPTS), default=list(config.PROMPTS),
        help="Which prompt conditions to run (default: all).",
    )
    parser.add_argument(
        "--datasets", default=config.DEFAULT_DATASET_GLOB,
        help='Glob for input files, e.g. "*homonymous_noLabel.csv" (default: %(default)s).',
    )
    return parser.parse_args()


def main(backend_cls: Type[LLMBackend]) -> None:
    args = parse_args(backend_cls.tag)
    llm = backend_cls()

    print("Pre-downloading sentence-transformer model...")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
    print("Model ready.")

    all_results = {}
    for prompt in args.prompt:
        for domain in args.domain:
            condition = config.condition_name(domain, llm.tag, prompt)
            all_results[condition] = run_condition(llm, domain, prompt, args.datasets)
            print_condition_summary(condition, all_results[condition])

    if len(all_results) > 1:
        print("\n" + "#" * 70)
        print("ALL CONDITIONS")
        print("#" * 70)
        for condition, results in all_results.items():
            print_condition_summary(condition, results)
