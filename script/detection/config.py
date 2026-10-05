"""Repo paths, experiment grid (domains x prompts) and output-folder naming."""
from __future__ import annotations

from pathlib import Path

# script/detection/config.py -> repo root is two levels above this package
REPO_ROOT = Path(__file__).resolve().parents[2]

# domain name -> input data folder (each holds <log>/<...>_noLabel.csv + _withLabel.csv pairs)
DOMAINS = {
    "bpic":      "data_natural",
    "synthetic": "data_synthetic",
}

# prompt condition -> prompt template file
PROMPTS = {
    "zero_shot": "prompts/zero_shot.txt",
    "few_shot":  "prompts/few_shot.txt",
}

DEFAULT_DATASET_GLOB = "*_noLabel.csv"

# Columns the agent's generated script must add to its output CSV
REQUIRED_OUTPUT_COLUMNS = ["pattern_detected", "detected_pattern_types"]

KNOWN_PATTERN_TYPES = [
    "collateral", "homonymous", "synonymous",
    "polluted", "distorted", "form-based", "formbased", "empty",
]

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"


def condition_name(domain: str, model_tag: str, prompt: str) -> str:
    """e.g. ("bpic", "gpt", "zero_shot") -> "bpic_detected_gpt_zero_shot"."""
    return f"{domain}_detected_{model_tag}_{prompt}"


def output_dir(domain: str, model_tag: str, prompt: str) -> Path:
    """Where detected CSVs / evaluation / iteration logs are written."""
    return REPO_ROOT / f"data_{condition_name(domain, model_tag, prompt)}"


def generated_scripts_dir(domain: str, model_tag: str, prompt: str) -> Path:
    """Where every .py the agent wrote (at every iteration) is kept."""
    return REPO_ROOT / f"scripts_{condition_name(domain, model_tag, prompt)}"


def load_prompt(prompt: str) -> str:
    path = REPO_ROOT / PROMPTS[prompt]
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
