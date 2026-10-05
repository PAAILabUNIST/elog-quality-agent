"""Run Claude detection over every domain (bpic, synthetic) x prompt (zero_shot, few_shot).

    python script/detect_claude.py
    python script/detect_claude.py --domain synthetic --prompt zero_shot
    python script/detect_claude.py --datasets "*homonymous_noLabel.csv"
"""
from detection.llm.anthropic_backend import AnthropicBackend
from detection.runner import main

if __name__ == "__main__":
    main(AnthropicBackend)
