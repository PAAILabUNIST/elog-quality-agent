"""Run GPT detection over every domain (bpic, synthetic) x prompt (zero_shot, few_shot).

    python script/detect_gpt.py
    python script/detect_gpt.py --domain bpic --prompt few_shot
    python script/detect_gpt.py --datasets "*homonymous_noLabel.csv"
"""
from detection.llm.openai_backend import OpenAIBackend
from detection.runner import main

if __name__ == "__main__":
    main(OpenAIBackend)
