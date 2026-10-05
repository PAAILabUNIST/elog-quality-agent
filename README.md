# Event Log Imperfection Detection (reproduction repo)

LLM agents that write and run their own Python detection scripts against
process-mining event logs to find imperfection patterns (collateral,
homonymous, synonymous, polluted, distorted, form-based), then score their
own output against ground truth. Four conditions: GPT vs. Claude, each with
a "zero-shot" and a "few-shot" prompt, run against two dataset domains:
synthetic (pub/credit) and natural (BPIC11/BPIC15).

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Set API keys before running:

```bash
export OPENAI_API_KEY=...       # for the two GPT conditions
export ANTHROPIC_API_KEY=...    # for the two Claude conditions
# optional overrides:
export OPENAI_MODEL=gpt-5.5             # default
export ANTHROPIC_MODEL=claude-opus-4-8  # default
```

First run will download the `all-MiniLM-L6-v2` sentence-transformer model
(used for synonymous-label detection) - needs internet access once, then
it's cached locally.

## Running it

There is one entry point per model. Each one runs **every domain x every
prompt** in turn (`bpic` + `synthetic`, `zero_shot` + `few_shot`):

```bash
python script/detect_gpt.py
python script/detect_claude.py
```

Narrow it down with flags (all optional, combinable):

```bash
python script/detect_gpt.py --domain bpic                   # only the BPIC (natural) logs
python script/detect_gpt.py --prompt few_shot               # only the few-shot prompt
python script/detect_claude.py --domain synthetic --prompt zero_shot
python script/detect_claude.py --datasets "*homonymous_noLabel.csv"   # re-run just matching datasets
```

Each (domain, model, prompt) condition writes to its own folders:

```
data_<domain>_detected_<model>_<prompt>/     # results, e.g. data_bpic_detected_gpt_zero_shot/
scripts_<domain>_detected_<model>_<prompt>/  # every .py the agent wrote, at every iteration
```

with `domain` in `bpic | synthetic`, `model` in `gpt | claude`,
`prompt` in `zero_shot | few_shot`.

Per dataset, the agent inspects the log, writes a detection script, runs it
against the real data, iterates (up to 20 turns, with a turn-budget safety
net that forces it to write code if it spends too long just inspecting),
and self-selects a final iteration via its own `finish()` call. That
iteration is scored against ground truth and written out as
`<dataset>.detected.csv` + `<dataset>.evaluation.json` +
`<dataset>.iterations.json` (plus `<dataset>.detected.vN.csv` for every
intermediate iteration).

## Repo layout

```
script/
  detect_gpt.py           # entry point: GPT    (OpenAI)
  detect_claude.py        # entry point: Claude (Anthropic)
  detection/              # shared logic, read roughly top to bottom:
    runner.py             #   CLI flags; loops over prompts x domains
    pipeline.py           #   per condition: loops over datasets -> agent -> evaluation
    agent.py              #   the agent loop: system prompt, tool dispatch, finish/selection
    tools.py              #   the 3 tool definitions (inspect / run script / finish)
    sandbox.py            #   inspect_event_log: read-only pandas sandbox
    script_runner.py      #   run_detection_script: save + execute the model's script, score it
    metrics.py            #   rate metrics (shown to agent) + ground-truth P/R/F1 (hidden)
    evaluation.py         #   writes <dataset>.evaluation.json
    iteration_log.py      #   writes <dataset>.iterations.json
    context.py            #   per-dataset state + file-name helpers (GT lookup, anonymizing)
    reporting.py          #   console output / averages
    config.py             #   paths, domains, prompts, output-folder naming
    llm/
      base.py             #   provider-neutral Conversation interface + shared retry policy
      openai_backend.py   #   GPT specifics (chat completions, streaming, tool format)
      anthropic_backend.py#   Claude specifics (messages API, streaming, tool format)

prompts/
  zero_shot.txt   # zero-shot prompt
  few_shot.txt    # few-shot prompt

data_synthetic/   # synthetic pub/credit logs, 7 injected error types x noLabel/withLabel pairs  (domain "synthetic")
data_natural/     # BPIC11/BPIC15 logs, same structure                                         (domain "bpic")
```

To add a new domain or prompt, add an entry to `DOMAINS` / `PROMPTS` in
`script/detection/config.py`. The only GPT-vs-Claude differences live in
`script/detection/llm/`; everything else is shared.
