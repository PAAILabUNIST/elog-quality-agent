"""The detection agent: one growing conversation in which the model calls
tools until it decides it's done.

  - inspect_event_log:    explore the input CSV before committing to code
  - run_detection_script: write + execute a candidate script, see the
                          real stdout/stderr/rate metrics as the result
  - finish:               the model itself decides when it's done

The turn loop keeps dispatching whatever tool calls come back until the
model calls finish() or the turn cap is hit - control flow is driven by the
model, not by a fixed Python retry loop. Provider differences (GPT vs.
Claude) live entirely in the LLM backend's Conversation.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import tools
from .context import DetectionContext
from .llm.base import LLMBackend, ToolCall
from .sandbox import inspect_event_log
from .script_runner import ScriptRunner

SYSTEM_PROMPT = """You are an expert in Event Log Imperfection Pattern Detection, working autonomously.

You have three tools:
- inspect_event_log(expression): explore the input event log (available as `df` in a read-only pandas sandbox) before writing detection code. Use it to check column values, activity label formats, timestamp granularity, etc.
- run_detection_script(python_code): write and execute a full detection script against the real input file. You get back real stdout/stderr on failure, or the detection rate + per-type breakdown on success.
- finish(reason, rate_reasonable, selected_iteration): call this yourself once you are satisfied. Ground truth is intentionally never shown to you, so judge "reasonable" from the flagged rate and its stability across your own runs, not from any hidden score.

CRITICAL RULES:
1. Be thorough — inspect every column carefully for potential imperfections. Prefer using inspect_event_log at least once before your first run_detection_script call on a dataset you have not seen before.
2. Be appropriately aggressive — avoid being both too conservative (missing real errors) and too aggressive (over-flagging clean entries).
3. Detect each imperfection type independently and do not conflate them.
4. The sentence-transformers model "all-MiniLM-L6-v2" is already cached locally — do not attempt to re-download it.
5. If a run_detection_script call fails or produces an implausible rate, fix it and call run_detection_script again yourself — do not stop until you're confident, but do not loop forever on a fix that isn't converging.
6. Always act by calling a tool. Only plain text with no tool call, or a call to finish, ends your turn."""

MSG_NO_TOOL_CALL = "Continue by calling inspect_event_log, run_detection_script, or finish."
MSG_INSPECT_NUDGE = (
    "You have spent several turns calling inspect_event_log without writing "
    "any detection code. Stop inspecting now and call run_detection_script with "
    "your current best understanding - you can still refine it in later iterations."
)
MSG_INSPECT_DISABLED = (
    "Your inspection budget is exhausted - inspect_event_log is no longer "
    "available for this dataset. Call run_detection_script now with your "
    "best attempt, or call finish if you are confident no script is needed."
)
OUTPUT_PATH_PLACEHOLDER = (
    "<provided automatically as sys.argv[2] each time you call run_detection_script - never hardcode it>"
)


@dataclass
class FinishState:
    called: bool = False
    reason: Optional[str] = None
    rate_reasonable: Optional[bool] = None
    selected_iteration: Optional[int] = None


class AgenticDetector:
    def __init__(
        self,
        llm: LLMBackend,
        prompt_template: str,
        scripts_dir: Path,
        max_turns: int = 20,
        script_timeout_seconds: int = 240,
        inspection_nudge_turn: int = 6,
        inspection_force_turn: int = 10,
    ):
        self.llm = llm
        self.prompt_template = prompt_template
        self.runner = ScriptRunner(scripts_dir, timeout_seconds=script_timeout_seconds)
        self.max_turns = max_turns
        # Safety net: without this, a model can spend its entire turn budget
        # calling inspect_event_log and never write a detection script at all
        # (observed on messier real-world logs) - nudge it after a few turns
        # of pure inspection, then force it by disabling the tool outright.
        self.inspection_nudge_turn = inspection_nudge_turn
        self.inspection_force_turn = inspection_force_turn

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def execute(self, context: DetectionContext) -> DetectionContext:
        print(f"\n[AGENT] Detection - {context.dataset_stem}")

        df_input = pd.read_csv(context.input_csv)
        history: List[Dict[str, Any]] = []
        finish = FinishState()
        convo = self.llm.new_conversation(SYSTEM_PROMPT, self._build_user_prompt(context))
        inspect_enabled = True

        for turn in range(self.max_turns):
            label = f"  Turn {turn + 1}/{self.max_turns}:"
            reply = convo.ask(tools.tool_specs(include_inspect=inspect_enabled))

            if not reply.tool_calls:
                preview = reply.text.strip().replace("\n", " ")[:200]
                print(f"{label} no tool call - nudging. Model said: {preview!r}")
                convo.add_user_text(MSG_NO_TOOL_CALL)
                continue

            results = [
                (call, self._dispatch(call, context, df_input, history, finish, label))
                for call in reply.tool_calls
            ]

            notes: List[str] = []
            if not history and not finish.called:
                if turn + 1 == self.inspection_force_turn:
                    inspect_enabled = False
                    print(f"{label} no run_detection_script call yet - disabling inspect_event_log to force a write.")
                    notes.append(MSG_INSPECT_DISABLED)
                elif turn + 1 == self.inspection_nudge_turn:
                    print(f"{label} no run_detection_script call yet - nudging.")
                    notes.append(MSG_INSPECT_NUDGE)

            convo.add_tool_results(results, notes)

            if finish.called:
                break
        else:
            print(f"  Reached max turns ({self.max_turns}) without the agent calling finish().")

        if history:
            last = history[-1]
            last["llm_stop_decision"] = "stop" if finish.called else "max_turns"
            last["llm_stop_reason"] = finish.reason or "Reached max turns without calling finish()"
            last["llm_rate_reasonable"] = finish.rate_reasonable
            if context.iter_logger:
                context.iter_logger.log(last)

        context.improvement_history = history
        return self._finalize(context, history, finish.selected_iteration)

    def _build_user_prompt(self, context: DetectionContext) -> str:
        return (
            self.prompt_template
            .replace("{INPUT_CSV_PATH}", context.llm_safe_input_path)
            .replace("{OUTPUT_CSV_PATH}", OUTPUT_PATH_PLACEHOLDER)
            .replace("{DATASET_STEM}", context.llm_safe_stem)
        )

    def _dispatch(
        self,
        call: ToolCall,
        context: DetectionContext,
        df_input: pd.DataFrame,
        history: List[Dict[str, Any]],
        finish: FinishState,
        label: str,
    ) -> str:
        """Run one tool call and return the text sent back to the model."""
        if call.name == tools.INSPECT:
            expr = call.args.get("expression", "")
            result_text = inspect_event_log(df_input, expr)
            preview = result_text.strip().replace("\n", " ")[:200]
            print(f"{label} inspect_event_log({expr!r}) -> {preview!r}")
            return result_text

        if call.name == tools.RUN_SCRIPT:
            result_text, record = self.runner.run(
                context, call.args.get("python_code", ""), iteration=len(history) + 1,
            )
            history.append(record)
            if context.iter_logger:
                context.iter_logger.log(record)
            return result_text

        if call.name == tools.FINISH:
            finish.called = True
            finish.reason = call.args.get("reason", "")
            finish.rate_reasonable = call.args.get("rate_reasonable")
            finish.selected_iteration = call.args.get("selected_iteration")
            print(f"{label} Agent finished: {finish.reason}")
            return "Session ended."

        return f"Unknown tool: {call.name}"

    # ------------------------------------------------------------------
    # Finalization — pick the run the agent settled on and copy it out
    # ------------------------------------------------------------------

    def _finalize(
        self,
        context: DetectionContext,
        history: List[Dict[str, Any]],
        selected_iteration: Optional[int],
    ) -> DetectionContext:
        successful = [r for r in history if r.get("rate_metrics") is not None]
        if not successful:
            print("  No successful run — skipping ground truth evaluation.")
            return context

        chosen = None
        if selected_iteration is not None:
            chosen = next((r for r in successful if r["iteration"] == selected_iteration), None)
            if chosen is None:
                print(f"  finish() selected iteration {selected_iteration} not found among successful runs; using most recent.")
        if chosen is None:
            chosen = successful[-1]
        chosen["llm_best_iteration_selected"] = chosen["iteration"]
        chosen["llm_best_iteration_reason"] = "Selected by the agent's own finish() call."

        shutil.copy(Path(chosen["iter_output_path"]), context.output_csv)

        if chosen.get("metrics_moderate"):
            context.evaluation_metrics = {
                "metrics_strict":   chosen["metrics_strict"],
                "metrics_moderate": chosen["metrics_moderate"],
                "metrics_generous": chosen["metrics_generous"],
            }
            m = chosen["metrics_generous"]["metrics"]
            print(
                f"  Ground truth (generous): "
                f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1_score']:.3f}"
            )

        return context
