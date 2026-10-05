"""Common interface the agent talks to, so the agent loop is identical for GPT
and Claude. Each provider only has to say how to:

  - build its client
  - turn the neutral tool specs into its own tool format
  - send a request and parse the reply into an LLMReply
  - append assistant turns / tool results / plain user text to the history
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ToolCall:
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class LLMReply:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)


class Conversation(ABC):
    """One growing chat with the model for a single dataset."""

    def __init__(self, backend: "LLMBackend", system_prompt: str, user_prompt: str):
        self.backend = backend
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = []
        self._start(user_prompt)

    @abstractmethod
    def _start(self, user_prompt: str) -> None: ...

    @abstractmethod
    def ask(self, tool_specs: List[Dict[str, Any]]) -> LLMReply:
        """Send the history, append the assistant's turn to it, return the reply."""

    @abstractmethod
    def add_user_text(self, text: str) -> None: ...

    @abstractmethod
    def add_tool_results(self, results: List[Tuple[ToolCall, str]], notes: List[str]) -> None:
        """Answer every tool call of the last assistant turn, plus optional
        extra user notes (e.g. "stop inspecting") sent alongside them."""


class LLMBackend(ABC):
    tag: str               # used in output folder names, e.g. "gpt" / "claude"
    api_key_env: str
    model_env: str
    default_model: str

    def __init__(self, temperature: float = 0.1, max_retries: int = 4):
        if not os.environ.get(self.api_key_env):
            raise RuntimeError(f"{self.api_key_env} not set")
        self.model = os.environ.get(self.model_env, self.default_model)
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = self.build_client()

    @abstractmethod
    def build_client(self, api_key: Optional[str] = None) -> Any: ...

    @abstractmethod
    def new_conversation(self, system_prompt: str, user_prompt: str) -> Conversation: ...

    def send_with_retries(self, send, kwargs: Dict[str, Any]) -> Any:
        """Call send(client, kwargs) with the shared retry policy:
        - drop `temperature` once if the model rejects it (some reasoning
          models don't support a custom temperature)
        - on a connection error, rebuild the client and back off.
        """
        use_temperature = True
        for attempt in range(self.max_retries):
            try:
                if use_temperature:
                    kwargs["temperature"] = self.temperature
                else:
                    kwargs.pop("temperature", None)
                return send(self.client, kwargs)
            except Exception as e:
                msg = str(e)
                if use_temperature and "temperature" in msg:
                    use_temperature = False
                    continue
                if "connection" in msg.lower() and attempt < self.max_retries - 1:
                    wait_time = min(30, 5 * (attempt + 1))
                    # str(e) on the SDKs' APIConnectionError is hardcoded to
                    # the literal "Connection error." - the real underlying
                    # httpx exception is chained as __cause__, so surface that
                    # too or every failure looks identical no matter the cause.
                    cause = e.__cause__
                    print(
                        f"[WARN] Connection error on attempt {attempt + 1}/{self.max_retries} "
                        f"(underlying: {type(cause).__name__ if cause else 'unknown'}: {cause}); "
                        f"rebuilding client and retrying in {wait_time}s..."
                    )
                    self.client = self.build_client(api_key=self.client.api_key)
                    time.sleep(wait_time)
                    continue
                raise
        raise RuntimeError("LLM call: exhausted retries")
