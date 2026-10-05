"""GPT via the OpenAI Chat Completions API."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from .base import Conversation, LLMBackend, LLMReply, ToolCall


class OpenAIBackend(LLMBackend):
    tag = "gpt"
    api_key_env = "OPENAI_API_KEY"
    model_env = "OPENAI_MODEL"
    default_model = "gpt-5.5"

    def build_client(self, api_key: Optional[str] = None) -> OpenAI:
        """Keep-alive connection reuse disabled: a stale/black-holed
        keep-alive connection being reused is what turns a single transient
        network blip into "Connection error." on every subsequent call on
        the same client - a fresh connection per request avoids that."""
        return OpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(120.0, connect=30.0),
            max_retries=0,
            http_client=httpx.Client(
                timeout=httpx.Timeout(120.0, connect=30.0),
                limits=httpx.Limits(max_keepalive_connections=0),
            ),
        )

    def new_conversation(self, system_prompt: str, user_prompt: str) -> Conversation:
        return OpenAIConversation(self, system_prompt, user_prompt)

    @staticmethod
    def format_tools(tool_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"type": "function", "function": spec} for spec in tool_specs]


class OpenAIConversation(Conversation):
    backend: OpenAIBackend

    def _start(self, user_prompt: str) -> None:
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def ask(self, tool_specs: List[Dict[str, Any]]) -> LLMReply:
        # Streamed rather than a single blocking call, and this is
        # load-bearing, not an optimization: api.openai.com is served through
        # Cloudflare's edge, which enforces its own idle/read timeout
        # independent of the client timeout. A non-streaming request sends
        # zero bytes back while a reasoning model is silently "thinking",
        # which can exceed that edge timeout and kill the connection no
        # matter how long httpx was told to wait. Streaming keeps bytes
        # flowing so the connection never goes idle long enough to be killed.
        kwargs: Dict[str, Any] = {
            "model":    self.backend.model,
            "messages": self.messages,
            "stream":   True,
        }
        if tool_specs:
            kwargs["tools"] = self.backend.format_tools(tool_specs)
            kwargs["tool_choice"] = "auto"

        content, raw_calls = self.backend.send_with_retries(
            lambda client, kw: _consume_stream(client.chat.completions.create(**kw)), kwargs,
        )

        assistant: Dict[str, Any] = {"role": "assistant", "content": content}
        if raw_calls:
            assistant["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in raw_calls
            ]
        self.messages.append(assistant)

        tool_calls = []
        for c in raw_calls:
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=c["id"], name=c["name"], args=args))
        return LLMReply(text=content, tool_calls=tool_calls)

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: List[Tuple[ToolCall, str]], notes: List[str]) -> None:
        # OpenAI: one "tool" message per call, then any notes as user messages.
        for call, text in results:
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        for note in notes:
            self.add_user_text(note)


def _consume_stream(stream) -> Tuple[str, List[Dict[str, str]]]:
    """Accumulate a streamed chat completion into (text, [{id, name, arguments}])."""
    content_parts: List[str] = []
    tool_call_parts: Dict[int, Dict[str, str]] = {}

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content_parts.append(delta.content)
        if delta.tool_calls:
            for tc in delta.tool_calls:
                slot = tool_call_parts.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

    return "".join(content_parts), [slot for _, slot in sorted(tool_call_parts.items())]
