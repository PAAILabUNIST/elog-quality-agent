"""Claude via the Anthropic Messages API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic

try:
    # anthropic>=1.x vendors its own httpx fork ("httpx2") and requires
    # http_client to be an instance of httpx2.Client specifically - a plain
    # httpx.Client raises TypeError. Older anthropic versions (<1.0) still
    # take a plain httpx.Client, so fall back to that if httpx2 isn't
    # installed (it's pulled in automatically as a dependency of the newer
    # anthropic SDK, not something to install separately).
    import httpx2 as httpx
except ImportError:
    import httpx

from .base import Conversation, LLMBackend, LLMReply, ToolCall

# max_tokens is REQUIRED by the Anthropic API; tool-use turns can carry a
# full generated script inside the tool_use input, which needs headroom.
DEFAULT_MAX_TOKENS = 8192


class AnthropicBackend(LLMBackend):
    tag = "claude"
    api_key_env = "ANTHROPIC_API_KEY"
    model_env = "ANTHROPIC_MODEL"
    # NOTE: verify the exact model string you have access to (this changes
    # over time) — check https://docs.claude.com for current model names.
    default_model = "claude-opus-4-8"

    def build_client(self, api_key: Optional[str] = None) -> Anthropic:
        """Keep-alive connection reuse disabled - same rationale as the
        OpenAI backend (the Anthropic SDK is also httpx-based)."""
        return Anthropic(
            api_key=api_key,
            timeout=httpx.Timeout(120.0, connect=30.0),
            max_retries=0,
            http_client=httpx.Client(
                timeout=httpx.Timeout(120.0, connect=30.0),
                limits=httpx.Limits(max_keepalive_connections=0),
            ),
        )

    def new_conversation(self, system_prompt: str, user_prompt: str) -> Conversation:
        return AnthropicConversation(self, system_prompt, user_prompt)

    @staticmethod
    def format_tools(tool_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Flat {name, description, input_schema}; passing `tools` is enough
        # for the model to choose freely (no tool_choice needed).
        return [
            {"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
            for s in tool_specs
        ]


class AnthropicConversation(Conversation):
    backend: AnthropicBackend

    def _start(self, user_prompt: str) -> None:
        # The system prompt is a separate top-level param, not a message.
        self.messages = [{"role": "user", "content": user_prompt}]

    def ask(self, tool_specs: List[Dict[str, Any]]) -> LLMReply:
        # .stream() rather than .create() for the same Cloudflare idle-timeout
        # reason as the OpenAI backend: it keeps bytes (including keepalive
        # pings during long generations) flowing. get_final_message() returns
        # the same Message shape .create() would.
        kwargs: Dict[str, Any] = {
            "model": self.backend.model,
            "messages": self.messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": self.system_prompt,
        }
        if tool_specs:
            kwargs["tools"] = self.backend.format_tools(tool_specs)

        def send(client, kw):
            with client.messages.stream(**kw) as stream:
                return stream.get_final_message()

        message = self.backend.send_with_retries(send, kwargs)
        self.messages.append({"role": "assistant", "content": _content_to_dicts(message.content)})

        text = " ".join(b.text for b in message.content if getattr(b, "type", None) == "text")
        tool_calls = [
            ToolCall(id=b.id, name=b.name, args=b.input or {})
            for b in message.content if getattr(b, "type", None) == "tool_use"
        ]
        return LLMReply(text=text, tool_calls=tool_calls)

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: List[Tuple[ToolCall, str]], notes: List[str]) -> None:
        # Anthropic requires every tool_use block from one assistant turn to
        # be answered by tool_result blocks batched into a SINGLE subsequent
        # user message; extra notes ride along as text blocks in that message.
        blocks: List[Dict[str, Any]] = [
            {"type": "tool_result", "tool_use_id": call.id, "content": text}
            for call, text in results
        ]
        blocks += [{"type": "text", "text": note} for note in notes]
        self.messages.append({"role": "user", "content": blocks})


def _content_to_dicts(content) -> List[Dict[str, Any]]:
    """Serialize response content blocks back into plain dicts so they can be
    replayed as the assistant turn on the next call. Only text and tool_use
    blocks are expected (extended thinking is not enabled); anything else is
    passed through by best-effort dict conversion rather than dropped."""
    blocks: List[Dict[str, Any]] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block_type == "tool_use":
            blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        else:
            blocks.append(block.model_dump())
    return blocks
