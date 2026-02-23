"""
EcodiaOS — LLM Provider Abstraction

Every system that needs LLM reasoning uses this interface.
Supports Anthropic Claude, OpenAI, and local models via Ollama.

Includes retry with exponential backoff for transient errors (429, 503, 529)
and automatic fallback to a secondary provider when the primary is unavailable.
"""

from __future__ import annotations

import asyncio
import json as _json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from ecodiaos.config import LLMConfig

logger = structlog.get_logger()

# Retry configuration
_MAX_RETRIES = 3
_BASE_DELAY_S = 1.0
_RETRYABLE_STATUS_CODES = {429, 503, 529}


class Message:
    """A chat message."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMResponse:
    """Response from an LLM call."""

    def __init__(
        self,
        text: str,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        finish_reason: str = "stop",
    ) -> None:
        self.text = text
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.finish_reason = finish_reason

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ToolDefinition:
    """A tool that can be called by the LLM."""

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema object

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    tool_use_id: str
    content: str
    is_error: bool = False

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Format as Anthropic tool_result content block."""
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }


@dataclass
class ToolAwareResponse:
    """Response from a tool-aware LLM call."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # end_turn | tool_use | max_tokens
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """Abstract interface for LLM calls."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        """Full generation call."""
        ...

    @abstractmethod
    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Short evaluation call (lower temp, smaller output)."""
        ...

    @abstractmethod
    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],  # Raw message dicts (supports content blocks)
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ToolAwareResponse:
        """
        Tool-use generation call. Messages use raw dicts to support
        Anthropic's content block format (text blocks + tool_result blocks).
        Returns ToolAwareResponse with any tool calls the model wants to make.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...


class AnthropicProvider(LLMProvider):
    """Claude API provider with retry and exponential backoff."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self._model = model
        # Strip whitespace/newlines — GCP Secret Manager can inject trailing \r\n
        clean_key = api_key.strip()
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": clean_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=60.0,
        )

    async def _post_with_retry(
        self, path: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST with exponential backoff on retryable status codes."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.post(path, json=payload)
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY_S * (2 ** attempt)
                    # Respect Retry-After header if present
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    logger.warning(
                        "llm_retrying",
                        status=response.status_code,
                        attempt=attempt + 1,
                        delay_s=round(delay, 1),
                    )
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY_S * (2 ** attempt)
                    logger.warning(
                        "llm_timeout_retrying",
                        attempt=attempt + 1,
                        delay_s=round(delay, 1),
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.HTTPStatusError as exc:
                # Include API error details in the exception message
                body = ""
                try:
                    body = exc.response.text[:500]
                except Exception:
                    pass
                raise httpx.HTTPStatusError(
                    message=f"{exc.response.status_code}: {body}",
                    request=exc.request,
                    response=exc.response,
                ) from exc
        raise last_exc or RuntimeError("LLM request failed after retries")

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [m.to_dict() for m in messages],
        }

        data = await self._post_with_retry("/messages", payload)

        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        return LLMResponse(
            text=text,
            model=data.get("model", self._model),
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "stop"),
        )

    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        return await self.generate(
            system_prompt="You are an evaluator. Be precise and concise.",
            messages=[Message("user", prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ToolAwareResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": messages,
            "tools": [t.to_dict() for t in tools],
        }

        data = await self._post_with_retry("/messages", payload)

        # Parse content blocks — may include text and tool_use blocks
        text = ""
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    input=block.get("input", {}),
                ))

        return ToolAwareResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=data.get("stop_reason", "end_turn"),
            model=data.get("model", self._model),
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
        )

    async def close(self) -> None:
        await self._client.aclose()


class OllamaProvider(LLMProvider):
    """Local model via Ollama."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        endpoint: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=endpoint,
            timeout=120.0,
        )

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        all_messages = [{"role": "system", "content": system_prompt}]
        all_messages.extend(m.to_dict() for m in messages)

        payload = {
            "model": self._model,
            "messages": all_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=self._model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        return await self.generate(
            system_prompt="You are an evaluator. Be precise and concise.",
            messages=[Message("user", prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ToolAwareResponse:
        # Ollama tool use: flatten messages to text, ask for JSON tool calls.
        # This is a best-effort implementation for local model fallback.
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in tools
        )
        user_content = (
            f"Available tools:\n{tool_descriptions}\n\n"
            "To use a tool, respond with JSON: "
        )
        user_content += "{\\\"tool\\\": \\\"<name>\\\", \\\"input\\\": {...}}\n\n"
        # Extract last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_content += content
                break

        plain_messages = [Message("user", user_content)]
        response = await self.generate(
            system_prompt=system_prompt,
            messages=plain_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Try to parse a tool call from the response
        tool_calls: list[ToolCall] = []
        try:
            parsed = _json.loads(response.text.strip())
            if "tool" in parsed:
                tool_calls.append(ToolCall(
                    id="ollama_" + str(parsed["tool"]),
                    name=parsed["tool"],
                    input=parsed.get("input", {}),
                ))
        except (_json.JSONDecodeError, KeyError):
            pass

        return ToolAwareResponse(
            text=response.text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=self._model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    async def close(self) -> None:
        await self._client.aclose()


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4o, GPT-4o-mini, o1, etc.)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._model = model
        clean_key = api_key.strip()
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {clean_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def _post_with_retry(
        self, path: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST with exponential backoff on retryable status codes."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.post(path, json=payload)
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY_S * (2 ** attempt)
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    logger.warning(
                        "llm_retrying",
                        status=response.status_code,
                        attempt=attempt + 1,
                        delay_s=round(delay, 1),
                    )
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY_S * (2 ** attempt)
                    logger.warning(
                        "llm_timeout_retrying",
                        attempt=attempt + 1,
                        delay_s=round(delay, 1),
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.HTTPStatusError as exc:
                body = ""
                try:
                    body = exc.response.text[:500]
                except Exception:
                    pass
                raise httpx.HTTPStatusError(
                    message=f"{exc.response.status_code}: {body}",
                    request=exc.request,
                    response=exc.response,
                ) from exc
        raise last_exc or RuntimeError("LLM request failed after retries")

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        output_format: str | None = None,
    ) -> LLMResponse:
        # OpenAI uses system message inside the messages array
        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            content = m.content if m.content else " "  # OpenAI rejects empty content
            all_messages.append({"role": m.role, "content": content})

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": all_messages,
        }

        data = await self._post_with_retry("/chat/completions", payload)

        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage = data.get("usage", {})

        return LLMResponse(
            text=text or "",
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choices[0].get("finish_reason", "stop") if choices else "stop",
        )

    async def evaluate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        return await self.generate(
            system_prompt="You are an evaluator. Be precise and concise.",
            messages=[Message("user", prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> ToolAwareResponse:
        # Convert Anthropic-style messages to OpenAI format
        all_messages: list[dict[str, Any]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Anthropic uses content blocks (list of dicts); OpenAI uses strings
            if isinstance(content, list):
                # Flatten content blocks to text
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            parts.append(f"[Tool result: {block.get('content', '')}]")
                    elif isinstance(block, str):
                        parts.append(block)
                content = "\n".join(parts) or " "
            all_messages.append({"role": role, "content": content or " "})

        # Convert tool definitions to OpenAI format
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": all_messages,
            "tools": openai_tools,
        }

        data = await self._post_with_retry("/chat/completions", payload)

        choices = data.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})
        text = message.get("content", "") or ""
        usage = data.get("usage", {})

        # Parse tool calls from OpenAI format
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls", []):
            fn = tc.get("function", {})
            try:
                args = _json.loads(fn.get("arguments", "{}"))
            except _json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                input=args,
            ))

        stop_reason = choice.get("finish_reason", "stop")
        # Map OpenAI stop reasons to our convention
        if stop_reason == "tool_calls":
            stop_reason = "tool_use"
        elif stop_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        return ToolAwareResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def close(self) -> None:
        await self._client.aclose()


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Factory to create the configured LLM provider."""
    if config.provider == "anthropic":
        return AnthropicProvider(api_key=config.api_key, model=config.model)
    elif config.provider == "openai":
        return OpenAIProvider(api_key=config.api_key, model=config.model)
    elif config.provider == "ollama":
        return OllamaProvider(model=config.model)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")
