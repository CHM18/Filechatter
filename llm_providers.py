"""LLM provider abstraction for the built-in chat.

Browser chat targets local, OpenAI-compatible endpoints (LM Studio today,
Ollama later, or any server exposing /v1/chat/completions). Claude Desktop and
Microsoft Copilot are not chat targets here - they reach Filechatter as MCP
hosts and are configured elsewhere.

A provider exposes:
- list_models()                -> [model_id, ...]
- stream(messages, tools)      -> generator of events:
      ("token", text_delta)    incremental assistant text
      ("message", message)     the final assembled assistant message
                               ({"role","content","tool_calls"})

The streaming shape lets the agent forward tokens to the browser as they
arrive while still getting a clean final message (with any tool calls
assembled from their streamed fragments).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterator
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when the LLM endpoint is unreachable or returns an error."""


def _normalize_base_url(base_url: str | None) -> str:
    """Normalize endpoint URLs from UI/user settings.

    Accepts bare host:port input (e.g. localhost:1234) and upgrades it to
    http://localhost:1234 so requests can resolve it.
    """
    value = (base_url or "").strip()
    if not value:
        return ""

    if "://" not in value:
        value = f"http://{value.lstrip('/')}"
    parsed = urlparse(value)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return value.rstrip("/")


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        timeout: int = 120,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def _require_base_url(self) -> str:
        if not self.base_url:
            raise ProviderError(
                "LLM base URL is empty or invalid. Use a full URL like "
                "http://localhost:1234"
            )
        return self.base_url

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _stream_timeout(self) -> tuple[int, int]:
        # Streamed generation can take longer before first token on local models.
        # Keep connect timeout short, but enforce a generous minimum read timeout.
        return (10, max(int(self.timeout), 300))

    def list_models(self) -> list[str]:
        base_url = self._require_base_url()
        try:
            response = requests.get(
                f"{base_url}/v1/models", headers=self._headers(), timeout=10
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"Could not reach {base_url}: {exc}") from exc
        payload = response.json()
        data = payload.get("data") or payload.get("models") or []
        models = []
        for item in data:
            if isinstance(item, dict):
                models.append(item.get("id") or item.get("model") or item.get("name"))
            else:
                models.append(str(item))
        return [m for m in models if m]

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> Iterator[tuple[str, Any]]:
        base_url = self._require_base_url()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self._stream_timeout(),
                stream=True,
            )
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"Could not reach the LLM at {base_url}: {exc}") from exc

        if response.status_code != 200:
            detail = response.text[:500]
            raise ProviderError(f"LLM returned HTTP {response.status_code}: {detail}")

        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}

        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage:
                    yield ("usage", usage)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                text = delta.get("content")
                if text:
                    content_parts.append(text)
                    yield ("token", text)

                for tool_delta in delta.get("tool_calls") or []:
                    index = tool_delta.get("index", 0)
                    accumulator = tool_calls.setdefault(
                        index, {"id": None, "name": "", "arguments": ""}
                    )
                    if tool_delta.get("id"):
                        accumulator["id"] = tool_delta["id"]
                    function = tool_delta.get("function") or {}
                    if function.get("name"):
                        accumulator["name"] = function["name"]
                    if function.get("arguments"):
                        accumulator["arguments"] += function["arguments"]
        except requests.exceptions.ReadTimeout as exc:
            _, read_timeout = self._stream_timeout()
            raise ProviderError(
                "LLM response timed out while waiting for streamed tokens "
                f"(read-timeout={read_timeout}s). Increase llm.timeout_seconds in settings."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ProviderError(f"Could not read streamed response from {base_url}: {exc}") from exc

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": call["id"] or f"call_{index}",
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"] or "{}"},
                }
                for index, call in sorted(tool_calls.items())
            ]
        yield ("message", message)


def build_provider(llm_settings: dict[str, Any]) -> OpenAICompatProvider:
    """Construct a provider from the persisted `llm` settings block.

    Only OpenAI-compatible providers are supported for browser chat. The
    'provider' label (lmstudio/ollama/openai_compat/...) is informational; they
    all speak the same wire protocol.
    """
    provider_name = str(llm_settings.get("provider", "lmstudio") or "lmstudio").lower()
    base_url = llm_settings.get("base_url")
    if provider_name == "lmstudio" and not str(base_url or "").strip():
        base_url = "http://localhost:1234"
    elif provider_name == "ollama" and not str(base_url or "").strip():
        base_url = "http://localhost:11434"

    model = str(llm_settings.get("model") or "").strip() or "local-model"

    return OpenAICompatProvider(
        base_url=base_url if base_url is not None else "http://localhost:1234",
        model=model,
        api_key=llm_settings.get("api_key") or None,
        temperature=float(llm_settings.get("temperature", 0.7)),
        timeout=int(llm_settings.get("timeout_seconds", 120)),
    )
