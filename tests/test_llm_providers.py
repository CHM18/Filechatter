"""Tests for the OpenAI-compatible provider (streaming parse, models, errors)."""
from __future__ import annotations

import json

import pytest
import requests

import llm_providers
from llm_providers import OpenAICompatProvider, ProviderError


class FakeStreamResponse:
    def __init__(self, status_code, lines):
        self.status_code = status_code
        self._lines = lines
        self.text = "\n".join(
            line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line)
            for line in lines
        )

    def iter_lines(self, decode_unicode=True):
        yield from self._lines


class TimeoutStreamResponse:
    status_code = 200
    text = ""

    def iter_lines(self, decode_unicode=True):
        raise requests.exceptions.ReadTimeout("timed out")


def sse(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False)


def delta(**d):
    return {"choices": [{"delta": d}]}


class TestStreaming:
    def test_text_streaming(self, monkeypatch):
        lines = [
            sse(delta(content="Hel")),
            sse(delta(content="lo")),
            "data: [DONE]",
        ]
        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeStreamResponse(200, lines))
        provider = OpenAICompatProvider("http://x", "m")
        events = list(provider.stream([{"role": "user", "content": "hi"}]))
        tokens = [v for kind, v in events if kind == "token"]
        message = next(v for kind, v in events if kind == "message")
        assert tokens == ["Hel", "lo"]
        assert message["content"] == "Hello"
        assert "tool_calls" not in message

    def test_tool_call_assembly_from_fragments(self, monkeypatch):
        lines = [
            sse(delta(tool_calls=[{"index": 0, "id": "call_1", "function": {"name": "search_documents", "arguments": '{"que'}}])),
            sse(delta(tool_calls=[{"index": 0, "function": {"arguments": 'ry":"x"}'}}])),
            "data: [DONE]",
        ]
        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeStreamResponse(200, lines))
        provider = OpenAICompatProvider("http://x", "m")
        message = next(v for kind, v in provider.stream([], tools=[{"x": 1}]) if kind == "message")
        assert message["tool_calls"][0]["function"]["name"] == "search_documents"
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"query": "x"}

    def test_http_error_raises(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeStreamResponse(500, ["boom"]))
        provider = OpenAICompatProvider("http://x", "m")
        with pytest.raises(ProviderError):
            list(provider.stream([]))

    def test_connection_error_raises(self, monkeypatch):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError()

        monkeypatch.setattr(requests, "post", boom)
        with pytest.raises(ProviderError):
            list(OpenAICompatProvider("http://x", "m").stream([]))

    def test_read_timeout_during_streaming_raises_provider_error(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: TimeoutStreamResponse())
        provider = OpenAICompatProvider("http://x", "m", timeout=7)
        with pytest.raises(ProviderError, match="timed out"):
            list(provider.stream([{"role": "user", "content": "hi"}]))

    def test_streaming_decodes_utf8_unicode_from_bytes(self, monkeypatch):
        lines = [
            sse(delta(content="Grüße aus Köln: déjà vu.")).encode("utf-8"),
            b"data: [DONE]",
        ]
        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeStreamResponse(200, lines))
        provider = OpenAICompatProvider("http://x", "m")
        events = list(provider.stream([{"role": "user", "content": "hi"}]))
        message = next(v for kind, v in events if kind == "message")
        assert message["content"] == "Grüße aus Köln: déjà vu."


class TestListModels:
    def test_parses_openai_shape(self, monkeypatch):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"id": "qwen"}, {"id": "llama"}]}

        monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
        assert OpenAICompatProvider("http://x", "m").list_models() == ["qwen", "llama"]

    def test_connection_error_raises(self, monkeypatch):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError()

        monkeypatch.setattr(requests, "get", boom)
        with pytest.raises(ProviderError):
            OpenAICompatProvider("http://x", "m").list_models()


class TestBuildProvider:
    def test_from_settings(self):
        provider = llm_providers.build_provider({
            "base_url": "http://localhost:1234/",
            "model": "qwen",
            "api_key": "sk",
            "temperature": 0.3,
            "timeout_seconds": 45,
        })
        assert provider.base_url == "http://localhost:1234"  # trailing slash stripped
        assert provider.model == "qwen"
        assert provider.api_key == "sk"
        assert provider.temperature == 0.3
        assert provider.timeout == 45

    def test_adds_http_scheme_for_bare_host(self):
        provider = llm_providers.build_provider({
            "provider": "lmstudio",
            "base_url": "localhost:1234",
            "model": "qwen",
        })
        assert provider.base_url == "http://localhost:1234"

    def test_provider_defaults_when_base_url_empty(self):
        lmstudio = llm_providers.build_provider({
            "provider": "lmstudio",
            "base_url": "",
            "model": "qwen",
        })
        ollama = llm_providers.build_provider({
            "provider": "ollama",
            "base_url": "",
            "model": "qwen",
        })
        assert lmstudio.base_url == "http://localhost:1234"
        assert ollama.base_url == "http://localhost:11434"

    def test_blank_model_falls_back_to_local_model(self):
        provider = llm_providers.build_provider({
            "provider": "lmstudio",
            "base_url": "http://localhost:1234",
            "model": "",
        })
        assert provider.model == "local-model"
