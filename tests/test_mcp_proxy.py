"""Tests for the thin MCP proxy (no running server required)."""
from __future__ import annotations

import requests

import rag_mcp_server


class DummyResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class TestCall:
    def test_forwards_result(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return DummyResponse(200, {"status": "ok", "result": {"answer": 42}})

        monkeypatch.setattr(requests, "post", fake_post)
        result = rag_mcp_server._call("search_documents", {"query": "x"})
        assert result == {"answer": 42}
        assert captured["url"].endswith("/tools/search_documents")
        assert captured["json"]["arguments"] == {"query": "x"}
        assert captured["json"]["client"] == "mcp"

    def test_connection_error_returns_helpful_message(self, monkeypatch):
        def fake_post(*_args, **_kwargs):
            raise requests.exceptions.ConnectionError()

        monkeypatch.setattr(requests, "post", fake_post)
        result = rag_mcp_server._call("list_sources", {})
        assert result["status"] == "error"
        assert "not reachable" in result["message"]

    def test_timeout_returns_error(self, monkeypatch):
        def fake_post(*_args, **_kwargs):
            raise requests.exceptions.Timeout()

        monkeypatch.setattr(requests, "post", fake_post)
        result = rag_mcp_server._call("list_sources", {}, timeout=1)
        assert result["status"] == "error"
        assert "did not answer" in result["message"]

    def test_http_error_passes_detail_through(self, monkeypatch):
        def fake_post(*_args, **_kwargs):
            return DummyResponse(400, {"detail": "Collection 'x' does not accept '.txt' files"})

        monkeypatch.setattr(requests, "post", fake_post)
        result = rag_mcp_server._call("ingest_file", {"path": "x.txt"})
        assert result["status"] == "error"
        assert "does not accept" in result["message"]


class TestToolWrappers:
    def test_ingest_file_uses_long_timeout(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["timeout"] = timeout
            return DummyResponse(200, {"result": {}})

        monkeypatch.setattr(requests, "post", fake_post)
        rag_mcp_server.ingest_file(path="big.pdf")
        assert captured["timeout"] == rag_mcp_server.INGEST_FILE_TIMEOUT

    def test_call_sends_mcp_client_flag(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["json"] = json
            return DummyResponse(200, {"result": {}})

        monkeypatch.setattr(requests, "post", fake_post)
        rag_mcp_server._call("search_documents", {"query": "x"})
        assert captured["json"]["client"] == "mcp"


class TestRegisterTools:
    def test_registers_only_visible_tools(self, monkeypatch):
        class Resp:
            status_code = 200

            def json(self):
                return {"tools": [{"name": "search_documents"}, {"name": "list_sources"}]}

        monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
        registered = []
        monkeypatch.setattr(rag_mcp_server.mcp, "add_tool", lambda fn, *a, **k: registered.append(fn.__name__))

        names = rag_mcp_server.register_tools()
        assert set(names) == {"search_documents", "list_sources"}
        assert "clear_index" not in names

    def test_registers_all_when_server_unreachable(self, monkeypatch):
        def boom(*_a, **_k):
            raise requests.exceptions.ConnectionError()

        monkeypatch.setattr(requests, "get", boom)
        monkeypatch.setattr(rag_mcp_server.mcp, "add_tool", lambda fn, *a, **k: None)
        names = rag_mcp_server.register_tools()
        # Fail open on visibility so tools are not silently missing; the server
        # still enforces at call time.
        assert set(names) == set(rag_mcp_server.ALL_TOOLS)

    def test_proxy_does_not_import_heavy_modules(self):
        import inspect

        # The proxy must stay lightweight: no store, no loader, no embeddings.
        # (Other tests import these modules, so check the proxy's source, not
        # sys.modules.)
        source = inspect.getsource(rag_mcp_server)
        for forbidden in ("rag_store", "document_loader", "sentence_transformers", "faiss"):
            assert forbidden not in source, f"proxy must not import {forbidden}"
