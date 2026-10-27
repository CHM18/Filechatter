"""Tests for the thin MCP proxy (no running server required)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import subprocess
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest
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

    def test_timeout_returns_error(self, monkeypatch, caplog):
        calls = []

        def fake_post(*_args, **_kwargs):
            calls.append(_kwargs)
            raise requests.exceptions.Timeout()

        monkeypatch.setattr(requests, "post", fake_post)
        result = rag_mcp_server._call("list_sources", {}, timeout=1)
        assert result["status"] == "error"
        assert "did not answer" in result["message"]
        assert len(calls) == 1
        assert "operation may still be running" in caplog.text
        assert "Not automatically replaying" in caplog.text

    def test_connection_failure_is_logged_without_arguments_and_next_call_recovers(self, monkeypatch, caplog):
        calls = []

        def fake_post(*_args, **_kwargs):
            calls.append(_kwargs)
            if len(calls) == 1:
                raise requests.exceptions.ConnectionError()
            return DummyResponse(200, {"result": {"answer": 42}})

        monkeypatch.setattr(requests, "post", fake_post)
        with caplog.at_level(logging.WARNING, logger="filechatter.mcp"):
            result = rag_mcp_server._call("search_documents", {"query": "private document text"})
        assert result["status"] == "error"
        assert len(calls) == 1
        assert "Proxy remains running" in caplog.text
        assert "private document text" not in caplog.text
        assert rag_mcp_server._call("list_sources", {}) == {"answer": 42}

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


class TestLifecycle:
    @pytest.fixture(autouse=True)
    def isolate_main(self, monkeypatch):
        monkeypatch.setattr(rag_mcp_server, "_configure_logging", lambda: None)
        monkeypatch.setattr(rag_mcp_server.faulthandler, "enable", lambda: None)
        monkeypatch.setattr(rag_mcp_server, "register_tools", lambda: ["list_sources"])

    def test_normal_transport_close_is_explained(self, monkeypatch, caplog):
        monkeypatch.setattr(rag_mcp_server.mcp, "run", lambda: None)
        previous = signal.getsignal(signal.SIGTERM)
        with caplog.at_level(logging.INFO, logger="filechatter.mcp"):
            assert rag_mcp_server.main() == 0
        assert "Starting MCP stdio proxy" in caplog.text
        assert "host closed stdin/EOF" in caplog.text
        assert "cannot be reattached" in caplog.text
        assert "MCP proxy stopping" in caplog.text
        assert signal.getsignal(signal.SIGTERM) == previous

    def test_exception_has_traceback_and_failure_exit_code(self, monkeypatch, caplog):
        def fail():
            raise RuntimeError("transport broke")

        monkeypatch.setattr(rag_mcp_server.mcp, "run", fail)
        previous = signal.getsignal(signal.SIGTERM)
        assert rag_mcp_server.main() == 1
        assert "RuntimeError: transport broke" in caplog.text
        assert any(record.exc_info for record in caplog.records)
        assert signal.getsignal(signal.SIGTERM) == previous

    def test_registration_failure_is_also_logged(self, monkeypatch, caplog):
        def fail():
            raise RuntimeError("registration broke")

        monkeypatch.setattr(rag_mcp_server, "register_tools", fail)
        assert rag_mcp_server.main() == 1
        assert "RuntimeError: registration broke" in caplog.text

    def test_keyboard_interrupt_is_distinct(self, monkeypatch, caplog):
        def interrupt():
            raise KeyboardInterrupt()

        monkeypatch.setattr(rag_mcp_server.mcp, "run", interrupt)
        with caplog.at_level(logging.INFO, logger="filechatter.mcp"):
            assert rag_mcp_server.main() == 130
        assert "Ctrl+C/SIGINT" in caplog.text
        assert "MCP proxy failed" not in caplog.text

    def test_sigterm_is_logged_and_exit_preserved(self, monkeypatch, caplog):
        def terminate():
            rag_mcp_server._handle_termination(signal.SIGTERM, None)

        monkeypatch.setattr(rag_mcp_server.mcp, "run", terminate)
        previous = signal.getsignal(signal.SIGTERM)
        with caplog.at_level(logging.INFO, logger="filechatter.mcp"):
            with pytest.raises(SystemExit) as exc:
                rag_mcp_server.main()
        assert exc.value.code == 128 + signal.SIGTERM
        assert "Received SIGTERM" in caplog.text
        assert "requested exit" in caplog.text
        assert signal.getsignal(signal.SIGTERM) == previous


class TestLogging:
    @pytest.mark.parametrize("file_logging", [True, False])
    def test_logging_leaves_stdout_clean_even_when_file_unavailable(self, tmp_path, file_logging):
        if not file_logging:
            (tmp_path / "logs").write_text("Blocks log directory creation", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import sys; import rag_mcp_server as proxy; "
                "proxy.__file__ = sys.argv[1]; raise SystemExit(proxy.main())",
                str(tmp_path / "rag_mcp_server.py"),
            ],
            cwd=Path(rag_mcp_server.__file__).resolve().parent,
            env=dict(os.environ, RAG_SERVER_URL="http://127.0.0.1:1"),
            input="", capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""
        assert "host closed stdin/EOF" in result.stderr
        if not file_logging:
            assert "logging to stderr only" in result.stderr
            return
        logs = list((tmp_path / "logs").glob("mcp-proxy-*.log"))
        assert len(logs) == 1
        text = logs[0].read_text(encoding="utf-8")
        for message in ("Starting MCP stdio proxy", "Tool visibility lookup failed",
                        f"Registered {len(rag_mcp_server.ALL_TOOLS)} tools",
                        "host closed stdin/EOF", "MCP proxy stopping"):
            assert message in text

    def test_real_mcp_session_survives_backend_failure(self, tmp_path):
        async def run_session():
            params = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-c",
                    "import sys; import rag_mcp_server as proxy; "
                    "proxy.__file__ = sys.argv[1]; raise SystemExit(proxy.main())",
                    str(tmp_path / "rag_mcp_server.py"),
                ],
                cwd=str(Path(rag_mcp_server.__file__).resolve().parent),
                env={"RAG_SERVER_URL": "http://127.0.0.1:1"},
            )
            with anyio.fail_after(30):
                with (tmp_path / "stderr.log").open("w", encoding="utf-8") as stderr:
                    async with stdio_client(params, errlog=stderr) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            assert {tool.name for tool in tools.tools} == set(rag_mcp_server.ALL_TOOLS)
                            result = await session.call_tool("list_sources", {})
                            assert "not reachable" in str(result)
                            # The same transport must still accept requests.
                            assert len((await session.list_tools()).tools) == len(tools.tools)

        anyio.run(run_session)
