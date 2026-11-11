"""API tests for the FastAPI server (collections CRUD, tools endpoint, settings)."""
from __future__ import annotations

import json
import os
import threading

import pytest


class TestHealth:
    def test_health_reports_collections(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert "collections_count" in payload
        assert "chunks_count" in payload

    def test_health_reuses_the_model_list_cache(self, client, monkeypatch):
        import rag_server

        calls = []

        class Provider:
            base_url = "http://localhost:1234"
            api_key = None
            model = "test-model"

            def list_models(self):
                calls.append("models")
                return ["test-model"]

        monkeypatch.setattr("rag_server.build_provider", lambda _settings: Provider())
        rag_server._MODEL_CACHE.clear()

        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200
        assert calls == ["models"]
        rag_server._MODEL_CACHE.clear()


class TestFileTypesApi:
    def test_file_types_endpoint(self, client):
        categories = client.get("/file-types").json()["categories"]
        names = {c["category"] for c in categories}
        assert {"Office", "Text", "Web", "Code"}.issubset(names)
        office = next(c for c in categories if c["category"] == "Office")
        exts = {e["ext"] for e in office["extensions"]}
        assert {".pdf", ".xlsx", ".pptx"}.issubset(exts)


class TestChunkInspection:
    def test_dump_all_includes_persisted_image_data_and_document_records(self, client, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        import cli

        source = "C:/photos/holiday.jpg"
        image_metadata = {
            "source": source,
            "path": source,
            "filename": "holiday.jpg",
            "image_description": "Grüner See",
            "image_summary": "A lake and mountains.",
            "image_keywords": ["lake", "mountains"],
            "image_exif": {"Model": "Camera X"},
            "image_exif_keywords": ["Camera X"],
            "file_changed_epoch": 1700000000,
        }
        response = client.post(
            "/upload",
            json={
                "documents": ["Image analysis chunk."],
                "metadata": [image_metadata],
                "collection_name": "photos",
            },
        )
        assert response.status_code == 200
        monkeypatch.setattr(
            cli.requests, "get", lambda url, params, timeout: client.get("/documents/inspect", params=params)
        )

        result = CliRunner().invoke(cli.app, ["dump-all", "HOLIDAY.JPG"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data["documents"]) == 1
        document = data["documents"][0]
        assert document["source"] == source
        assert document["source_name"] == "holiday.jpg"
        assert document["file_changed_at"] == 1700000000
        assert document["last_ingested_at"]
        assert document["collection_name"] == "photos"
        assert len(data["chunks"]) == 1
        chunk = data["chunks"][0]
        assert chunk["content"] == "Image analysis chunk."
        for key in ("image_description", "image_summary", "image_keywords", "image_exif", "image_exif_keywords"):
            assert chunk["metadata"][key] == image_metadata[key]

        output = tmp_path / "dump.json"
        result = CliRunner().invoke(cli.app, ["dump-all", source, "--output", str(output)])
        assert result.exit_code == 0, result.output
        assert result.output == ""
        assert json.loads(output.read_text(encoding="utf-8"))["chunks"] == data["chunks"]
        assert "Grüner See" in output.read_text(encoding="utf-8")

    def test_document_inspection_pages_all_collections(self, client, monkeypatch):
        import rag_server

        class Store:
            def __init__(self, collection):
                self.collection_name = collection

            def get_document_records(self, source):
                return [{"source": source, "collection_name": self.collection_name}]

            def get_chunks(self, *, source, limit, offset):
                return [
                    {"id": i, "source": source, "collection_name": self.collection_name, "chunk_index": i}
                    for i in range(offset, min(offset + limit, 1001))
                ]

        manager = rag_server.runtime.collections()
        monkeypatch.setattr(manager, "selected_stores", lambda name: [Store("a"), Store("b")])

        response = client.get("/documents/inspect", params={"source": "image.jpg"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 2
        assert len(data["chunks"]) == 2002
        assert [chunk["chunk_index"] for chunk in data["chunks"] if chunk["collection_name"] == "b"] == list(range(1001))

    def test_document_inspection_filename_and_path_filter(self, client):
        for source in ("C:/one/photo.jpg", "C:/two/photo.jpg"):
            response = client.post("/upload", json={
                "documents": [source],
                "metadata": [{"source": source, "filename": "photo.jpg"}],
                "collection_name": "photos",
            })
            assert response.status_code == 200

        result = client.get("/documents/inspect", params={"source": "PHOTO.JPG"}).json()
        assert {item["source"] for item in result["documents"]} == {"C:/one/photo.jpg", "C:/two/photo.jpg"}
        assert {item["source"] for item in result["chunks"]} == {"C:/one/photo.jpg", "C:/two/photo.jpg"}
        result = client.get("/documents/inspect", params={"source": "C:/one/photo.jpg"}).json()
        assert [item["source"] for item in result["documents"]] == ["C:/one/photo.jpg"]
        assert [item["source"] for item in result["chunks"]] == ["C:/one/photo.jpg"]
        assert client.get("/documents/inspect", params={"source": " "}).status_code == 400
        assert client.get("/documents/inspect", params={"source": "missing.jpg"}).json()["chunks"] == []

    def test_dump_all_reports_failures(self, client, tmp_path, monkeypatch):
        from types import SimpleNamespace
        from typer.testing import CliRunner

        import cli

        def disconnected(*args, **kwargs):
            raise cli.requests.exceptions.ConnectionError()

        monkeypatch.setattr(cli.requests, "get", disconnected)
        result = CliRunner().invoke(cli.app, ["dump-all", "photo.jpg"])
        assert result.exit_code == 1
        assert "Cannot connect to server" in result.output

        def unavailable(*args, **kwargs):
            def raise_error():
                raise cli.requests.exceptions.HTTPError("404 Client Error")
            return SimpleNamespace(raise_for_status=raise_error)

        monkeypatch.setattr(cli.requests, "get", unavailable)
        result = CliRunner().invoke(cli.app, ["dump-all", "photo.jpg"])
        assert result.exit_code == 1
        assert "404 Client Error" in result.output

        monkeypatch.setattr(
            cli.requests, "get", lambda url, params, timeout: client.get("/documents/inspect", params=params)
        )
        result = CliRunner().invoke(cli.app, ["dump-all", "missing.jpg"])
        assert result.exit_code == 0
        assert json.loads(result.output)["documents"] == []
        assert json.loads(result.output)["chunks"] == []
        result = CliRunner().invoke(cli.app, ["dump-all", "photo.jpg", "-o", str(tmp_path)])
        assert result.exit_code == 1
        assert "Cannot write output file" in result.output
    def test_debug_search_contains_normal_response_and_candidates(self, client):
        response = client.post("/upload", json={
            "documents": ["Ostern 2012 Schnee", "Ostern 2023 Sonne"],
            "metadata": [{"source": "2012.jpg"}, {"source": "2023.jpg"}],
            "collection_name": "photos",
        })
        assert response.status_code == 200
        query = {"question": "ostern 2012 schnee", "collection_name": "photos"}
        regular = client.post("/search", json=query)
        debug = client.post("/search/debug", json=query)
        assert debug.status_code == 200
        data = debug.json()
        assert data["results"] == regular.json()["results"]
        assert len(data["collections"]) == 1
        assert data["collections"][0]["collection_name"] == "photos"
        candidates = data["collections"][0]["candidates"]
        assert candidates
        for result in data["results"]:
            candidate = next(item for item in candidates if item["chunk_id"] == result["chunk_id"])
            assert candidate["combined_rank"] == result["rank"]
            assert candidate["combined_score"] == result["score"]
            assert candidate["selection"].startswith("selected")
        assert client.post("/search/debug", json={**query, "include_context": False}).status_code == 400
        assert client.post("/search/debug", json={**query, "question": " "}).status_code == 400
        assert client.post("/search/debug", json={**query, "collection_name": "missing"}).status_code == 404
        client.patch("/collections/photos", json={"read_allowed": False})
        assert client.post("/search/debug", json=query).status_code == 403

    def test_filename_lookup_and_exact_path_lookup(self, client):
        source = r"C:\docs\Snke Operating system 2-3 - Software User Guide.pdf"
        other_source = r"C:\other\Snke Operating system 2-3 - Software User Guide.pdf"
        filename = "Snke Operating system 2-3 - Software User Guide.pdf"
        response = client.post(
            "/upload",
            json={
                "documents": ["First guide content.", "Second guide content."],
                "metadata": [
                    {"source": source, "filename": filename},
                    {"source": other_source, "filename": filename},
                ],
                "collection_name": "guides",
            },
        )
        assert response.status_code == 200

        chunks = client.get("/chunks", params={"source": filename.lower()}).json()["chunks"]
        assert {chunk["source"] for chunk in chunks} == {source, other_source}

        chunks = client.get("/chunks", params={"source": source}).json()["chunks"]
        assert [chunk["source"] for chunk in chunks] == [source]

    def test_cli_prints_chunks_for_filename(self, client, monkeypatch):
        from typer.testing import CliRunner

        import cli

        filename = "Snke Operating system 2-3 - Software User Guide.pdf"
        client.post(
            "/upload",
            json={
                "documents": ["First stored chunk.", "Second stored chunk."],
                "metadata": [
                    {"source": f"C:/docs/{filename}", "filename": filename},
                    {"source": f"C:/docs/{filename}", "filename": filename},
                ],
                "collection_name": "guides",
            },
        )
        monkeypatch.setattr(
            cli.requests, "get", lambda url, params, timeout: client.get("/chunks", params=params)
        )

        result = CliRunner().invoke(cli.app, ["dump-chunks", filename, "--limit", "3"])

        assert result.exit_code == 0
        assert result.output.strip() == (
            "First stored chunk.\n-----------------------------\nSecond stored chunk."
        )
        assert "chunk 0" not in result.output
        assert "╭" not in result.output
        assert "Error:" not in result.output

    @pytest.mark.parametrize("arguments, expected_count", [([], 1000), (["--limit", "7"], 7)])
    def test_cli_dump_limit_through_api(self, client, tmp_path, monkeypatch, arguments, expected_count):
        from types import SimpleNamespace
        from typer.testing import CliRunner

        import cli
        import rag_server

        chunks = [
            {
                "id": index + 1,
                "source": "example.pdf",
                "content": f"Stored chunk {index}.",
                "chunk_index": index,
                "collection_name": "guides",
            }
            for index in range(1005)
        ]
        store = SimpleNamespace(get_chunks=lambda *, source, limit, offset: chunks[offset : offset + limit])
        manager = rag_server.runtime.collections()
        monkeypatch.setattr(manager, "selected_stores", lambda name: [store])
        monkeypatch.setattr(
            cli.requests, "get", lambda url, params, timeout: client.get("/chunks", params=params)
        )
        output = tmp_path / "chunks.txt"

        result = CliRunner().invoke(
            cli.app, ["dump-chunks", "example.pdf", "-o", str(output), *arguments]
        )

        assert result.exit_code == 0, result.output
        text = output.read_text(encoding="utf-8")
        assert text.count("Stored chunk ") == expected_count
        assert f"Stored chunk {expected_count - 1}." in text
        assert f"Stored chunk {expected_count}." not in text

    @pytest.mark.parametrize("option", ["-o", "--output"])
    @pytest.mark.parametrize("fallback", [False, True])
    def test_cli_writes_utf8_file(self, tmp_path, monkeypatch, option, fallback):
        from types import SimpleNamespace
        from typer.testing import CliRunner

        import cli

        contents = ["Direct \u2013 calculation error\n" + "Long line " * 40, "\u00e4 \u2022 \u25b3 [x]"]
        chunks = [{"content": content} for content in contents]
        closed = []

        def get_chunks(*args, **kwargs):
            if fallback:
                raise cli.requests.exceptions.ConnectionError()
            return SimpleNamespace(
                status_code=200,
                raise_for_status=lambda: None,
                json=lambda: {"chunks": chunks},
            )

        monkeypatch.setattr(cli.requests, "get", get_chunks)
        monkeypatch.setattr(cli, "RagStore", lambda: SimpleNamespace(
            get_chunks=lambda **kwargs: chunks,
            close=lambda: closed.append(True),
        ))
        output = tmp_path / "chunk dump.txt"
        output.write_text("Old contents", encoding="utf-8")

        result = CliRunner().invoke(cli.app, ["dump-chunks", "example.docx", option, str(output)])

        assert result.exit_code == 0, result.output
        expected = "\n-----------------------------\n".join(contents) + "\n"
        assert output.read_bytes() == expected.replace("\n", os.linesep).encode("utf-8")
        assert "calculation error" not in result.output
        assert closed == ([True] if fallback else [])

    def test_cli_reports_output_write_failure(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        from typer.testing import CliRunner

        import cli

        monkeypatch.setattr(cli.requests, "get", lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"chunks": [{"content": "Stored text"}]},
        ))
        result = CliRunner().invoke(cli.app, ["dump-chunks", "example.docx", "-o", str(tmp_path)])

        assert result.exit_code == 1
        assert "Cannot write output file" in result.output

    def test_cli_empty_result_is_success(self, client, monkeypatch):
        from typer.testing import CliRunner

        import cli

        monkeypatch.setattr(
            cli.requests, "get", lambda url, params, timeout: client.get("/chunks", params=params)
        )

        result = CliRunner().invoke(cli.app, ["dump-chunks", "missing.pdf"])

        assert result.exit_code == 0
        assert "No chunks found" in result.output
        assert "Error:" not in result.output

    def test_cli_synchronizes_windows_console_before_using_utf8(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        import cli

        calls = []
        stream = SimpleNamespace(
            isatty=lambda: True,
            reconfigure=lambda **kwargs: calls.append(kwargs),
        )
        fake_ctypes = SimpleNamespace(
            windll=SimpleNamespace(
                kernel32=SimpleNamespace(SetConsoleOutputCP=lambda code: calls.append(code))
            )
        )
        with monkeypatch.context() as context:
            context.setattr(sys, "platform", "win32")
            context.setattr(sys, "stdout", stream)
            context.setattr(sys, "stderr", stream)
            context.setitem(sys.modules, "ctypes", fake_ctypes)
            cli.configure_output()

        assert calls == [
            65001,
            {"encoding": "utf-8", "errors": "backslashreplace"},
            {"encoding": "utf-8", "errors": "backslashreplace"},
        ]

    def test_cli_preserves_unicode_with_legacy_output_encoding(self):
        import subprocess
        import sys
        from pathlib import Path

        import cli

        script = r'''
import sys
from types import SimpleNamespace
import cli

sys.stdout.reconfigure(encoding="cp1252", errors="strict")
sys.stderr.reconfigure(encoding="cp1252", errors="strict")
text = "\u2022 \u25b3! Warning: Warnings are indicated by triangular warning symbols."
chunk = {
    "source": "warning.pdf",
    "chunk_index": 0,
    "start_char": 0,
    "end_char": len(text),
    "content": text,
}
cli.requests.get = lambda *args, **kwargs: SimpleNamespace(
    status_code=200,
    raise_for_status=lambda: None,
    json=lambda: {"chunks": [chunk]},
)
cli.app(["dump-chunks", "warning.pdf", "--limit", "3"])
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(cli.__file__).parent,
            capture_output=True,
            encoding="utf-8",
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert "\u2022 \u25b3! Warning:" in result.stdout
        assert "triangular warning symbols." in result.stdout
        assert "Error:" not in result.stdout


class TestChunkExportApi:
    def test_export_downloads_all_source_chunks_as_markdown(self, client, monkeypatch):
        import rag_server

        source = "C:/docs/annual-report.pdf"
        chunks = [
            {
                "id": index + 1,
                "source": source,
                "content": f"Review text for chunk {index}.",
                "chunk_index": index,
                "start_char": index * 100,
                "end_char": (index + 1) * 100,
                "collection_name": "reports",
            }
            for index in range(205)
        ]

        class Store:
            def get_chunks(self, *, source, limit, offset):
                assert source == "C:/docs/annual-report.pdf"
                return chunks[offset : offset + limit]

        class Manager:
            def normalize_name(self, name):
                return name

            def selected_stores(self, _name):
                return [Store()]

            def close_all(self):
                pass

        monkeypatch.setattr(rag_server.runtime, "collections", lambda: Manager())

        response = client.get(
            "/chunks/export.md",
            params={"source": source, "collection_name": "reports"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert response.headers["content-disposition"] == 'attachment; filename="annual-report-chunks.md"'
        assert "# Chunks: annual-report.pdf" in response.text
        assert "- Chunk count: 205" in response.text
        assert "## Chunk 204" in response.text
        assert "Review text for chunk 204." in response.text

    def test_export_requires_source(self, client):
        response = client.get("/chunks/export.md")

        assert response.status_code == 400


class TestCollectionsApi:
    def test_create_and_list(self, client):
        response = client.post(
            "/collections",
            json={"name": "api-docs", "description": "API documentation", "allowed_extensions": ["pdf"]},
        )
        assert response.status_code == 201
        collection = response.json()["collection"]
        assert collection["name"] == "api-docs"
        assert collection["allowed_extensions"] == [".pdf"]
        assert collection["document_count"] == 0
        assert collection["chunk_count"] == 0
        assert collection["last_updated"] is None

        listing = client.get("/collections").json()["collections"]
        assert [item["name"] for item in listing] == ["api-docs"]

    def test_create_invalid_name(self, client):
        response = client.post("/collections", json={"name": "Bad Name!"})
        assert response.status_code == 400

    def test_create_duplicate(self, client):
        assert client.post("/collections", json={"name": "twice"}).status_code == 201
        response = client.post("/collections", json={"name": "twice"})
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_update_collection(self, client):
        client.post("/collections", json={"name": "editable"})
        response = client.patch(
            "/collections/editable",
            json={
                "description": "new text",
                "allowed_extensions": [".txt"],
                "read_allowed": False,
                "write_allowed": True,
                "semantic_weight": 1.5,
                "keyword_weight": 3.0,
                "metadata_weight": 5.0,
            },
        )
        assert response.status_code == 200
        collection = response.json()["collection"]
        assert collection["description"] == "new text"
        assert collection["allowed_extensions"] == [".txt"]
        assert collection["read_allowed"] is False
        assert collection["write_allowed"] is True
        assert collection["semantic_weight"] == 1.5
        assert collection["keyword_weight"] == 3.0
        assert collection["metadata_weight"] == 5.0

    def test_delete_requires_confirm(self, client):
        client.post("/collections", json={"name": "cautious"})
        response = client.delete("/collections/cautious")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        assert len(client.get("/collections").json()["collections"]) == 1

    def test_delete_with_confirm(self, client):
        client.post("/collections", json={"name": "goner"})
        response = client.delete("/collections/goner?confirm=true")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert client.get("/collections").json()["collections"] == []

    def test_delete_unknown(self, client):
        assert client.delete("/collections/ghost?confirm=true").status_code == 404


class TestToolsApi:
    def test_list_tools(self, client):
        payload = client.get("/tools").json()
        tools = {tool["name"]: tool for tool in payload["tools"]}
        assert len(tools) == 11
        assert tools["search_documents"]["access"] == "read"
        assert tools["read_document"]["access"] == "read"
        assert tools["clear_index"]["access"] == "write"
        assert tools["ingest_file"]["params_schema"]["required"] == ["path"]

    def test_execute_search(self, client):
        response = client.post(
            "/tools/search_documents",
            json={"arguments": {"query": "anything", "collection_name": "all"}},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["result"] == []

    def test_unknown_tool_404(self, client):
        response = client.post("/tools/nonexistent", json={"arguments": {}})
        assert response.status_code == 404

    def test_list_tools_includes_permission(self, client):
        tools = {tool["name"]: tool for tool in client.get("/tools").json()["tools"]}
        assert tools["search_documents"]["permission"] == "allow"
        assert tools["ingest_file"]["permission"] == "ask"

    def test_visible_filter_excludes_denied(self, client):
        client.put("/settings", json={"changes": {"permissions": {"mode": "advanced", "tools": {"clear_index": "deny"}}}})
        all_names = {tool["name"] for tool in client.get("/tools").json()["tools"]}
        visible_names = {tool["name"] for tool in client.get("/tools?visible=true").json()["tools"]}
        assert "clear_index" in all_names
        assert "clear_index" not in visible_names

    def test_denied_tool_execution_forbidden(self, client):
        client.put("/settings", json={"changes": {"permissions": {"mode": "advanced", "tools": {"search_documents": "deny"}}}})
        response = client.post("/tools/search_documents", json={"arguments": {"query": "x"}})
        assert response.status_code == 403
        assert "deny" in response.json()["detail"].lower()

    def test_ask_from_mcp_denied_when_on_ask_deny(self, client):
        client.put("/settings", json={"changes": {"mcp_hosts": {"on_ask": "deny"}}})
        # ingest_file resolves to 'ask' (write group default); mcp client + on_ask=deny -> 403
        response = client.post(
            "/tools/ingest_file",
            json={"arguments": {"path": "x.txt"}, "client": "mcp"},
        )
        assert response.status_code == 403

    def test_ask_from_api_client_proceeds(self, client, tmp_path):
        source = tmp_path / "ask.txt"
        source.write_text("ask-path content", encoding="utf-8")
        # Default write=ask; api client should proceed (confirmation handled client-side).
        response = client.post(
            "/tools/ingest_file",
            json={"arguments": {"path": str(source), "collection_name": "ask-db"}, "client": "api"},
        )
        assert response.status_code == 200
        assert response.json()["result"]["documents_uploaded"] == 1

    def test_bad_arguments_400(self, client):
        # ingest_file without required 'path' argument.
        response = client.post("/tools/ingest_file", json={"arguments": {}})
        assert response.status_code == 400

    def test_ingest_and_search_via_tools(self, client, tmp_path):
        source = tmp_path / "api-note.txt"
        source.write_text("Content ingested through the tools endpoint.", encoding="utf-8")

        ingest = client.post(
            "/tools/ingest_file",
            json={"arguments": {"path": str(source), "collection_name": "via-api"}},
        )
        assert ingest.status_code == 200
        assert ingest.json()["result"]["documents_uploaded"] == 1

        # last_updated visible through the collections API.
        listing = client.get("/collections").json()["collections"]
        entry = next(item for item in listing if item["name"] == "via-api")
        assert entry["last_updated"] is not None
        assert entry["chunk_count"] >= 1


class TestSettingsApi:
    def test_get_settings(self, client):
        payload = client.get("/settings").json()
        assert payload["permissions"]["mode"] == "simple"
        assert payload["llm"]["provider"] == "lmstudio"

    def test_update_settings(self, client):
        response = client.put(
            "/settings",
            json={"changes": {"permissions": {"groups": {"write": "deny"}}}},
        )
        assert response.status_code == 200
        assert response.json()["permissions"]["groups"]["write"] == "deny"

    def test_invalid_settings_rejected(self, client):
        response = client.put(
            "/settings",
            json={"changes": {"permissions": {"mode": "chaotic"}}},
        )
        assert response.status_code == 400

    def test_api_key_masked_in_response(self, client):
        response = client.put("/settings", json={"changes": {"llm": {"api_key": "sk-123"}}})
        assert response.json()["llm"]["api_key"] == "***"


class TestPermissionsReplaceApi:
    def test_group_reset_clears_overrides(self, client):
        # Set an override, then send a group-reset payload (empty tools).
        client.put(
            "/settings/permissions",
            json={"permissions": {"mode": "advanced", "groups": {"read": "allow", "write": "ask"}, "tools": {"clear_index": "deny"}}},
        )
        assert client.get("/settings").json()["permissions"]["tools"] == {"clear_index": "deny"}

        response = client.put(
            "/settings/permissions",
            json={"permissions": {"mode": "advanced", "groups": {"read": "allow", "write": "deny"}, "tools": {}}},
        )
        assert response.status_code == 200
        perms = response.json()["permissions"]
        assert perms["tools"] == {}
        assert perms["groups"]["write"] == "deny"
        # clear_index now follows the write group again.
        tools = {t["name"]: t for t in client.get("/tools").json()["tools"]}
        assert tools["clear_index"]["permission"] == "deny"

    def test_invalid_permissions_rejected(self, client):
        response = client.put(
            "/settings/permissions",
            json={"permissions": {"mode": "advanced", "groups": {"read": "nope"}, "tools": {}}},
        )
        assert response.status_code == 400


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        line = block.strip()
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


class TestChatStreamApi:
    def test_chat_stream_with_fake_provider(self, client, tmp_path, monkeypatch):
        import chat_agent

        # Patch the provider builder so no real endpoint is needed.
        class FakeProvider:
            def stream(self, messages, tools=None):
                yield ("token", "Hi ")
                yield ("token", "there")
                yield ("message", {"role": "assistant", "content": "Hi there"})

        monkeypatch.setattr("rag_server.build_provider", lambda _llm: FakeProvider())

        response = client.post(
            "/chat/stream",
            json={"message": "hello", "read_collections": [], "write_collections": []},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)
        types = [e["type"] for e in events]
        assert types[0] == "session"
        assert "token" in types
        assert events[-1]["type"] == "done"
        assert events[-1]["content"] == "Hi there"

    def test_chat_stream_preserves_unicode(self, client, monkeypatch):
        class FakeProvider:
            def stream(self, messages, tools=None):
                yield ("token", "Grüße aus Köln: déjà vu.")
                yield ("message", {"role": "assistant", "content": "Grüße aus Köln: déjà vu."})

        monkeypatch.setattr("rag_server.build_provider", lambda _llm: FakeProvider())
        response = client.post(
            "/chat/stream",
            json={"message": "Wie geht es?", "read_collections": [], "write_collections": []},
        )

        assert "charset=utf-8" in response.headers["content-type"].lower()
        events = parse_sse(response.text)
        assert any(event.get("text") == "Grüße aus Köln: déjà vu." for event in events)

    def test_llm_test_endpoint_reports_failure(self, client, monkeypatch):
        from llm_providers import ProviderError

        class Broken:
            def list_models(self):
                raise ProviderError("nope")

        monkeypatch.setattr("rag_server.build_provider", lambda _llm: Broken())
        payload = client.post("/llm/test").json()
        assert payload["ok"] is False
        assert "nope" in payload["error"]

    def test_llm_models_endpoint(self, client, monkeypatch):
        class Ok:
            base_url = "http://localhost:1234"
            api_key = None
            model = "qwen"

            def list_models(self):
                return ["qwen", "llama"]

        monkeypatch.setattr("rag_server.build_provider", lambda _llm: Ok())
        assert client.get("/llm/models").json()["models"] == ["qwen", "llama"]

    def test_chat_stream_preserves_unicode(self, client, monkeypatch):
        class FakeProvider:
            def stream(self, messages, tools=None):
                text = "Grüße aus Köln: déjà vu."
                yield ("token", text)
                yield ("message", {"role": "assistant", "content": text})

        monkeypatch.setattr("rag_server.build_provider", lambda _llm: FakeProvider())

        response = client.post(
            "/chat/stream",
            json={"message": "Wie geht es?", "read_collections": [], "write_collections": []},
        )

        assert response.status_code == 200
        assert "charset=utf-8" in response.headers["content-type"].lower()
        events = parse_sse(response.text)
        assert any(event.get("text") == "Grüße aus Köln: déjà vu." for event in events)
    def test_chat_cancel_idle_for_unknown_session(self, client):
        payload = client.post("/chat/cancel", json={"session_id": "missing"}).json()
        assert payload["status"] == "idle"

    def test_chat_cancel_active_session(self, client):
        import runtime

        session_id = "session-cancel"
        runtime.chat_sessions().create(session_id, [], [])
        runtime.chat_sessions().begin_stream(session_id)

        payload = client.post("/chat/cancel", json={"session_id": session_id}).json()
        assert payload["status"] == "success"
        assert runtime.chat_sessions().get(session_id)["cancel_event"].is_set() is True

        runtime.chat_sessions().end_stream(session_id)


class TestMemoryApi:
    def test_memory_crud(self, client):
        created = client.post(
            "/memories",
            json={"category": "preference", "fact": "Prefers concise technical explanations."},
        )
        assert created.status_code == 201
        memory = created.json()["memory"]

        listed = client.get("/memories")
        assert listed.status_code == 200
        assert [item["fact"] for item in listed.json()["memories"]] == [memory["fact"]]

        updated = client.patch(
            f"/memories/{memory['id']}", json={"enabled": False}
        )
        assert updated.status_code == 200
        assert updated.json()["memory"]["enabled"] is False

        assert client.delete(f"/memories/{memory['id']}").status_code == 204
        assert client.get("/memories").json()["memories"] == []

    def test_chat_uses_relevant_memory_and_extraction_excludes_tool_data(self, client, monkeypatch):
        import runtime

        runtime.memories().add("preference", "Prefers Python for automation scripts.")
        extracted = threading.Event()
        captured_chat_messages = []
        captured_extraction_messages = []

        class FakeProvider:
            def stream(self, messages, tools=None):
                if messages[0]["content"] == "You return valid JSON and nothing else.":
                    captured_extraction_messages.extend(messages)
                    extracted.set()
                    yield (
                        "message",
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "memories": [
                                        {
                                            "category": "workflow",
                                            "fact": "Uses Python for small automation utilities.",
                                        }
                                    ]
                                }
                            ),
                        },
                    )
                    return
                captured_chat_messages.extend(messages)
                yield ("message", {"role": "assistant", "content": "Use Python."})

        monkeypatch.setattr("rag_server.build_provider", lambda _llm: FakeProvider())
        response = client.post(
            "/chat/stream",
            json={
                "message": "Which language should I use for an automation utility?",
                "read_collections": [],
                "write_collections": [],
            },
        )

        assert response.status_code == 200
        assert "Relevant saved user memory:\n- Prefers Python for automation scripts." in (
            captured_chat_messages[0]["content"]
        )
        assert extracted.wait(timeout=1)
        assert len(captured_extraction_messages) == 2
        extraction_text = captured_extraction_messages[1]["content"]
        assert "tool_call" not in extraction_text
        assert "tool_result" not in extraction_text
        assert [memory.fact for memory in runtime.memories().list()] == [
            "Uses Python for small automation utilities.",
            "Prefers Python for automation scripts.",
        ]
