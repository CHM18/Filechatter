"""API tests for the FastAPI server (collections CRUD, tools endpoint, settings)."""
from __future__ import annotations

import json


class TestHealth:
    def test_health_reports_collections(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert "collections_count" in payload
        assert "chunks_count" in payload


class TestFileTypesApi:
    def test_file_types_endpoint(self, client):
        categories = client.get("/file-types").json()["categories"]
        names = {c["category"] for c in categories}
        assert {"Office", "Text", "Web", "Code"}.issubset(names)
        office = next(c for c in categories if c["category"] == "Office")
        exts = {e["ext"] for e in office["extensions"]}
        assert {".pdf", ".xlsx", ".pptx"}.issubset(exts)


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
            json={"description": "new text", "allowed_extensions": [".txt"]},
        )
        assert response.status_code == 200
        collection = response.json()["collection"]
        assert collection["description"] == "new text"
        assert collection["allowed_extensions"] == [".txt"]

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
        assert len(tools) == 8
        assert tools["search_documents"]["access"] == "read"
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
            def list_models(self):
                return ["qwen", "llama"]

        monkeypatch.setattr("rag_server.build_provider", lambda _llm: Ok())
        assert client.get("/llm/models").json()["models"] == ["qwen", "llama"]
