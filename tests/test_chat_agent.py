"""Agent-loop tests using a scripted fake provider (no LLM required)."""
from __future__ import annotations

import json

import pytest

import chat_agent
import permissions
import runtime
import tool_registry


def tool_call(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class FakeProvider:
    """Yields scripted assistant turns. Each turn: {content?, tool_calls?, tokens?}."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    def stream(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        turn = self.turns.pop(0)
        for token in turn.get("tokens", []):
            yield ("token", token)
        message = {"role": "assistant", "content": turn.get("content", "")}
        if turn.get("tool_calls"):
            message["tool_calls"] = turn["tool_calls"]
        yield ("message", message)


def collect(provider, messages, read_cols, write_cols, decisions=None):
    return list(chat_agent.run(provider, messages, read_cols, write_cols, decisions))


def ingest(tmp_path, collection, text, name="doc.txt"):
    path = tmp_path / f"{collection}_{name}"
    path.write_text(text, encoding="utf-8")
    tool_registry.execute("ingest_file", {"path": str(path), "collection_name": collection})


class TestBuildTools:
    def test_read_tools_drop_collection_name(self, data_env):
        perms = runtime.settings().get()["permissions"]
        tools, plan = chat_agent.build_chat_tools(["docs"], [], perms)
        search = next(t for t in tools if t["function"]["name"] == "search_documents")
        assert "collection_name" not in search["function"]["parameters"]["properties"]
        assert plan["search_documents"] == "read"

    def test_write_tools_hidden_without_write_db(self, data_env):
        perms = runtime.settings().get()["permissions"]
        names = {t["function"]["name"] for t in chat_agent.build_chat_tools(["docs"], [], perms)[0]}
        assert "ingest_file" not in names
        assert "clear_index" not in names
        assert "search_documents" in names

    def test_single_write_db_autofills(self, data_env):
        runtime.collections().create("docs")
        perms = runtime.settings().get()["permissions"]
        tools, _ = chat_agent.build_chat_tools([], ["docs"], perms)
        ingest_tool = next(t for t in tools if t["function"]["name"] == "ingest_file")
        assert "collection_name" not in ingest_tool["function"]["parameters"]["properties"]

    def test_multi_write_db_enum(self, data_env):
        runtime.collections().create("a")
        runtime.collections().create("b")
        perms = runtime.settings().get()["permissions"]
        tools, _ = chat_agent.build_chat_tools([], ["a", "b"], perms)
        ingest_tool = next(t for t in tools if t["function"]["name"] == "ingest_file")
        prop = ingest_tool["function"]["parameters"]["properties"]["collection_name"]
        assert set(prop["enum"]) == {"a", "b"}

    def test_denied_tool_excluded(self, data_env):
        runtime.settings().update({"permissions": {"mode": "advanced", "tools": {"search_documents": "deny"}}})
        perms = runtime.settings().get()["permissions"]
        names = {t["function"]["name"] for t in chat_agent.build_chat_tools(["docs"], [], perms)[0]}
        assert "search_documents" not in names


class TestReadFlow:
    def test_search_then_answer(self, data_env, tmp_path):
        runtime.collections().create("docs")
        ingest(tmp_path, "docs", "The capital of France is Paris.")
        provider = FakeProvider([
            {"tool_calls": [tool_call("c1", "search_documents", {"query": "capital of France"})]},
            {"content": "It is Paris."},
        ])
        events = collect(provider, [], ["docs"], [])
        types = [e["type"] for e in events]
        assert "tool_call" in types and "tool_result" in types
        assert events[-1] == {"type": "done", "content": "It is Paris."}
        tool_result = next(e for e in events if e["type"] == "tool_result")
        assert tool_result["ok"] is True

    def test_read_scoped_to_selected_databases(self, data_env, tmp_path):
        runtime.collections().create("docs")
        runtime.collections().create("secret")
        ingest(tmp_path, "docs", "Pumpkins are orange.")
        ingest(tmp_path, "secret", "Pumpkins are secretly blue.")
        provider = FakeProvider([
            {"tool_calls": [tool_call("c1", "search_documents", {"query": "pumpkins"})]},
            {"content": "done"},
        ])
        collect(provider, [], ["docs"], [])
        # The second provider call sees the tool result; it must only contain docs.
        tool_messages = [m for m in provider.calls[1]["messages"] if m.get("role") == "tool"]
        assert tool_messages
        payload = json.loads(tool_messages[0]["content"])
        collections = {hit["collection_name"] for hit in payload}
        assert collections == {"docs"}


class TestWriteAskFlow:
    def _provider(self):
        return FakeProvider([
            {"tool_calls": [tool_call("w1", "ingest_file", {"path": "IGNORED"})]},
            {"content": "Ingested."},
        ])

    def test_write_pauses_for_approval(self, data_env, tmp_path):
        runtime.collections().create("docs")
        source = tmp_path / "note.txt"
        source.write_text("hello world", encoding="utf-8")
        provider = FakeProvider([
            {"tool_calls": [tool_call("w1", "ingest_file", {"path": str(source)})]},
            {"content": "Ingested."},
        ])
        messages = []
        events = collect(provider, messages, [], ["docs"])
        confirmation = next(e for e in events if e["type"] == "need_confirmation")
        pending = confirmation["pending"][0]
        assert pending["id"] == "w1"
        assert pending["needs_approval"] is True
        # Nothing ingested yet.
        assert tool_registry.execute("list_sources", {"collection_name": "docs"})["total_documents"] == 0

        # Resume with approval.
        resume = collect(provider, messages, [], ["docs"], {"w1": {"approved": True}})
        assert resume[-1]["type"] == "done"
        assert tool_registry.execute("list_sources", {"collection_name": "docs"})["total_documents"] == 1

    def test_write_denied(self, data_env, tmp_path):
        runtime.collections().create("docs")
        source = tmp_path / "note.txt"
        source.write_text("hello world", encoding="utf-8")
        provider = FakeProvider([
            {"tool_calls": [tool_call("w1", "ingest_file", {"path": str(source)})]},
            {"content": "Okay, skipped."},
        ])
        messages = []
        collect(provider, messages, [], ["docs"])
        resume = collect(provider, messages, [], ["docs"], {"w1": {"approved": False}})
        denied = next(e for e in resume if e["type"] == "tool_result")
        assert denied["ok"] is False
        assert resume[-1]["type"] == "done"
        assert tool_registry.execute("list_sources", {"collection_name": "docs"})["total_documents"] == 0

    def test_allow_write_runs_without_confirmation(self, data_env, tmp_path):
        runtime.settings().update({"permissions": {"groups": {"write": "allow"}}})
        runtime.collections().create("docs")
        source = tmp_path / "note.txt"
        source.write_text("hello world", encoding="utf-8")
        provider = FakeProvider([
            {"tool_calls": [tool_call("w1", "ingest_file", {"path": str(source)})]},
            {"content": "Ingested."},
        ])
        events = collect(provider, [], [], ["docs"])
        assert not any(e["type"] == "need_confirmation" for e in events)
        assert tool_registry.execute("list_sources", {"collection_name": "docs"})["total_documents"] == 1


class TestMultiWriteRouting:
    def test_ambiguous_write_asks_to_choose(self, data_env, tmp_path):
        runtime.collections().create("manuals")
        runtime.collections().create("code")
        source = tmp_path / "note.txt"
        source.write_text("content", encoding="utf-8")
        provider = FakeProvider([
            {"tool_calls": [tool_call("w1", "ingest_file", {"path": str(source)})]},
            {"content": "Ingested."},
        ])
        messages = []
        events = collect(provider, messages, [], ["manuals", "code"])
        pending = next(e for e in events if e["type"] == "need_confirmation")["pending"][0]
        assert pending["needs_approval"] is True
        assert set(pending["options"]) == {"manuals", "code"}

        # Resume: approve + choose 'manuals'.
        collect(provider, messages, [], ["manuals", "code"], {"w1": {"approved": True, "collection": "manuals"}})
        assert tool_registry.execute("list_sources", {"collection_name": "manuals"})["total_documents"] == 1
        assert tool_registry.execute("list_sources", {"collection_name": "code"})["total_documents"] == 0

    def test_model_picked_valid_target_still_needs_approval_only(self, data_env, tmp_path):
        runtime.collections().create("manuals")
        runtime.collections().create("code")
        source = tmp_path / "note.txt"
        source.write_text("content", encoding="utf-8")
        provider = FakeProvider([
            {"tool_calls": [tool_call("w1", "ingest_file", {"path": str(source), "collection_name": "code"})]},
            {"content": "Ingested."},
        ])
        messages = []
        events = collect(provider, messages, [], ["manuals", "code"])
        pending = next(e for e in events if e["type"] == "need_confirmation")["pending"][0]
        # Model already chose a valid target, so no options needed, just approval.
        assert pending["options"] is None
        assert pending["needs_approval"] is True
        collect(provider, messages, [], ["manuals", "code"], {"w1": {"approved": True}})
        assert tool_registry.execute("list_sources", {"collection_name": "code"})["total_documents"] == 1


class TestErrors:
    def test_provider_error_surfaces(self, data_env):
        from llm_providers import ProviderError

        class BrokenProvider:
            def stream(self, messages, tools=None):
                raise ProviderError("endpoint down")
                yield  # pragma: no cover

        events = collect(BrokenProvider(), [], ["docs"], [])
        assert events[-1]["type"] == "error"
        assert "endpoint down" in events[-1]["message"]
