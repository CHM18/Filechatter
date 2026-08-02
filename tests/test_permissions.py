"""Unit tests for permission resolution and enforcement."""
from __future__ import annotations

import pytest

import permissions
import runtime


def set_permissions(**changes):
    runtime.settings().update({"permissions": changes})


class TestResolveSimpleMode:
    def test_defaults_read_allow_write_ask(self, data_env):
        assert permissions.resolve("search_documents") == "allow"   # read
        assert permissions.resolve("list_sources") == "allow"       # read
        assert permissions.resolve("ingest_file") == "ask"          # write
        assert permissions.resolve("clear_index") == "ask"          # write

    def test_group_change_affects_all_in_group(self, data_env):
        set_permissions(groups={"write": "deny"})
        assert permissions.resolve("ingest_file") == "deny"
        assert permissions.resolve("start_ingest_directory") == "deny"
        assert permissions.resolve("clear_index") == "deny"
        # Reads unaffected.
        assert permissions.resolve("search_documents") == "allow"

    def test_tool_overrides_ignored_in_simple_mode(self, data_env):
        set_permissions(mode="simple", tools={"clear_index": "allow"})
        # Simple mode ignores per-tool entries; clear_index follows write group.
        assert permissions.resolve("clear_index") == "ask"


class TestResolveAdvancedMode:
    def test_tool_override_wins(self, data_env):
        set_permissions(mode="advanced", groups={"write": "ask"}, tools={"clear_index": "deny"})
        assert permissions.resolve("clear_index") == "deny"
        # Other write tools still follow the group default.
        assert permissions.resolve("ingest_file") == "ask"

    def test_unlisted_tool_falls_back_to_group(self, data_env):
        set_permissions(mode="advanced", groups={"read": "deny"}, tools={"search_documents": "allow"})
        assert permissions.resolve("search_documents") == "allow"  # explicit override
        assert permissions.resolve("list_sources") == "deny"       # falls back to read group


class TestVisibility:
    def test_denied_tool_not_visible(self, data_env):
        set_permissions(mode="advanced", tools={"clear_index": "deny"})
        names = permissions.visible_tool_names()
        assert "clear_index" not in names
        assert "search_documents" in names

    def test_all_visible_by_default(self, data_env):
        names = permissions.visible_tool_names()
        assert len(names) == 8
        assert {"search_documents", "list_sources", "ingest_file", "start_ingest_directory", "clear_index"}.issubset(names)

    def test_unknown_tool_raises(self, data_env):
        with pytest.raises(KeyError):
            permissions.resolve("no_such_tool")


class TestEnforce:
    def test_allow_passes(self, data_env):
        assert permissions.enforce("search_documents") == "allow"

    def test_deny_blocks_any_client(self, data_env):
        set_permissions(mode="advanced", tools={"clear_index": "deny"})
        with pytest.raises(permissions.PermissionDenied):
            permissions.enforce("clear_index", client=permissions.CLIENT_API)
        with pytest.raises(permissions.PermissionDenied):
            permissions.enforce("clear_index", client=permissions.CLIENT_MCP)

    def test_ask_passes_for_api_client(self, data_env):
        # Web chat handles the confirmation above this layer, so ask -> proceed.
        assert permissions.enforce("ingest_file", client=permissions.CLIENT_API) == "ask"

    def test_ask_from_mcp_host_confirm_allows(self, data_env):
        runtime.settings().update({"mcp_hosts": {"on_ask": "host_confirm"}})
        assert permissions.enforce("ingest_file", client=permissions.CLIENT_MCP) == "ask"

    def test_ask_from_mcp_deny_blocks(self, data_env):
        runtime.settings().update({"mcp_hosts": {"on_ask": "deny"}})
        with pytest.raises(permissions.PermissionDenied):
            permissions.enforce("ingest_file", client=permissions.CLIENT_MCP)

    def test_ask_from_mcp_elicit_allows(self, data_env):
        runtime.settings().update({"mcp_hosts": {"on_ask": "elicit"}})
        # Elicitation falls back to proceeding at the server layer.
        assert permissions.enforce("ingest_file", client=permissions.CLIENT_MCP) == "ask"


class TestDescribe:
    def test_describe_reports_override(self, data_env):
        set_permissions(mode="advanced", tools={"clear_index": "deny"})
        described = permissions.describe("clear_index")
        assert described["permission"] == "deny"
        assert described["has_override"] is True
        assert described["access"] == "write"

    def test_describe_no_override(self, data_env):
        described = permissions.describe("search_documents")
        assert described["has_override"] is False
        assert described["permission"] == "allow"
