"""Unit tests for settings_manager."""
from __future__ import annotations

import json

import pytest

from settings_manager import SettingsManager, default_settings


@pytest.fixture()
def manager(data_env):
    return SettingsManager(data_env)


class TestDefaults:
    def test_defaults_written_on_first_start(self, manager, data_env):
        assert (data_env / "settings.json").exists()
        settings = manager.get()
        assert settings["permissions"]["mode"] == "simple"
        assert settings["permissions"]["groups"] == {"read": "allow", "write": "ask"}
        assert settings["mcp_hosts"]["on_ask"] == "host_confirm"

    def test_get_returns_copy(self, manager):
        settings = manager.get()
        settings["permissions"]["mode"] = "advanced"
        assert manager.get()["permissions"]["mode"] == "simple"


class TestUpdate:
    def test_deep_merge_preserves_siblings(self, manager):
        manager.update({"llm": {"model": "qwen3-14b"}})
        settings = manager.get()
        assert settings["llm"]["model"] == "qwen3-14b"
        # Sibling keys untouched by the merge.
        assert settings["llm"]["base_url"] == default_settings()["llm"]["base_url"]

    def test_update_persists(self, manager, data_env):
        manager.update({"permissions": {"mode": "advanced", "tools": {"clear_index": "deny"}}})
        stored = json.loads((data_env / "settings.json").read_text(encoding="utf-8"))
        assert stored["permissions"]["mode"] == "advanced"
        assert stored["permissions"]["tools"]["clear_index"] == "deny"

    def test_reload_after_restart(self, manager, data_env):
        manager.update({"llm": {"provider": "anthropic"}})
        reloaded = SettingsManager(data_env)
        assert reloaded.get()["llm"]["provider"] == "anthropic"

    @pytest.mark.parametrize(
        "changes",
        [
            {"permissions": {"mode": "bogus"}},
            {"permissions": {"groups": {"read": "maybe"}}},
            {"permissions": {"groups": {"execute": "allow"}}},
            {"permissions": {"tools": {"search_documents": "sometimes"}}},
            {"mcp_hosts": {"on_ask": "explode"}},
        ],
    )
    def test_invalid_values_rejected(self, manager, changes):
        with pytest.raises(ValueError):
            manager.update(changes)

    def test_failed_update_leaves_settings_unchanged(self, manager):
        before = manager.get()
        with pytest.raises(ValueError):
            manager.update({"permissions": {"mode": "bogus"}})
        assert manager.get() == before


class TestSetPermissions:
    def test_replace_removes_overrides(self, manager):
        manager.update({"permissions": {"mode": "advanced", "tools": {"clear_index": "deny", "ingest_file": "allow"}}})
        # Replace with a block that drops the overrides (what a group reset sends).
        manager.set_permissions({"mode": "advanced", "groups": {"read": "allow", "write": "ask"}, "tools": {}})
        perms = manager.get()["permissions"]
        assert perms["tools"] == {}
        assert perms["groups"]["write"] == "ask"

    def test_partial_payload_fills_defaults(self, manager):
        manager.set_permissions({"mode": "advanced"})
        perms = manager.get()["permissions"]
        assert perms["groups"] == {"read": "allow", "write": "ask"}
        assert perms["tools"] == {}

    def test_invalid_rejected_and_unchanged(self, manager):
        before = manager.get()["permissions"]
        with pytest.raises(ValueError):
            manager.set_permissions({"mode": "bogus", "groups": {}, "tools": {}})
        assert manager.get()["permissions"] == before


class TestPublicView:
    def test_api_key_masked(self, manager):
        manager.update({"llm": {"api_key": "sk-secret"}})
        assert manager.public_view()["llm"]["api_key"] == "***"
        # Real value still stored.
        assert manager.get()["llm"]["api_key"] == "sk-secret"

    def test_empty_api_key_not_masked(self, manager):
        assert manager.public_view()["llm"]["api_key"] is None
