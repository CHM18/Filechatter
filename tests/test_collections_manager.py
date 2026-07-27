"""Unit tests for collections_manager."""
from __future__ import annotations

import json

import pytest

from collections_manager import CollectionsManager, normalize_extensions


@pytest.fixture()
def manager(data_env):
    return CollectionsManager(data_env)


class TestNormalizeExtensions:
    def test_adds_dot_and_lowercases(self):
        assert normalize_extensions(["PDF", ".Txt"]) == [".pdf", ".txt"]

    def test_deduplicates(self):
        assert normalize_extensions([".pdf", "pdf"]) == [".pdf"]

    def test_empty_means_no_restriction(self):
        assert normalize_extensions(None) == []
        assert normalize_extensions([]) == []


class TestCreate:
    def test_create_and_list(self, manager):
        entry = manager.create("project-docs", "Project documentation", [".pdf", "txt"])
        assert entry.name == "project-docs"
        assert entry.allowed_extensions == [".pdf", ".txt"]
        assert entry.created_at
        assert entry.last_updated is None
        assert [e.name for e in manager.list_entries()] == ["project-docs"]
        # Store directory materialized immediately.
        assert (manager.data_dir / "project-docs" / "rag.sqlite3").exists()

    def test_duplicate_rejected(self, manager):
        manager.create("dup")
        with pytest.raises(ValueError, match="already exists"):
            manager.create("dup")

    @pytest.mark.parametrize("bad_name", ["all", "*", "Has Spaces", "-leading", "a/b", ""])
    def test_invalid_names_rejected(self, manager, bad_name):
        with pytest.raises(ValueError):
            manager.create(bad_name)

    def test_uppercase_names_are_normalized(self, manager):
        entry = manager.create("MixedCase")
        assert entry.name == "mixedcase"

    def test_ensure_implicitly_creates(self, manager):
        entry = manager.ensure("autogen")
        assert entry.name == "autogen"
        assert manager.exists("autogen")

    def test_ensure_rejects_all(self, manager):
        with pytest.raises(ValueError):
            manager.ensure("all")


class TestAllowsExtension:
    def test_empty_allows_everything(self, manager):
        entry = manager.create("anything")
        assert entry.allows_extension(".pdf")
        assert entry.allows_extension(".txt")

    def test_restricted(self, manager):
        entry = manager.create("pdf-only", allowed_extensions=[".pdf"])
        assert entry.allows_extension(".pdf")
        assert entry.allows_extension("PDF")
        assert not entry.allows_extension(".txt")


class TestDelete:
    def test_delete_removes_entry_and_directory(self, manager):
        manager.create("doomed")
        collection_dir = manager.data_dir / "doomed"
        assert collection_dir.exists()
        result = manager.delete("doomed")
        assert result["deleted"] is True
        assert not collection_dir.exists()
        assert not manager.exists("doomed")

    def test_delete_populated_collection(self, manager):
        """Deleting a collection that holds indexed data must succeed - the
        store's sqlite/faiss files are released before the directory removal."""
        manager.create("populated")
        store = manager.get_store("populated")
        store.add_documents(
            ["Some indexed content about turbines and maintenance."],
            [{"source": "doc-1"}],
        )
        assert store.chunk_count > 0
        collection_dir = manager.data_dir / "populated"
        assert (collection_dir / "rag.sqlite3").exists()

        result = manager.delete("populated")
        assert result["deleted"] is True
        assert not collection_dir.exists()
        assert not manager.exists("populated")

    def test_delete_unknown_raises(self, manager):
        with pytest.raises(KeyError):
            manager.delete("nope")

    def test_delete_all_refused(self, manager):
        with pytest.raises(ValueError):
            manager.delete("all")


class TestPersistence:
    def test_registry_survives_restart(self, manager):
        manager.create("persist", "keep me", [".txt"])
        manager.close_all()

        reloaded = CollectionsManager(manager.data_dir)
        entry = reloaded.get_entry("persist")
        assert entry is not None
        assert entry.description == "keep me"
        assert entry.allowed_extensions == [".txt"]

    def test_touch_sets_last_updated(self, manager):
        manager.create("touched")
        manager.touch("touched")
        entry = manager.get_entry("touched")
        assert entry.last_updated is not None

        # Persisted to disk as well.
        payload = json.loads((manager.data_dir / "collections.json").read_text(encoding="utf-8"))
        stored = next(item for item in payload["collections"] if item["name"] == "touched")
        assert stored["last_updated"] == entry.last_updated

    def test_discovers_existing_directories(self, manager, data_env):
        # Simulate a collection directory that predates collections.json.
        legacy_dir = data_env / "legacy"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "rag.sqlite3").touch()

        manager.close_all()
        reloaded = CollectionsManager(data_env)
        assert reloaded.exists("legacy")

    def test_selected_stores_all(self, manager):
        manager.create("one")
        manager.create("two")
        stores = manager.selected_stores("all")
        assert sorted(store.collection_name for store in stores) == ["one", "two"]
