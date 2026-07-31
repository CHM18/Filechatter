from __future__ import annotations

import pytest

from memory_store import MemoryStore


def test_memory_store_crud_and_fts_retrieval(data_env):
    store = MemoryStore(data_env)
    python = store.add("preference", "Prefers Python for automation scripts.")
    store.add("workflow", "Uses LM Studio locally for Filechatter.")

    assert [memory.fact for memory in store.search("Which language for an automation utility?")] == [
        "Prefers Python for automation scripts."
    ]

    updated = store.update(python.id, "preference", "Prefers TypeScript for browser code.", True)
    assert updated is not None
    assert store.search("automation") == []
    assert [memory.fact for memory in store.search("browser code")] == [
        "Prefers TypeScript for browser code."
    ]

    assert store.delete(python.id) is True
    assert store.get(python.id) is None


def test_memory_store_rejects_duplicates_and_disables_retrieval(data_env):
    store = MemoryStore(data_env)
    memory = store.add("workflow", "Use concise technical explanations.")

    with pytest.raises(ValueError, match="identical"):
        store.add("workflow", "  Use concise technical explanations. ")

    assert store.update(memory.id, "workflow", memory.fact, False) is not None
    assert store.search("technical explanations") == []