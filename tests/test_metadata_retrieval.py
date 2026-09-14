from __future__ import annotations

from rag_store import RagStore


def test_source_path_matches_outrank_generic_content_matches(data_env, monkeypatch):
    store = RagStore("metadata-ranking")
    store.add_documents(
        [
            "Short project note with no identifying names.",
            "Schneider Köln appears in this generic meeting transcript.",
        ],
        [
            {"source": "archives/2024/Köln/Schneider_budget.xlsx"},
            {"source": "notes/general.txt"},
        ],
    )
    monkeypatch.setattr(store, "_semantic_candidates", lambda _question: [])

    results = store.search("Schneider Köln")

    assert results[0].source == "archives/2024/Köln/Schneider_budget.xlsx"
    assert results[1].source == "notes/general.txt"