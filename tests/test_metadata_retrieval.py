from __future__ import annotations

import pytest

import config
from rag_store import RagStore


@pytest.mark.parametrize("channel", ["semantic", "keyword"])
def test_source_filter_precedes_candidate_limit(data_env, monkeypatch, channel):
    store = RagStore("source-filter")
    query = "Development output verification"
    store.add_documents(
        [query] * 20 + ["Contents and introduction.", query],
        [{"source": f"forms/form-{index}.pdf"} for index in range(20)]
        + [{"source": "sops/SOP 7300.pdf"}] * 2,
    )
    monkeypatch.setattr(config, "RETRIEVAL_K", 1)
    monkeypatch.setattr(config, "KEYWORD_RETRIEVAL_K", 1)
    store.semantic_weight = 1.0 if channel == "semantic" else 0.0
    store.keyword_weight = 1.0 if channel == "keyword" else 0.0
    store.metadata_weight = 0.0

    for source in ["sops/SOP 7300.pdf", "sop 7300.PDF"]:
        results = store.search(query, source=source)
        assert len(results) == 1
        assert results[0].source == "sops/SOP 7300.pdf"
        assert results[0].content == query
    assert store.search(query, source="missing.pdf") == []
    store.read_allowed = False
    assert store.search(query, source="SOP 7300.pdf") == []


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


def test_analysis_state_and_force_replace(data_env):
    store = RagStore("analysis-state")
    meta = {"source": "a.jpg", "file_changed_epoch": 100.0, "image_analysis_fingerprint": "fp1"}
    assert store.analysis_state("a.jpg") is None
    store.add_documents(["first"], [meta])
    assert store.analysis_state("a.jpg") == (100.0, "fp1")

    result = store.add_documents(["second"], [{**meta, "image_analysis_fingerprint": "fp2"}])
    assert result["documents_uploaded"] == 0

    result = store.add_documents(["second"], [{**meta, "image_analysis_fingerprint": "fp2"}], force_replace=True)
    assert result["documents_uploaded"] == 1
    assert store.analysis_state("a.jpg") == (100.0, "fp2")


def test_search_returns_no_results_when_collection_is_not_readable(data_env):
    store = RagStore("read-disabled")
    store.add_documents(["A document about turbine maintenance."], [{"source": "doc.txt"}])
    store.read_allowed = False

    assert store.search("turbine") == []


def test_search_returns_no_results_when_all_retrieval_weights_are_zero(data_env):
    store = RagStore("zero-weight")
    store.add_documents(["A document about turbine maintenance."], [{"source": "doc.txt"}])
    store.semantic_weight = 0.0
    store.keyword_weight = 0.0
    store.metadata_weight = 0.0

    assert store.search("turbine") == []