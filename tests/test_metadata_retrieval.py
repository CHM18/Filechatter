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


def test_fingerprint_replacement_keeps_old_chunks_when_embedding_fails(data_env, monkeypatch):
    store = RagStore("fingerprint-failure")
    metadata = {"source": "a.pdf", "file_changed_epoch": 100.0, "extraction_fingerprint": "old"}
    store.add_documents(["original content"], [metadata])

    def fail(*args, **kwargs):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(store.embedding_model, "encode", fail)
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        store.add_documents(["new content"], [{**metadata, "extraction_fingerprint": "new"}])
    assert [chunk["content"] for chunk in store.get_chunks("a.pdf")] == ["original content"]


def test_fingerprints_keep_identical_and_reject_older_sources(data_env):
    store = RagStore("fingerprint-order")
    metadata = {"source": "a.pdf", "file_changed_epoch": 100.0, "extraction_fingerprint": "old"}
    store.add_documents(["original"], [metadata])
    unchanged = store.add_documents(["same"], [metadata])
    older = store.add_documents(["older"], [{**metadata, "file_changed_epoch": 99.0,
                                              "extraction_fingerprint": "new"}])
    assert unchanged["skipped_outdated_documents"] == 1
    assert older["skipped_outdated_documents"] == 1
    assert store.get_chunks("a.pdf")[0]["content"] == "original"


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


def test_search_debug_reports_actual_ranks_and_selection(data_env, monkeypatch):
    store = RagStore("debug-ranking")
    store.add_documents(
        ["snow winter", "easter " + "x" * 100, "snow spring", "other"],
        [
            {"source": "a.jpg", "image_capture_time": "2012"},
            {"source": "b.jpg"},
            {"source": "a.jpg", "file_changed_epoch": 1},
            {"source": "c.jpg"},
        ],
    )
    ids = {row["content"]: row["id"] for row in store._connection.execute("SELECT id, content FROM chunks")}
    a, b, c, d = (ids[text] for text in ("snow winter", "easter " + "x" * 100, "snow spring", "other"))
    monkeypatch.setattr(store, "_semantic_candidates", lambda question: [a, b, c, d])
    monkeypatch.setattr(store, "_keyword_candidates", lambda question: ([b, a], [c, a]))
    monkeypatch.setattr(config, "MAX_CONTEXT_CHARS", 20)
    monkeypatch.setattr(config, "MAX_CHUNKS_PER_SOURCE", 1)
    store.semantic_weight = store.keyword_weight = store.metadata_weight = 1.0

    expected = store.search("easter 2012 snow")
    actual, debug = store.search_debug("easter 2012 snow")

    assert actual == expected
    candidates = {item["chunk_id"]: item for item in debug["candidates"]}
    assert candidates[a]["semantic_rank"] == 1
    assert candidates[a]["keyword_rank"] == 2
    assert candidates[a]["path_rank"] == 2
    assert candidates[a]["combined_score"] == pytest.approx(1 / 61 + 1 / 62 + 1 / 62)
    assert candidates[a]["combined_rank"] == 1
    assert candidates[a]["selection"] == "selected"
    assert candidates[a]["metadata"]["image_capture_time"] == "2012"
    assert candidates[b]["selection"] == "context_budget"
    assert candidates[c]["selection"] == "max_chunks_per_source"
    assert candidates[d]["selection"] == "selected"
    assert candidates[d]["keyword_rank"] is None
    assert debug["weights"] == {"semantic": 1.0, "keyword": 1.0, "path": 1.0}


def test_search_debug_reports_candidates_after_budget_exhaustion(data_env, monkeypatch):
    store = RagStore("debug-exhausted")
    store.add_documents(["abcdefghij", "second"], [{"source": "a"}, {"source": "b"}])
    ids = [row["id"] for row in store._connection.execute("SELECT id FROM chunks ORDER BY id")]
    monkeypatch.setattr(store, "_semantic_candidates", lambda question: ids)
    store.keyword_weight = store.metadata_weight = 0.0
    monkeypatch.setattr(config, "MAX_CONTEXT_CHARS", 4)

    results, debug = store.search_debug("test")

    assert results == store.search("test")
    assert results[0].content == "abcd"
    assert [candidate["selection"] for candidate in debug["candidates"]] == [
        "selected_truncated", "not_evaluated_context_budget"
    ]