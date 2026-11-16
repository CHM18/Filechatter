from __future__ import annotations

import config
import pytest
from rag_store import RagStore, _path_search_text


@pytest.fixture(autouse=True)
def unified_search(monkeypatch):
    monkeypatch.setattr(config, "LEGACY_METADATA_SEARCH", False)


def test_path_keywords_and_unambiguous_dates():
    assert "Ostern 2012" in _path_search_text(r"D:\Fotos\Ostern2012\a.jpg")
    assert "2012-03-13" in _path_search_text(r"Ostern13-03-2012\a.jpg")
    assert "März" in _path_search_text(r"Ostern03-03-2012\a.jpg")
    assert "2012-04-03" not in _path_search_text(r"Ostern03-04-2012\a.jpg")
    assert "März" not in _path_search_text(r"Ostern31-02-2012\a.jpg")


def test_unified_search_prefers_exif_year_and_filters_before_limit(data_env, monkeypatch):
    store = RagStore("photos")
    store.add_documents(
        ["Ostern Schnee"] * 4,
        [
            {"source": "Ostern2023/a.jpg", "image_capture_time": "2023-04-09T12:00:00"},
            {"source": "Ostern2012/b.jpg", "image_capture_time": "2012-04-08T12:00:00"},
            {"source": "Ostern2012/c.jpg", "image_capture_time": "2023-04-09T12:00:00"},
            {"source": "Ostern2012/d.jpg"},
        ],
    )
    store.semantic_weight = 0
    store.keyword_weight = 1
    store.metadata_weight = 10
    monkeypatch.setattr(config, "KEYWORD_RETRIEVAL_K", 20)
    results, debug = store.search_debug("ostern 2012 schnee")
    assert results[0].source == "Ostern2012/b.jpg"
    assert debug["preferred_year"] == 2012
    assert debug["weights"]["path"] == 0
    assert next(item for item in debug["candidates"] if item["source"] == "Ostern2012/c.jpg")["preferred_year_match"] is False
    assert next(item for item in debug["candidates"] if item["source"] == "Ostern2012/d.jpg")["preferred_year_match"] is True
    monkeypatch.setattr(config, "KEYWORD_RETRIEVAL_K", 1)
    results = store.search("ostern schnee", capture_year=2012)
    assert [item.source for item in results] == ["Ostern2012/b.jpg"]
    store.close()


def test_unified_index_rebuilds_and_stays_in_sync(data_env):
    store = RagStore("photos")
    store.add_documents(["content"], [{"source": "Ostern2012/a.jpg", "image_keywords": ["Schnee"]}])
    store._connection.execute("DELETE FROM chunks_unified_fts")
    store._connection.commit()
    store.close()
    store = RagStore("photos")
    store.semantic_weight = 0
    assert store.search("schnee")
    store.clear_source("Ostern2012/a.jpg")
    assert store._connection.execute("SELECT COUNT(*) FROM chunks_unified_fts").fetchone()[0] == 0
    store.close()


def test_capture_year_filters_semantic_candidates_before_limit(data_env, monkeypatch):
    store = RagStore("semantic-filter")
    store.add_documents(
        ["Ostern Schnee", "unrelated text"],
        [
            {"source": "Ostern2023/a.jpg", "image_capture_time": "2023-04-09T12:00:00"},
            {"source": "Ostern2012/b.jpg", "image_capture_time": "2012-04-08T12:00:00"},
        ],
    )
    store.semantic_weight = 1
    store.keyword_weight = 0
    monkeypatch.setattr(config, "RETRIEVAL_K", 1)
    results = store.search("Ostern Schnee", capture_year=2012)
    assert [item.source for item in results] == ["Ostern2012/b.jpg"]
    store.close()


def test_unified_collection_and_filter_api(client):
    assert client.post("/collections", json={"name": "photos"}).status_code == 201
    assert client.post("/upload", json={
        "collection_name": "photos",
        "documents": ["snow", "snow"],
        "metadata": [
            {"source": "Ostern2012/a.jpg", "image_capture_time": "2012-03-01T10:00:00"},
            {"source": "Ostern2023/b.jpg", "image_capture_time": "2023-03-01T10:00:00"},
        ],
    }).status_code == 200
    request = {"question": "snow", "collection_name": "photos", "capture_year": 2012}
    response = client.post("/search", json=request)
    assert response.status_code == 200
    assert [item["source"] for item in response.json()["results"]] == ["Ostern2012/a.jpg"]
    debug = client.post("/search/debug", json=request).json()
    assert debug["results"] == response.json()["results"]
    assert debug["collections"][0]["unified_search"] is True
    assert client.post("/search", json={**request, "capture_year": 10000}).status_code == 422


def test_snow_and_easter_year_rank_above_unrelated_photo(data_env, monkeypatch):
    store = RagStore("ranking")
    sources = [f"Ostern/Ostern2012/{i:03}.jpg" for i in range(70)] + ["Ostern/Ostern2023/other.jpg"]
    documents = ["Schnee am Ostersonntag" if i < 10 else "Osterfest ohne Winter" for i in range(70)]
    documents.append("Ein Kuchen auf der Terrasse")
    metadata = [
        {"source": source, "image_capture_time": "2012-04-08T12:00:00"}
        for source in sources[:-1]
    ]
    metadata.append({"source": sources[-1], "image_capture_time": "2023-04-09T12:00:00"})
    store.add_documents(documents, metadata)
    store.semantic_weight = 0
    store.keyword_weight = 1
    monkeypatch.setattr(config, "KEYWORD_RETRIEVAL_K", 100)
    monkeypatch.setattr(config, "MAX_CONTEXT_CHARS", 50000)
    results = store.search("ostern 2012 schnee")
    assert results
    assert "Ostern2012" in results[0].source
    assert "Schnee" in results[0].content
    assert next(i for i, result in enumerate(results) if result.source == sources[-1]) >= 10
    store.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_global_mode_applies_to_all_collections_and_ignores_saved_flags(client, monkeypatch, legacy):
    import runtime

    monkeypatch.setattr(config, "LEGACY_METADATA_SEARCH", legacy)
    manager = runtime.collections()
    for collection in ("photos", "documents"):
        manager.create(collection, semantic_weight=0, keyword_weight=1, metadata_weight=1)
        store = manager.get_store(collection)
        store.apply_collection_settings({
            "semantic_weight": 0, "keyword_weight": 1, "metadata_weight": 1,
            "unified_search": legacy,
        })
        store.add_documents(["ordinary content"], [{"source": "Ostern2012/photo.jpg"}])

    response = client.post("/search/debug", json={"question": "2012", "collection_name": "all"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["collections"]) >= 2
    for collection in ("photos", "documents"):
        debug = next(item for item in data["collections"] if item["collection_name"] == collection)
        assert debug["unified_search"] is not legacy
        assert debug["weights"]["path"] == (1 if legacy else 0)
        assert len(debug["candidates"]) == 1
        candidate = debug["candidates"][0]
        assert candidate["keyword_rank"] == (None if legacy else 1)
        assert candidate["path_rank"] == (1 if legacy else None)


def test_obsolete_per_collection_setting_is_rejected(client):
    response = client.post("/collections", json={"name": "photos", "unified_search": False})
    assert response.status_code == 400
    assert "LEGACY_METADATA_SEARCH" in response.json()["detail"]
    assert client.post("/collections", json={"name": "photos"}).status_code == 201
    response = client.patch("/collections/photos", json={"unified_search": True})
    assert response.status_code == 400
    assert "LEGACY_METADATA_SEARCH" in response.json()["detail"]


@pytest.mark.parametrize("value,expected", [("true", True), ("0", False), ("YES", True), ("off", False)])
def test_legacy_config_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("LEGACY_METADATA_SEARCH", value)
    assert config._bool_env("LEGACY_METADATA_SEARCH", False) is expected


def test_legacy_config_default_and_invalid(monkeypatch):
    monkeypatch.delenv("LEGACY_METADATA_SEARCH", raising=False)
    assert config._bool_env("LEGACY_METADATA_SEARCH", False) is False
    monkeypatch.setenv("LEGACY_METADATA_SEARCH", "sometimes")
    with pytest.raises(ValueError, match="LEGACY_METADATA_SEARCH"):
        config._bool_env("LEGACY_METADATA_SEARCH", False)
