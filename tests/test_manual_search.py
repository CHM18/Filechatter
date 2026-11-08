from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import requests


spec = importlib.util.spec_from_file_location("manual_search", Path(__file__).with_name("manual_search.py"))
assert spec is not None and spec.loader is not None
manual_search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manual_search)


def test_manual_search_writes_complete_response(client, tmp_path, monkeypatch):
    client.post("/upload", json={
        "documents": ["Schnee zu Ostern 2012"],
        "metadata": [{"source": "2012.jpg"}],
        "collection_name": "photos",
    })
    calls = []

    def post(url, json, timeout):
        calls.append((url, json, timeout))
        return client.post("/search/debug", json=json)

    monkeypatch.setattr(manual_search.requests, "post", post)
    output = tmp_path / "suchergebnis.json"
    monkeypatch.setattr(sys, "argv", [
        "manual_search.py", "ostern 2012 schnee", "photos", "-o", str(output),
        "--server", "http://example.test/", "--source", "2012.jpg",
    ])
    assert manual_search.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [("http://example.test/search/debug", {
        "question": "ostern 2012 schnee", "collection_name": "photos", "source": "2012.jpg"
    }, 120)]
    assert result["results"][0]["source"] == "2012.jpg"
    candidate = result["collections"][0]["candidates"][0]
    assert candidate["selection"] == "selected"
    for channel in ("semantic_rank", "keyword_rank", "path_rank"):
        assert result["results"][0][channel] == candidate[channel]
    assert "Schnee zu Ostern" in output.read_text(encoding="utf-8")


def test_result_ranks_use_collection_and_chunk_id():
    data = {
        "results": [
            {"collection_name": "a", "chunk_id": 1, "rank": 2},
            {"collection_name": "b", "chunk_id": 1, "rank": 1},
        ],
        "collections": [
            {"collection_name": "a", "candidates": [
                {"chunk_id": 1, "semantic_rank": 2, "keyword_rank": None, "path_rank": 1}
            ]},
            {"collection_name": "b", "candidates": [
                {"chunk_id": 1, "semantic_rank": None, "keyword_rank": 3, "path_rank": 4}
            ]},
        ],
    }
    manual_search.add_result_ranks(data)
    assert data["results"] == [
        {"semantic_rank": 2, "keyword_rank": None, "path_rank": 1,
         "collection_name": "a", "chunk_id": 1, "rank": 2},
        {"semantic_rank": None, "keyword_rank": 3, "path_rank": 4,
         "collection_name": "b", "chunk_id": 1, "rank": 1},
    ]


def test_manual_search_prints_and_reports_errors(monkeypatch, capsys, tmp_path):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [], "collections": []}

    monkeypatch.setattr(manual_search.requests, "post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(sys, "argv", ["manual_search.py", "question", "all"])
    assert manual_search.main() == 0
    assert json.loads(capsys.readouterr().out)["results"] == []

    monkeypatch.setattr(sys, "argv", ["manual_search.py", "question", "all", "-o", str(tmp_path)])
    assert manual_search.main() == 1
    assert "Cannot write output file" in capsys.readouterr().err

    def fail(*args, **kwargs):
        raise requests.ConnectionError("server unavailable")

    monkeypatch.setattr(manual_search.requests, "post", fail)
    assert manual_search.main() == 1
    assert "server unavailable" in capsys.readouterr().err


def test_manual_search_rejects_missing_candidate(monkeypatch, capsys):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"collection_name": "photos", "chunk_id": 7}], "collections": []}

    monkeypatch.setattr(manual_search.requests, "post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(sys, "argv", ["manual_search.py", "question", "photos"])
    assert manual_search.main() == 1
    assert "Incomplete or invalid search-debug response" in capsys.readouterr().err
