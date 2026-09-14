"""Unit tests for the shared tool registry (search, ingest, clear, jobs)."""
from __future__ import annotations

from time import sleep, time

import pytest

import runtime
import tool_registry


def wait_for_job(job_id: str, timeout: float = 15.0) -> dict:
    deadline = time() + timeout
    while time() < deadline:
        status = tool_registry.execute("get_ingest_status", {"job_id": job_id})
        if status["status"] in {"completed", "failed"}:
            return status
        sleep(0.05)
    raise AssertionError(f"Ingest job {job_id} did not finish within {timeout}s")


class TestRegistry:
    def test_all_tools_registered_with_access(self):
        specs = {spec.name: spec for spec in tool_registry.list_specs()}
        assert set(specs) == {
            "search_documents",
            "list_sources",
            "get_document_support",
            "get_ingest_status",
            "get_ingest_log",
            "ingest_file",
            "start_ingest_directory",
            "clear_index",
        }
        read_tools = {name for name, spec in specs.items() if spec.access == "read"}
        write_tools = {name for name, spec in specs.items() if spec.access == "write"}
        assert write_tools == {"ingest_file", "start_ingest_directory", "clear_index"}
        assert read_tools == set(specs) - write_tools

    def test_every_tool_has_params_schema(self):
        for spec in tool_registry.list_specs():
            assert spec.params_schema["type"] == "object"
            assert "properties" in spec.params_schema

    def test_execute_unknown_tool_raises(self, data_env):
        with pytest.raises(KeyError):
            tool_registry.execute("does_not_exist", {})


class TestIngestFile:
    def test_roundtrip_search(self, data_env, tmp_path):
        source = tmp_path / "note.txt"
        source.write_text("The Filechatter retriever stores document chunks persistently.", encoding="utf-8")

        result = tool_registry.execute(
            "ingest_file", {"path": str(source), "collection_name": "notes"}
        )
        assert result["documents_uploaded"] == 1
        assert result["chunks_uploaded"] >= 1
        assert result["collection_name"] == "notes"

        hits = tool_registry.execute(
            "search_documents", {"query": "Filechatter retriever", "collection_name": "notes"}
        )
        assert hits
        assert hits[0]["collection_name"] == "notes"

        listing = tool_registry.execute("list_sources", {"collection_name": "notes"})
        assert listing["total_documents"] == 1
        assert listing["total_chunks"] >= 1
        assert listing["summary"].startswith("1 document")
        assert len(listing["sources"]) == 1

    def test_updates_last_updated(self, data_env, tmp_path):
        source = tmp_path / "stamp.txt"
        source.write_text("timestamp test content", encoding="utf-8")
        tool_registry.execute("ingest_file", {"path": str(source), "collection_name": "stamped"})
        entry = runtime.collections().get_entry("stamped")
        assert entry.last_updated is not None

    def test_extension_enforcement(self, data_env, tmp_path):
        runtime.collections().create("pdf-only", allowed_extensions=[".pdf"])
        source = tmp_path / "rejected.txt"
        source.write_text("should not be ingested", encoding="utf-8")

        with pytest.raises(ValueError, match="does not accept"):
            tool_registry.execute(
                "ingest_file", {"path": str(source), "collection_name": "pdf-only"}
            )

    def test_auto_routing_all(self, data_env, tmp_path):
        source = tmp_path / "auto.txt"
        source.write_text("auto-routed document content", encoding="utf-8")
        result = tool_registry.execute("ingest_file", {"path": str(source), "collection_name": "all"})
        # .txt routes to the documentation collection via the loader metadata.
        assert result["collection_name"] == "documentation"

    def test_missing_file_raises(self, data_env):
        with pytest.raises(ValueError, match="File not found"):
            tool_registry.execute("ingest_file", {"path": str(data_env / "ghost.txt")})


class TestDirectoryIngest:
    def test_background_job_completes(self, data_env, tmp_path):
        source_dir = tmp_path / "docs"
        source_dir.mkdir()
        (source_dir / "a.txt").write_text("alpha document about apples", encoding="utf-8")
        (source_dir / "b.txt").write_text("beta document about bananas", encoding="utf-8")

        start = tool_registry.execute(
            "start_ingest_directory", {"path": str(source_dir), "collection_name": "fruit"}
        )
        assert start["status"] in {"started", "queued"}
        status = wait_for_job(start["job_id"])
        assert status["status"] == "completed"
        assert status["succeeded_files"] == 2
        assert status["chunks_uploaded"] >= 2

        log = tool_registry.execute("get_ingest_log", {"job_id": start["job_id"]})
        assert "JOB COMPLETED" in log["log"]

    def test_get_ingest_log_uses_job_id_even_with_wrong_collection_hint(self, data_env, tmp_path):
        source_dir = tmp_path / "docs2"
        source_dir.mkdir()
        (source_dir / "a.txt").write_text("alpha", encoding="utf-8")

        start = tool_registry.execute(
            "start_ingest_directory", {"path": str(source_dir), "collection_name": "fruit"}
        )
        status = wait_for_job(start["job_id"])
        assert status["status"] == "completed"

        log = tool_registry.execute(
            "get_ingest_log",
            {
                "job_id": start["job_id"],
                "collection_name": "definitely-wrong-name",
            },
        )
        assert "JOB COMPLETED" in log["log"]

    def test_skips_files_not_allowed_by_type(self, data_env, tmp_path):
        runtime.collections().create("pdf-db", allowed_extensions=[".pdf"])
        source_dir = tmp_path / "mixed"
        source_dir.mkdir()
        (source_dir / "one.txt").write_text("text file one", encoding="utf-8")
        (source_dir / "two.txt").write_text("text file two", encoding="utf-8")

        start = tool_registry.execute(
            "start_ingest_directory", {"path": str(source_dir), "collection_name": "pdf-db"}
        )
        status = wait_for_job(start["job_id"])
        assert status["status"] == "completed"
        assert status["skipped_by_type"] == 2
        assert status["succeeded_files"] == 0

    def test_missing_directory_raises(self, data_env):
        with pytest.raises(ValueError, match="Directory not found"):
            tool_registry.execute("start_ingest_directory", {"path": str(data_env / "nowhere")})


class TestClearIndex:
    def test_requires_confirm(self, data_env, tmp_path):
        source = tmp_path / "keep.txt"
        source.write_text("document that stays", encoding="utf-8")
        tool_registry.execute("ingest_file", {"path": str(source), "collection_name": "keeper"})

        result = tool_registry.execute("clear_index", {"collection_name": "keeper"})
        assert result["status"] == "cancelled"
        listing = tool_registry.execute("list_sources", {"collection_name": "keeper"})
        assert listing["total_chunks"] > 0

    def test_clears_collection_with_confirm(self, data_env, tmp_path):
        source = tmp_path / "gone.txt"
        source.write_text("document that goes away", encoding="utf-8")
        tool_registry.execute("ingest_file", {"path": str(source), "collection_name": "wiped"})

        result = tool_registry.execute("clear_index", {"confirm": True, "collection_name": "wiped"})
        assert result["status"] == "success"
        assert result["deleted_chunks"] > 0
        listing = tool_registry.execute("list_sources", {"collection_name": "wiped"})
        assert listing["total_chunks"] == 0

    def test_clears_single_source(self, data_env, tmp_path):
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("first document content", encoding="utf-8")
        second.write_text("second document content", encoding="utf-8")
        tool_registry.execute("ingest_file", {"path": str(first), "collection_name": "partial"})
        tool_registry.execute("ingest_file", {"path": str(second), "collection_name": "partial"})

        result = tool_registry.execute(
            "clear_index",
            {"confirm": True, "collection_name": "partial", "source": str(first)},
        )
        assert result["status"] == "success"
        assert result["deleted_chunks"] > 0
        listing = tool_registry.execute("list_sources", {"collection_name": "partial"})
        assert listing["total_documents"] == 1

    def test_list_sources_is_compact(self, data_env, tmp_path):
        runtime.collections().create("bulk")
        for index in range(12):
            source = tmp_path / f"doc_{index}.txt"
            source.write_text(f"document {index}", encoding="utf-8")
            tool_registry.execute("ingest_file", {"path": str(source), "collection_name": "bulk"})

        listing = tool_registry.execute("list_sources", {"collection_name": "bulk", "max_sources": 5})
        assert listing["total_documents"] == 12
        assert len(listing["sources"]) == 5
        assert listing["sources_omitted"] == 7


class TestDocumentSupport:
    def test_reports_txt_ready(self, data_env):
        support = tool_registry.execute("get_document_support", {})
        assert ".txt" in support["supported_extensions"]
        assert support["formats"][".txt"]["available"] is True
