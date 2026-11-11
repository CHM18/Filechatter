"""Unit tests for the shared tool registry (search, ingest, clear, jobs)."""
from __future__ import annotations

from time import sleep, time

import pytest

import document_loader
import ingest_jobs
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
            "read_document",
            "list_sources",
            "get_document_support",
            "get_ingest_status",
            "get_ingest_log",
            "ingest_file",
            "start_ingest_directory",
            "clear_index",
            "create_collection",
            "delete_collection",
        }
        read_tools = {name for name, spec in specs.items() if spec.access == "read"}
        write_tools = {name for name, spec in specs.items() if spec.access == "write"}
        assert write_tools == {
            "ingest_file", "start_ingest_directory", "clear_index",
            "create_collection", "delete_collection",
        }
        assert read_tools == set(specs) - write_tools

    def test_every_tool_has_params_schema(self):
        for spec in tool_registry.list_specs():
            assert spec.params_schema["type"] == "object"
            assert "properties" in spec.params_schema

    def test_execute_unknown_tool_raises(self, data_env):
        with pytest.raises(KeyError):
            tool_registry.execute("does_not_exist", {})


class TestIngestFile:
    def test_scanned_pdf_produces_same_chunks_for_file_and_directory(self, data_env, tmp_path, monkeypatch):
        import fitz

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source = source_dir / "scan.pdf"
        document = fitz.open()
        try:
            document.new_page()
            document.save(str(source))
        finally:
            document.close()
        monkeypatch.setattr(document_loader, "is_ocr_available", lambda: True)
        monkeypatch.setattr(document_loader, "_ocr_image_bytes", lambda _data: "OCR extracted body text")
        for collection in ("single", "directory"):
            runtime.collections().create(collection, ocr_enabled=True)

        result = tool_registry.ingest_file(str(source), "single")
        start = tool_registry.start_ingest_directory(str(source_dir), collection_name="directory")
        status = wait_for_job(start["job_id"])
        single = runtime.collections().get_store("single").get_chunks(source=str(source.resolve()))
        directory = runtime.collections().get_store("directory").get_chunks(source=str(source.resolve()))

        assert result["documents_uploaded"] == 1
        assert status["failed_files"] == 0
        assert [chunk["content"] for chunk in single] == [chunk["content"] for chunk in directory]
        assert "OCR extracted body text" in single[0]["content"]
        assert single[0]["metadata"]["ocr_used"] is True
        assert directory[0]["metadata"]["ocr_used"] is True

    @pytest.mark.parametrize("collection_name", ["scans", "all"])
    def test_single_file_and_directory_use_target_collection_ocr(self, data_env, tmp_path, monkeypatch, collection_name):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source = source_dir / "note.txt"
        source.write_text("shared extraction content", encoding="utf-8")
        target = "documentation" if collection_name == "all" else collection_name
        runtime.collections().ensure(target)
        runtime.collections().update_entry(target, ocr_enabled=True)
        calls = []
        original_loader = tool_registry.load_document

        def capture(path, *, ocr_enabled=False):
            calls.append(ocr_enabled)
            return original_loader(path, ocr_enabled=ocr_enabled)

        monkeypatch.setattr(tool_registry, "load_document", capture)
        monkeypatch.setattr(ingest_jobs, "load_document", capture)
        result = tool_registry.ingest_file(str(source), collection_name)
        start = tool_registry.start_ingest_directory(str(source_dir), collection_name=collection_name)
        status = wait_for_job(start["job_id"])

        assert result["documents_uploaded"] == 1
        assert status["failed_files"] == 0
        assert calls == [True, True]

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

    def test_scoped_search_and_paginated_document_read(self, data_env):
        runtime.collections().create("manuals")
        store = runtime.collections().selected_stores("manuals")[0]
        store.add_documents(
            ["Unrelated template content.", "Development output is verified.", "Approval records are retained."],
            [
                {"source": "forms/template.pdf", "filename": "template.pdf"},
                {"source": "SOP 7300.pdf", "filename": "SOP 7300.pdf"},
                {"source": "SOP 7300.pdf", "filename": "SOP 7300.pdf"},
            ],
        )
        store._semantic_candidates = lambda _query, source=None: []

        hits = tool_registry.execute(
            "search_documents",
            {"query": "Development output", "collection_name": "manuals", "source": "SOP 7300.pdf"},
        )
        assert hits
        assert {hit["source"] for hit in hits} == {"SOP 7300.pdf"}

        first = tool_registry.execute(
            "read_document", {"source": "SOP 7300.pdf", "collection_name": "manuals", "limit": 1}
        )
        second = tool_registry.execute(
            "read_document",
            {
                "source": "SOP 7300.pdf",
                "collection_name": "manuals",
                "offset": first["next_offset"],
                "limit": 1,
            },
        )
        assert first["total_chunks"] == 2
        assert first["chunks"][0]["content"] == "Development output is verified."
        assert first["has_more"] is True
        assert second["chunks"][0]["content"] == "Approval records are retained."
        assert second["has_more"] is False

        runtime.collections().create("archive")
        runtime.collections().selected_stores("archive")[0].add_documents(
            ["Archived copy."], [{"source": "SOP 7300.pdf", "filename": "SOP 7300.pdf"}]
        )
        ambiguous = tool_registry.execute("read_document", {"source": "SOP 7300.pdf"})
        assert ambiguous["status"] == "ambiguous"
        assert {item["collection_name"] for item in ambiguous["matches"]} == {"archive", "manuals"}

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
