"""Integration tests with the sibling private audit client and real HTTP API."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import fitz
import pytest

import document_loader
import runtime
import structured_extraction
import tool_registry
from test_tool_registry import wait_for_job


AUDIT_ROOT = Path(__file__).resolve().parents[2] / "snke-qm-rag"
if not (AUDIT_ROOT / "dhf_audit").is_dir():
    pytest.skip("Three-path integration requires the sibling snke-qm-rag repository", allow_module_level=True)
sys.path.insert(0, str(AUDIT_ROOT))

from dhf_audit.client import FilechatterClient
from dhf_audit import pipeline as audit_pipeline
from dhf_audit.pipeline import audit


def write_pdf(path, texts):
    with fitz.open() as document:
        for text in texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(str(path))


def write_office(path, text="Complete Office body references 0000413991"):
    if path.suffix == ".docx":
        from docx import Document

        document = Document()
        document.add_paragraph(text)
        document.save(str(path))
    elif path.suffix == ".xlsx":
        from openpyxl import Workbook

        workbook = Workbook()
        workbook.active.append([text])
        workbook.save(path)
        workbook.close()
    else:
        from pptx import Presentation
        from pptx.util import Inches

        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(1)).text = text
        presentation.save(str(path))


def audit_client(client):
    return FilechatterClient("http://testserver", confirm=True, session=client)


@pytest.mark.parametrize("kind", [
    "digital", "scanned", "mixed", "fallback-docx", "fallback-xlsx", "fallback-pptx",
    "office-docx", "office-xlsx", "office-pptx",
])
def test_identical_source_chunks_across_all_three_paths(client, tmp_path, monkeypatch, kind):
    folder = tmp_path / "sources"
    folder.mkdir()
    source = folder / "Main.pdf"
    ocr = kind in {"scanned", "mixed"}
    texts = ["Record Number: 0000402357\nDigital evidence with a usable text layer"]
    if kind == "scanned":
        texts = [""]
    elif kind == "mixed":
        texts.append("")
    elif kind.startswith("fallback"):
        texts = ["APPROVAL INFORMATION\nRecord Number: 0000402357", ""]
        write_office(folder / ("Main_ATT1." + kind.split("-")[1]))
    write_pdf(source, texts)
    if kind.startswith("office"):
        source = folder / ("Supplement_ATT1." + kind.split("-")[1])
        write_office(source)
    monkeypatch.setattr(document_loader, "is_ocr_available", lambda: True)
    monkeypatch.setattr(document_loader, "_ocr_image_bytes", lambda _data: "OCR body evidence 0000413991")
    for name in ("single", "directory", "audited"):
        runtime.collections().create(name, ocr_enabled=ocr)

    tool_registry.ingest_file(str(source), "single")
    job = tool_registry.start_ingest_directory(str(folder), collection_name="directory")
    assert wait_for_job(job["job_id"])["failed_files"] == 0
    package, _ = audit(folder, collection="audited", ingest=True, client=audit_client(client))
    assert package["status"] == "ingested", package
    chunks = [runtime.collections().get_store(name).get_chunks(str(source.resolve()))
              for name in ("single", "directory", "audited")]
    assert chunks[0]
    assert [chunk["content"] for chunk in chunks[0]] == [chunk["content"] for chunk in chunks[1]]
    assert [chunk["content"] for chunk in chunks[0]] == [chunk["content"] for chunk in chunks[2]]
    fingerprints = [chunk_list[0]["metadata"]["extraction_fingerprint"] for chunk_list in chunks]
    assert len(set(fingerprints)) == 1
    if kind.startswith("fallback"):
        assert "Complete Office body" in "\n".join(chunk["content"] for chunk in chunks[0])
        assert chunks[2][0]["metadata"]["body_fallback"]["retained_pdf_pages"] == [1]
    if ocr:
        assert chunks[2][0]["metadata"]["ocr_used"] is True
        assert any("ocr_text_requires_review" in warning
                   for warning in package["documents"][0]["metadata"]["warnings"])


@pytest.mark.parametrize("failure", ["blank", "ambiguous", "ocr-failed", "ocr-empty", "ocr-unavailable"])
def test_failures_never_index_partial_source_in_any_path(client, tmp_path, monkeypatch, failure):
    folder = tmp_path / "sources"
    folder.mkdir()
    source = folder / "Main.pdf"
    write_pdf(source, [""])
    if failure == "ambiguous":
        write_office(folder / "Main_ATT1.docx")
        write_office(folder / "Main_ATT2.docx")
    ocr = failure.startswith("ocr")
    monkeypatch.setattr(document_loader, "is_ocr_available", lambda: failure != "ocr-unavailable")

    def recognize(_data):
        if failure == "ocr-failed":
            raise RuntimeError("OCR backend failed")
        return ""

    monkeypatch.setattr(document_loader, "_ocr_image_bytes", recognize)
    for name in ("single", "directory", "audited"):
        runtime.collections().create(name, ocr_enabled=ocr)
    with pytest.raises((ValueError, RuntimeError)):
        tool_registry.ingest_file(str(source), "single")
    job = tool_registry.start_ingest_directory(str(folder), collection_name="directory")
    assert wait_for_job(job["job_id"])["failed_files"] >= 1
    package, _ = audit(folder, collection="audited", ingest=True, client=audit_client(client))
    assert package["status"] == "ingestion_failed"
    for name in ("single", "directory", "audited"):
        assert runtime.collections().get_store(name).get_chunks(str(source.resolve())) == []


def test_companion_hash_and_ocr_settings_change_snapshot_and_reindex(client, tmp_path, monkeypatch):
    folder = tmp_path / "sources"
    folder.mkdir()
    source = folder / "Main.pdf"
    write_pdf(source, [""])
    companion = folder / "Main_ATT1.docx"
    write_office(companion, "Original Office evidence")
    runtime.collections().create("audited")
    backend = audit_client(client)
    first, _ = audit(folder, collection="audited", ingest=True, client=backend)
    assert first["status"] == "ingested"
    retry, _ = audit(folder, collection="audited", ingest=True, client=backend)
    assert retry["status"] == "ingested", retry
    assert retry["snapshot_id"] == first["snapshot_id"]

    before = document_loader.load_document(source)
    write_office(companion, "Changed Office evidence")
    after = document_loader.load_document(source)
    assert before.metadata["extraction_fingerprint"] != after.metadata["extraction_fingerprint"]
    changed, _ = audit(folder, collection="audited", ingest=True, client=backend)
    assert changed["snapshot_id"] != first["snapshot_id"]
    assert changed["status"] == "ingestion_failed"
    assert "another DHF snapshot" in changed["error"]

    monkeypatch.setattr(document_loader, "is_ocr_available", lambda: True)
    monkeypatch.setattr(document_loader, "_ocr_image_bytes", lambda _data: "Recovered OCR evidence")
    runtime.collections().update_entry("audited", ocr_enabled=True)
    settings, _ = audit(folder, collection="audited", ingest=True, client=backend)
    assert settings["snapshot_id"] != changed["snapshot_id"]
    assert settings["status"] == "ingestion_failed"

    runtime.collections().create("single")
    tool_registry.ingest_file(str(source), "single")
    write_office(companion, "Newer companion while PDF timestamp stays unchanged")
    result = tool_registry.ingest_file(str(source), "single")
    assert result["replaced_documents"] == 1


def test_extraction_rejects_changed_companion_during_read(tmp_path, monkeypatch):
    source = tmp_path / "Main.pdf"
    write_pdf(source, [""])
    companion = tmp_path / "Main_ATT1.docx"
    write_office(companion)
    original = structured_extraction.read_office

    def mutate(path):
        units = original(path)
        write_office(path, "Changed during extraction")
        return units

    monkeypatch.setattr(structured_extraction, "read_office", mutate)
    with pytest.raises(ValueError, match="changed during extraction"):
        document_loader.load_document(source)


def test_extraction_api_permissions_settings_hash_and_no_index_side_effect(client, tmp_path, monkeypatch):
    source = tmp_path / "Main.pdf"
    write_pdf(source, ["Digital evidence with a real usable text layer"])
    runtime.collections().create("audit")
    payload = {"path": str(source), "collection_name": "audit",
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "confirm": True}
    response = client.post("/documents/extract", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["pdf_pages"][0]["page"] == 1
    assert runtime.collections().get_store("audit").chunk_count == 0
    assert client.post("/documents/extract", json={**payload, "source_sha256": "stale"}).status_code == 400
    runtime.collections().update_entry("audit", write_allowed=False)
    assert client.post("/documents/extract", json=payload).status_code == 403
    runtime.collections().update_entry("audit", write_allowed=True, read_allowed=False)
    assert client.post("/documents/extract", json=payload).status_code == 403
    runtime.collections().update_entry("audit", read_allowed=True)
    runtime.collections().update_entry("audit", write_allowed=True, allowed_extensions=[".txt"])
    assert client.post("/documents/extract", json=payload).status_code == 400
    runtime.collections().update_entry("audit", allowed_extensions=[".pdf"])
    monkeypatch.setattr("permissions.resolve", lambda *_args, **_kwargs: "ask")
    assert client.post("/documents/extract", json={**payload, "confirm": False}).status_code == 403
    monkeypatch.setattr("permissions.resolve", lambda *_args, **_kwargs: "deny")
    assert client.post("/documents/extract", json=payload).status_code == 403


@pytest.mark.parametrize("fallback", [False, True])
def test_ingesting_audit_never_calls_legacy_extractors(client, tmp_path, monkeypatch, fallback):
    source = tmp_path / "Main.pdf"
    write_pdf(source, [""] if fallback else ["A digital document with enough text to index"])
    if fallback:
        write_office(tmp_path / "Main_ATT1.docx")
    runtime.collections().create("audited")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Ingestion must not call an audit-local extractor")

    for name in ("read_pdf", "read_office", "prepare_body"):
        monkeypatch.setattr(audit_pipeline, name, forbidden)
    package, _ = audit(tmp_path, collection="audited", ingest=True, client=audit_client(client))
    assert package["status"] == "ingested", package


@pytest.mark.parametrize("change", ["companion", "added-companion", "ocr-setting"])
def test_changes_between_extraction_and_upload_are_withheld(client, tmp_path, monkeypatch, change):
    source = tmp_path / "Main.pdf"
    write_pdf(source, [""])
    companion = tmp_path / "Main_ATT1.docx"
    write_office(companion)
    runtime.collections().create("audited")
    backend = audit_client(client)
    original = backend.extract_document

    def extract(name, path, sha256):
        result = original(name, path, sha256)
        if path == str(source):
            if change == "companion":
                write_office(companion, "Changed after extraction")
            elif change == "added-companion":
                write_office(tmp_path / "Main_ATT2.docx")
            else:
                runtime.collections().update_entry("audited", ocr_enabled=True)
        return result

    monkeypatch.setattr(backend, "extract_document", extract)
    package, _ = audit(tmp_path, collection="audited", ingest=True, client=backend)
    assert package["status"] == "ingestion_failed"
    assert runtime.collections().get_store("audited").get_chunks(str(source)) == []


def test_unreadable_retained_cover_is_not_repaired_by_office_body(tmp_path, monkeypatch):
    source = tmp_path / "Main.pdf"
    write_pdf(source, ["APPROVAL INFORMATION", "Complete digital body evidence"])
    write_office(tmp_path / "Main_ATT1.docx")
    original = document_loader.read_pdf_units

    def damaged_cover(path):
        units = original(path)
        units[0]["error"] = "Cover extraction failed"
        return units

    monkeypatch.setattr(document_loader, "read_pdf_units", damaged_cover)
    with pytest.raises(ValueError, match="unassessed/failed text units.*PDF page 1"):
        document_loader.load_document(source)


def test_companion_addition_during_extraction_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "Main.pdf"
    write_pdf(source, [""])
    write_office(tmp_path / "Main_ATT1.docx")
    original = structured_extraction.read_office

    def add_companion(path):
        units = original(path)
        write_office(tmp_path / "Main_ATT2.docx")
        return units

    monkeypatch.setattr(structured_extraction, "read_office", add_companion)
    with pytest.raises(RuntimeError, match="companions changed during extraction"):
        document_loader.load_document(source)
