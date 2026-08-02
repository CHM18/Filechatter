"""Tests for document_loader: new readers and the file-type catalog."""
from __future__ import annotations

from pathlib import Path

import pytest

import document_loader as dl


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestTextLikeReaders:
    @pytest.mark.parametrize("name", ["notes.md", "script.py", "data.csv", "config.yaml", "page.xml"])
    def test_reads_as_text(self, tmp_path, name):
        path = write(tmp_path, name, "hello world content")
        loaded = dl.load_document(path)
        assert "hello world content" in loaded.content

    def test_markdown_routes_to_documentation(self, tmp_path):
        path = write(tmp_path, "readme.md", "# Title")
        assert dl.load_document(path).metadata["collection_name"] == "documentation"

    def test_code_routes_to_code(self, tmp_path):
        path = write(tmp_path, "main.py", "print('hi')")
        assert dl.load_document(path).metadata["collection_name"] == "code"

    def test_html_strips_tags(self, tmp_path):
        path = write(tmp_path, "page.html",
                     "<html><head><style>x{}</style></head><body><h1>Hi</h1><p>There &amp; more</p></body></html>")
        content = dl.load_document(path).content
        assert "Hi" in content and "There & more" in content
        assert "<h1>" not in content
        assert "x{}" not in content  # style stripped


class TestOfficeReaders:
    def test_reads_xlsx(self, tmp_path):
        import openpyxl

        path = tmp_path / "book.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Data"
        ws.append(["Name", "Score"])
        ws.append(["Ada", 99])
        wb.save(str(path))

        content = dl.load_document(path).content
        assert "Sheet: Data" in content
        assert "Name | Score" in content
        assert "Ada" in content and "99" in content

    def test_reads_pptx(self, tmp_path):
        from pptx import Presentation
        from pptx.util import Inches

        path = tmp_path / "deck.pptx"
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
        box.text_frame.text = "Quarterly results summary"
        prs.save(str(path))

        content = dl.load_document(path).content
        assert "Slide 1" in content
        assert "Quarterly results summary" in content


class TestSupportAndCatalog:
    def test_supported_extensions_expanded(self):
        for ext in [".md", ".py", ".html", ".csv", ".xlsx", ".pptx", ".json"]:
            assert ext in dl.SUPPORTED_EXTENSIONS

    def test_catalog_has_categories(self):
        catalog = dl.file_type_catalog()
        names = {c["category"] for c in catalog}
        assert {"Office", "Images", "Text", "Web", "Code"}.issubset(names)

    def test_catalog_reports_availability(self):
        catalog = {c["category"]: c for c in dl.file_type_catalog()}
        office = {e["ext"]: e for e in catalog["Office"]["extensions"]}
        assert office[".xlsx"]["available"] is True  # openpyxl installed in test env
        text = {e["ext"]: e for e in catalog["Text"]["extensions"]}
        assert text[".md"]["available"] is True
        assert text[".md"]["dependency"] is None

    def test_support_status_marks_text_ready(self):
        support = dl.get_document_support_status()
        assert support[".md"]["available"] is True
        assert support[".py"]["category"] == "Code"

    def test_image_uses_vision_and_exif_pipeline(self, tmp_path, monkeypatch):
        path = tmp_path / "image.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")

        monkeypatch.setattr(dl, "_summarize_image_with_llm", lambda _path: "A scenic mountain view.")
        monkeypatch.setattr(dl, "_extract_image_exif", lambda _path: {"Make": "Canon"})

        loaded = dl.load_document(path)
        assert "A scenic mountain view." in loaded.content
        assert loaded.metadata["content_kind"] == "image"
        assert loaded.metadata["image_exif"]["Make"] == "Canon"

    def test_unsupported_extension_rejected(self, tmp_path):
        path = tmp_path / "blob.bin"
        path.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="Unsupported"):
            dl.load_document(path)
