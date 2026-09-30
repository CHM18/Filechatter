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

        monkeypatch.setattr(
            dl,
            "_summarize_image_with_llm",
            lambda _path: "A scenic mountain view.\nSearchable keywords: mountain, alpine lake, hiking",
        )
        monkeypatch.setattr(dl, "_extract_image_exif", lambda _path: {"Make": "Canon"})

        loaded = dl.load_document(path)
        assert "A scenic mountain view." in loaded.content
        assert "Searchable keywords: mountain, alpine lake, hiking" in loaded.content
        assert loaded.metadata["content_kind"] == "image"
        assert loaded.metadata["image_summary"] == "A scenic mountain view."
        assert loaded.metadata["image_keywords"] == ["mountain", "alpine lake", "hiking"]
        assert loaded.metadata["image_exif"]["Make"] == "Canon"

    def test_vision_corrects_exif_orientation_without_changing_source(self, tmp_path, monkeypatch):
        import base64
        import io
        from PIL import Image

        path = tmp_path / "oriented.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (4, 2), "red").save(path, exif=exif)
        original = path.read_bytes()
        captured = {}

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "A red image."}}]}

        monkeypatch.setattr(dl.config, "IMAGE_VISION_BASE_URL", "http://localhost:1234")
        monkeypatch.setattr(dl.config, "IMAGE_VISION_MODEL", "test-model")
        monkeypatch.setattr(dl.requests, "post", lambda _url, **kwargs: captured.update(kwargs) or Response())

        assert dl._summarize_image_with_llm(path) == "A red image."

        payload = captured["json"]
        prompt = payload["messages"][0]["content"][0]["text"]
        assert "Searchable keywords:" in prompt
        encoded = payload["messages"][0]["content"][1]["image_url"]["url"].split(",", 1)[1]
        corrected_bytes = base64.b64decode(encoded)
        with Image.open(io.BytesIO(corrected_bytes)) as corrected:
            assert corrected.size == (2, 4)
            assert corrected.getexif().get(274) in (None, 1)
        assert path.read_bytes() == original

    @pytest.mark.parametrize("orientation", [None, 1])
    def test_vision_does_not_rotate_image_without_needed_correction(self, tmp_path, monkeypatch, orientation):
        import base64
        from PIL import Image

        path = tmp_path / "already-rotated.jpg"
        image = Image.new("RGB", (2, 4), "blue")
        if orientation is None:
            image.save(path)
        else:
            exif = Image.Exif()
            exif[274] = orientation
            image.save(path, exif=exif)
        original = path.read_bytes()
        captured = {}

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "A blue image."}}]}

        monkeypatch.setattr(dl.config, "IMAGE_VISION_BASE_URL", "http://localhost:1234")
        monkeypatch.setattr(dl.config, "IMAGE_VISION_MODEL", "test-model")
        monkeypatch.setattr(dl.requests, "post", lambda _url, **kwargs: captured.update(kwargs) or Response())

        dl._summarize_image_with_llm(path)

        encoded = captured["json"]["messages"][0]["content"][1]["image_url"]["url"].split(",", 1)[1]
        assert base64.b64decode(encoded) == original
        assert path.read_bytes() == original

    def test_unsupported_extension_rejected(self, tmp_path):
        path = tmp_path / "blob.bin"
        path.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="Unsupported"):
            dl.load_document(path)


class TestOCR:
    def test_is_ocr_available_reflects_module_and_binary(self, monkeypatch):
        monkeypatch.setattr(dl, "_has_module", lambda name: name == "pytesseract")
        monkeypatch.setattr(dl, "_has_tesseract_binary", lambda: True)
        assert dl.is_ocr_available() is True

        monkeypatch.setattr(dl, "_has_tesseract_binary", lambda: False)
        assert dl.is_ocr_available() is False

    def test_image_ocr_enabled_uses_ocr_instead_of_vision(self, tmp_path, monkeypatch):
        path = tmp_path / "bill.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")

        monkeypatch.setattr(dl, "is_ocr_available", lambda: True)
        monkeypatch.setattr(dl, "_open_image_rgb", lambda _path: object())
        monkeypatch.setattr(dl, "_ocr_pil_image", lambda _image: "Total: 42.00 EUR")

        def unexpected_call(_path):
            raise AssertionError("OCR image ingestion must not run photo analysis or EXIF extraction")

        monkeypatch.setattr(dl, "_summarize_image_with_llm", unexpected_call)
        monkeypatch.setattr(dl, "_extract_image_exif", unexpected_call)

        loaded = dl.load_document(path, ocr_enabled=True)
        assert "Total: 42.00 EUR" in loaded.content
        assert loaded.metadata["ocr_used"] is True
        assert loaded.metadata["extraction_method"] == "ocr"
        assert "image_exif" not in loaded.metadata

    def test_image_ocr_disabled_keeps_vision_pipeline(self, tmp_path, monkeypatch):
        path = tmp_path / "photo.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")

        monkeypatch.setattr(dl, "_summarize_image_with_llm", lambda _path: "A scenic mountain view.")
        monkeypatch.setattr(dl, "_extract_image_exif", lambda _path: {})

        loaded = dl.load_document(path, ocr_enabled=False)
        assert "A scenic mountain view." in loaded.content
        assert loaded.metadata["ocr_used"] is False

    def test_image_ocr_requested_but_unavailable_falls_back(self, tmp_path, monkeypatch):
        path = tmp_path / "photo.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")

        monkeypatch.setattr(dl, "is_ocr_available", lambda: False)
        monkeypatch.setattr(dl, "_summarize_image_with_llm", lambda _path: "A scenic mountain view.")
        monkeypatch.setattr(dl, "_extract_image_exif", lambda _path: {})

        loaded = dl.load_document(path, ocr_enabled=True)
        assert "A scenic mountain view." in loaded.content
        assert loaded.metadata["ocr_requested_but_unavailable"] is True
        assert loaded.metadata["ocr_used"] is False

    def test_image_ocr_and_vision_analysis_are_mutually_exclusive(self, tmp_path, monkeypatch):
        """OCR and the vision-LLM description are never both run for the same image."""
        path = tmp_path / "photo.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")

        vision_calls: list[Path] = []
        ocr_calls: list[object] = []
        monkeypatch.setattr(dl, "_summarize_image_with_llm", lambda p: vision_calls.append(p) or "A scenic mountain view.")
        monkeypatch.setattr(dl, "_extract_image_exif", lambda _path: {})
        monkeypatch.setattr(dl, "is_ocr_available", lambda: True)
        monkeypatch.setattr(dl, "_open_image_rgb", lambda _path: object())
        monkeypatch.setattr(dl, "_ocr_pil_image", lambda image: ocr_calls.append(image) or "Total: 42.00 EUR")

        # OCR enabled: only the OCR path should run.
        loaded_ocr = dl.load_document(path, ocr_enabled=True)
        assert ocr_calls and not vision_calls
        assert loaded_ocr.metadata["ocr_used"] is True
        assert "image_summary" not in loaded_ocr.metadata

        vision_calls.clear()
        ocr_calls.clear()

        # OCR disabled: only the vision-description path should run.
        loaded_vision = dl.load_document(path, ocr_enabled=False)
        assert vision_calls and not ocr_calls
        assert loaded_vision.metadata["ocr_used"] is False
        assert loaded_vision.metadata["content_kind"] == "image"
        assert "image_summary" in loaded_vision.metadata

    def _make_pdf_with_text(self, path: Path, text: str) -> None:
        import fitz

        document = fitz.open()
        try:
            page = document.new_page()
            page.insert_text((72, 72), text)
            document.save(str(path))
        finally:
            document.close()

    def test_scanned_pdf_triggers_ocr_fallback(self, tmp_path, monkeypatch):
        path = tmp_path / "scan.pdf"
        import fitz

        document = fitz.open()
        try:
            document.new_page()  # blank page: no extractable text layer
            document.save(str(path))
        finally:
            document.close()

        monkeypatch.setattr(dl, "is_ocr_available", lambda: True)
        monkeypatch.setattr(dl, "_ocr_image_bytes", lambda _data: "Invoice total: 99.90 EUR")

        loaded = dl.load_document(path, ocr_enabled=True)
        assert "Invoice total: 99.90 EUR" in loaded.content
        assert loaded.metadata["ocr_used"] is True
        assert loaded.metadata["extraction_method"] == "ocr"

    def test_text_pdf_skips_ocr(self, tmp_path, monkeypatch):
        path = tmp_path / "digital.pdf"
        self._make_pdf_with_text(path, "This invoice already has a real embedded text layer.")

        monkeypatch.setattr(dl, "is_ocr_available", lambda: True)
        ocr_calls: list[bytes] = []
        monkeypatch.setattr(dl, "_ocr_image_bytes", lambda data: ocr_calls.append(data) or "SHOULD NOT BE USED")

        loaded = dl.load_document(path, ocr_enabled=True)
        assert "This invoice already has a real embedded text layer." in loaded.content
        assert loaded.metadata["ocr_used"] is False
        assert not ocr_calls

    def test_pdf_text_page_with_background_image_ocrs_only_the_image(self, tmp_path, monkeypatch):
        import fitz
        import io
        from PIL import Image

        path = tmp_path / "mixed.pdf"
        document = fitz.open()
        try:
            page = document.new_page()
            page.insert_text((72, 72), "Real text layer on top of a background photo.")
            buffer = io.BytesIO()
            Image.new("RGB", (20, 20), color="white").save(buffer, format="PNG")
            page.insert_image(fitz.Rect(72, 200, 172, 300), stream=buffer.getvalue())
            document.save(str(path))
        finally:
            document.close()

        monkeypatch.setattr(dl, "is_ocr_available", lambda: True)
        monkeypatch.setattr(dl, "_ocr_image_bytes", lambda _data: "Hidden text in background photo")

        loaded = dl.load_document(path, ocr_enabled=True)
        assert "Real text layer on top of a background photo." in loaded.content
        assert "[Image OCR]: Hidden text in background photo" in loaded.content
        assert loaded.metadata["ocr_used"] is True
        assert loaded.metadata["extraction_method"] == "mixed"
        assert loaded.metadata["ocr_image_count"] == 1
        assert loaded.metadata["ocr_pages"] == 0

    def test_ocr_disabled_never_touches_ocr_path(self, tmp_path, monkeypatch):
        path = tmp_path / "scan.pdf"
        import fitz

        document = fitz.open()
        try:
            document.new_page()
            document.save(str(path))
        finally:
            document.close()

        def _boom(_data):
            raise AssertionError("OCR should not run when ocr_enabled=False")

        monkeypatch.setattr(dl, "is_ocr_available", lambda: True)
        monkeypatch.setattr(dl, "_ocr_image_bytes", _boom)

        loaded = dl.load_document(path, ocr_enabled=False)
        assert loaded.metadata["ocr_used"] is False

    def test_docx_embedded_image_ocr(self, tmp_path, monkeypatch):
        from docx import Document as DocxDocument
        import io
        from PIL import Image

        path = tmp_path / "letter.docx"
        document = DocxDocument()
        document.add_paragraph("Cover letter text.")
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10), color="white").save(buffer, format="PNG")
        buffer.seek(0)
        document.add_picture(buffer)
        document.save(str(path))

        monkeypatch.setattr(dl, "is_ocr_available", lambda: True)
        monkeypatch.setattr(dl, "_ocr_image_bytes", lambda _data: "Scanned signature block")

        loaded = dl.load_document(path, ocr_enabled=True)
        assert "Cover letter text." in loaded.content
        assert "[Image OCR]: Scanned signature block" in loaded.content
        assert loaded.metadata["ocr_used"] is True
        assert loaded.metadata["ocr_image_count"] == 1

    def test_pptx_embedded_image_ocr(self, tmp_path, monkeypatch):
        from pptx import Presentation
        from pptx.util import Inches
        import io
        from PIL import Image

        path = tmp_path / "deck.pptx"
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10), color="white").save(buffer, format="PNG")
        buffer.seek(0)
        slide.shapes.add_picture(buffer, Inches(1), Inches(1), Inches(2), Inches(2))
        prs.save(str(path))

        monkeypatch.setattr(dl, "is_ocr_available", lambda: True)
        monkeypatch.setattr(dl, "_ocr_image_bytes", lambda _data: "Receipt line items")

        loaded = dl.load_document(path, ocr_enabled=True)
        assert "[Image OCR]: Receipt line items" in loaded.content
        assert loaded.metadata["ocr_used"] is True
        assert loaded.metadata["ocr_image_count"] == 1

    def test_xlsx_embedded_image_ocr(self, tmp_path, monkeypatch):
        import openpyxl
        from openpyxl.drawing.image import Image as XlImage
        import io
        from PIL import Image

        path = tmp_path / "book.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Name", "Score"])
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10), color="white").save(buffer, format="PNG")
        buffer.seek(0)
        xl_image = XlImage(buffer)
        ws.add_image(xl_image, "A3")
        wb.save(str(path))

        monkeypatch.setattr(dl, "is_ocr_available", lambda: True)
        monkeypatch.setattr(dl, "_ocr_image_bytes", lambda _data: "Table receipt text")

        loaded = dl.load_document(path, ocr_enabled=True)
        assert "[Image OCR" in loaded.content
        assert "Table receipt text" in loaded.content
        assert loaded.metadata["ocr_used"] is True
        assert loaded.metadata["ocr_image_count"] == 1
