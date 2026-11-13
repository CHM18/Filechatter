"""Location-preserving extraction and local Windchill companion policy."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import re
from threading import get_ident
from typing import Any


EXTRACTION_VERSION = 2
OFFICE_EXTENSIONS = {".docx", ".xlsx", ".pptx"}
ATT_PATTERN = re.compile(r"^(?P<name>.+)_ATT.*$", re.IGNORECASE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _PageWarnings(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.thread = get_ident()
        self.messages: list[str] = []

    def filter(self, record: logging.LogRecord) -> bool:
        if record.thread == self.thread and record.levelno >= logging.WARNING:
            self.messages.append(record.getMessage())
        return True


def read_pdf_units(path: Path) -> list[dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError(f"Encrypted PDF requires a password: {path}")
    units = []
    for number, page in enumerate(reader.pages, 1):
        unit: dict[str, Any] = {
            "page": number, "source_path": str(path), "text": "", "links": [],
            "extraction_method": "pypdf_layout",
        }
        logger = logging.getLogger("pypdf._text_extraction._layout_mode._fixed_width_page")
        capture = _PageWarnings()
        logger.addFilter(capture)
        try:
            unit["text"] = page.extract_text(extraction_mode="layout") or ""
        except Exception as exc:
            unit["error"] = str(exc)
        finally:
            logger.removeFilter(capture)
        unit["extraction_warnings"] = list(dict.fromkeys(capture.messages))
        for annotation in page.get("/Annots", []):
            try:
                annotation = annotation.get_object()
                action = annotation.get("/A")
                if action:
                    uri = action.get_object().get("/URI")
                    if uri:
                        unit["links"].append({
                            "url": str(uri), "evidence": str(annotation.get("/Contents") or uri),
                            "rectangle": [float(value) for value in annotation.get("/Rect", [])],
                        })
            except Exception as exc:
                unit["error"] = f"Hyperlink extraction failed: {exc}"
        units.append(unit)
    return units


def cover_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not pages or not re.search(r"^\s*APPROVAL INFORMATION\s*$", pages[0]["text"],
                                  re.IGNORECASE | re.MULTILINE):
        return []
    cover = pages[:1]
    if len(pages) > 1:
        second = pages[1]
        if not re.search(r"Last released\s*:", second["text"], re.IGNORECASE) and re.search(
                r"APPROVAL INFORMATION|Approved by\s+Approved on\s+Role|"
                r"Central European Time\s*\(CET\)|Format\s+dd\.mm\.yyyy",
                second["text"], re.IGNORECASE):
            cover = pages[:2]
    return cover


def needs_fallback(units: list[dict[str, Any]]) -> bool:
    return not units or any(unit.get("error") or unit.get("extraction_warnings")
                            or not unit["text"].strip() for unit in units)


def companion_paths(path: Path) -> list[Path]:
    attachment = ATT_PATTERN.match(path.stem)
    return sorted(
        other for other in path.parent.iterdir()
        if other.is_file() and not other.is_symlink() and other.suffix.lower() in OFFICE_EXTENSIONS
        and (match := ATT_PATTERN.match(other.stem))
        and (other.stem.casefold() == path.stem.casefold() if attachment else
             match.group("name").casefold() == path.stem.casefold())
    )


def read_office(path: Path) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []

    def append(location: str, text: str, links: list[dict[str, str]] | None = None) -> None:
        if text.strip():
            units.append({"page": None, "source_path": str(path), "source_location": location,
                          "text": text, "links": links or [], "extraction_method": "office_attachment"})

    if path.suffix.lower() == ".docx":
        from docx import Document
        from docx.table import Table

        def read_blocks(container: Any, prefix: str) -> None:
            for index, block in enumerate(container.iter_inner_content(), 1):
                if isinstance(block, Table):
                    seen_cells = set()
                    for row_number, row in enumerate(block.rows, 1):
                        for column, cell in enumerate(row.cells, 1):
                            if cell._tc in seen_cells:
                                continue
                            seen_cells.add(cell._tc)
                            read_blocks(cell, f"{prefix} table {index} row {row_number} cell {column}")
                else:
                    append(f"{prefix} paragraph {index}", block.text,
                           [{"url": link.url, "evidence": link.text} for link in block.hyperlinks if link.url])

        read_blocks(Document(str(path)), "DOCX")
    elif path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(path, data_only=False)
        try:
            for sheet in workbook:
                for row in sheet.iter_rows():
                    values = []
                    links = []
                    for cell in row:
                        if cell.value is not None:
                            values.append(f"{cell.coordinate}: {cell.value}")
                        if cell.hyperlink and cell.hyperlink.target:
                            links.append({"url": cell.hyperlink.target,
                                          "evidence": str(cell.value or cell.coordinate)})
                    append(f"XLSX worksheet {sheet.title!r} row {row[0].row}" if values else "",
                           "\t".join(values), links)
        finally:
            workbook.close()
    elif path.suffix.lower() == ".pptx":
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        def read_shapes(shapes: Any, prefix: str) -> None:
            for index, shape in enumerate(shapes, 1):
                location = f"{prefix} shape {index}"
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    read_shapes(shape.shapes, location)
                elif shape.has_table:
                    for row_number, row in enumerate(shape.table.rows, 1):
                        append(f"{location} table row {row_number}",
                               "\t".join(f"cell {column}: {cell.text}"
                                         for column, cell in enumerate(row.cells, 1)))
                elif shape.has_text_frame:
                    append(location, shape.text,
                           [{"url": run.hyperlink.address, "evidence": run.text}
                            for paragraph in shape.text_frame.paragraphs for run in paragraph.runs
                            if run.hyperlink.address])
                elif getattr(shape, "has_chart", False):
                    raise ValueError(f"{location}: chart extraction is not supported for body fallback")

        for number, slide in enumerate(Presentation(str(path)).slides, 1):
            read_shapes(slide.shapes, f"PPTX slide {number}")
    else:
        raise ValueError(f"Unsupported Office fallback format: {path.suffix}")
    if not units:
        raise ValueError(f"No extractable body text in Office attachment: {path}")
    return units


def replace_pdf_body(path: Path, pages: list[dict[str, Any]]) -> tuple[
        list[dict[str, Any]], dict[str, Any] | None, list[dict[str, str]]]:
    candidates = companion_paths(path)
    dependencies = [{"path": str(other), "sha256": file_sha256(other)} for other in candidates]
    cover = cover_pages(pages)
    if not needs_fallback(pages) or (cover and not needs_fallback(pages[len(cover):])) or not candidates:
        return pages, None, dependencies
    if len(candidates) != 1:
        raise ValueError("Ambiguous Office body fallback: " + ", ".join(map(str, candidates)))
    attachment = candidates[0]
    units = read_office(attachment)
    if file_sha256(attachment) != dependencies[0]["sha256"]:
        raise ValueError(f"Office attachment changed during extraction: {attachment}")
    details = {
        "status": "used", "strategy": "replace_entire_body",
        "cover_policy": "retain_recognized_cover_pages_only; unreadable_cover_evidence_cannot_be_recovered",
        "attachment_path": str(attachment), "attachment_sha256": dependencies[0]["sha256"],
        "retained_pdf_pages": [page["page"] for page in cover],
        "replaced_pdf_pages": [page["page"] for page in pages[len(cover):]],
        "attachment_locations": [unit["source_location"] for unit in units],
    }
    return cover + units, details, dependencies


def render_units(units: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{unit['source_location']}]\n{unit['text']}" if unit.get("source_location") else
        f"[PDF page {unit['page']}]\n{unit['text']}" for unit in units)
