"""Helpers for loading supported document types into plain text.

Readers fall into three groups:
- rich formats with dedicated parsers: .pdf, .docx, .doc, .xlsx, .pptx
- markup that is stripped to text: .html/.htm
- everything else (plain text, markdown, code, config) read as UTF-8 text

Extensions are organized into categories (Office / Text / Web / Data & Config
/ Code) that the web UI uses to offer grouped, then individual, selection.
"""
from __future__ import annotations

import html as html_module
import importlib.util
import logging
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


# ----------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _read_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = _SCRIPT_STYLE_RE.sub(" ", raw)
    text = html_module.unescape(_TAG_RE.sub(" ", raw))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Reading .pdf files requires the optional dependency 'pypdf'.") from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise RuntimeError("Reading .docx files requires the optional dependency 'python-docx'.") from exc
    document = DocxDocument(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))
    return "\n".join(paragraphs)


def _read_doc(path: Path) -> str:
    try:
        import win32com.client  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Reading .doc files requires pywin32 and Microsoft Word on Windows.") from exc
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(path.resolve()))
        return document.Content.Text
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()


def _read_xlsx(path: Path) -> str:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("Reading .xlsx files requires the optional dependency 'openpyxl'.") from exc
    workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    parts: list[str] = []
    try:
        for worksheet in workbook.worksheets:
            parts.append(f"# Sheet: {worksheet.title}")
            for row in worksheet.iter_rows(values_only=True):
                cells = [str(cell) for cell in row if cell is not None]
                if cells:
                    parts.append(" | ".join(cells))
    finally:
        workbook.close()
    return "\n".join(parts)


def _read_pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError("Reading .pptx files requires the optional dependency 'python-pptx'.") from exc
    presentation = Presentation(str(path))
    parts: list[str] = []
    for index, slide in enumerate(presentation.slides, start=1):
        parts.append(f"# Slide {index}")
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in paragraph.runs)
                    if text.strip():
                        parts.append(text)
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Categories and reader registry
# ----------------------------------------------------------------------

FILE_TYPE_CATEGORIES: dict[str, list[str]] = {
    "Office": [".pdf", ".docx", ".doc", ".xlsx", ".pptx"],
    "Text": [".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".log"],
    "Web": [".html", ".htm", ".xml", ".css", ".scss", ".vue", ".svelte"],
    "Data & Config": [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"],
    "Code": [
        ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".c", ".cc",
        ".cpp", ".cxx", ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh",
        ".bash", ".zsh", ".ps1", ".psm1", ".sql", ".swift", ".kt", ".kts", ".dart",
        ".scala", ".lua", ".pl", ".pm", ".r", ".groovy", ".gradle",
    ],
}

# Dedicated parsers; everything else in the catalog is read as plain text.
_SPECIAL_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".doc": _read_doc,
    ".xlsx": _read_xlsx,
    ".pptx": _read_pptx,
    ".html": _read_html,
    ".htm": _read_html,
}

READERS: dict[str, Any] = {}
for _category, _extensions in FILE_TYPE_CATEGORIES.items():
    for _extension in _extensions:
        READERS[_extension] = _SPECIAL_READERS.get(_extension, _read_text)

SUPPORTED_EXTENSIONS = set(READERS)

# Optional-dependency requirements per extension (absent => always available).
_OPTIONAL_DEPENDENCIES: dict[str, tuple[str, str]] = {
    ".pdf": ("pypdf", "pypdf"),
    ".docx": ("python-docx", "docx"),
    ".doc": ("pywin32 + Microsoft Word (Windows)", "win32com"),
    ".xlsx": ("openpyxl", "openpyxl"),
    ".pptx": ("python-pptx", "pptx"),
}

CODE_EXTENSIONS = set(FILE_TYPE_CATEGORIES["Code"]) | set(FILE_TYPE_CATEGORIES["Data & Config"])
CODE_FILENAMES = {"dockerfile", "makefile"}


@dataclass(slots=True)
class LoadedDocument:
    path: Path
    content: str
    metadata: dict[str, Any]


def _extension_available(extension: str) -> bool:
    dependency = _OPTIONAL_DEPENDENCIES.get(extension)
    if dependency is None:
        return True
    if extension == ".doc":
        return sys.platform.startswith("win") and _has_module("win32com")
    return _has_module(dependency[1])


def get_document_support_status() -> dict[str, dict[str, Any]]:
    """Per-extension readiness (used by the get_document_support tool)."""
    category_of = {ext: cat for cat, exts in FILE_TYPE_CATEGORIES.items() for ext in exts}
    support: dict[str, dict[str, Any]] = {}
    for extension in sorted(SUPPORTED_EXTENSIONS):
        dependency = _OPTIONAL_DEPENDENCIES.get(extension)
        available = _extension_available(extension)
        support[extension] = {
            "available": available,
            "dependency": dependency[0] if dependency else None,
            "category": category_of.get(extension, "Other"),
            "message": (
                "Ready."
                if available
                else f"Requires the optional dependency '{dependency[0]}'."
                if dependency
                else "Ready."
            ),
            "status": "ready" if available else "missing_dependency",
        }
    return support


def file_type_catalog() -> list[dict[str, Any]]:
    """Categories with per-extension availability, for the UI selector."""
    catalog: list[dict[str, Any]] = []
    for category, extensions in FILE_TYPE_CATEGORIES.items():
        catalog.append(
            {
                "category": category,
                "extensions": [
                    {
                        "ext": extension,
                        "available": _extension_available(extension),
                        "dependency": (
                            _OPTIONAL_DEPENDENCIES[extension][0]
                            if extension in _OPTIONAL_DEPENDENCIES
                            else None
                        ),
                    }
                    for extension in extensions
                ],
            }
        )
    return catalog


def is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def infer_collection_name(path: Path) -> str:
    resolved_path = path.expanduser().resolve()
    name = resolved_path.name.lower()
    if name in CODE_FILENAMES or resolved_path.suffix.lower() in CODE_EXTENSIONS:
        return "code"
    return "documentation"


def load_document(path: Path) -> LoadedDocument:
    resolved_path = path.expanduser().resolve()
    if not is_supported_file(resolved_path):
        raise ValueError(f"Unsupported file type: {resolved_path.suffix or '<none>'}")

    reader = READERS[resolved_path.suffix.lower()]
    content = reader(resolved_path).strip()
    stat = resolved_path.stat()
    changed_epoch = float(stat.st_mtime)
    changed_at = datetime.fromtimestamp(changed_epoch, tz=timezone.utc).isoformat()
    collection_name = infer_collection_name(resolved_path)
    metadata = {
        "source": str(resolved_path),
        "path": str(resolved_path),
        "filename": resolved_path.name,
        "extension": resolved_path.suffix.lower(),
        "file_changed_at": changed_at,
        "file_changed_epoch": changed_epoch,
        "collection_name": collection_name,
        "content_type": collection_name,
    }
    return LoadedDocument(path=resolved_path, content=content, metadata=metadata)


def discover_documents(directory: Path, recursive: bool = True) -> list[Path]:
    resolved_dir = directory.expanduser().resolve()
    iterator = resolved_dir.rglob("*") if recursive else resolved_dir.glob("*")
    return sorted(path for path in iterator if is_supported_file(path))


def load_documents_from_directory(directory: Path, recursive: bool = True) -> list[LoadedDocument]:
    loaded_documents: list[LoadedDocument] = []
    for path in discover_documents(directory, recursive=recursive):
        try:
            document = load_document(path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue
        if document.content.strip():
            loaded_documents.append(document)
    return loaded_documents
