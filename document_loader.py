"""Helpers for loading supported document types into plain text."""
from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc"}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".psm1",
    ".sql",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".swift",
    ".kt",
    ".kts",
    ".dart",
    ".scala",
    ".lua",
    ".pl",
    ".pm",
    ".r",
    ".m",
    ".mm",
    ".groovy",
    ".gradle",
    ".lock",
    ".cmake",
}
CODE_FILENAMES = {"dockerfile", "makefile"}


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


@dataclass(slots=True)
class LoadedDocument:
    path: Path
    content: str
    metadata: dict[str, Any]


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Reading .pdf files requires the optional dependency 'pypdf'."
        ) from exc

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise RuntimeError(
            "Reading .docx files requires the optional dependency 'python-docx'."
        ) from exc

    document = DocxDocument(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]

    # Include table content as plain text to avoid silently dropping structured data.
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
        raise RuntimeError(
            "Reading .doc files requires pywin32 and Microsoft Word on Windows."
        ) from exc

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


READERS = {
    ".txt": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".doc": _read_doc,
}


def get_document_support_status() -> dict[str, dict[str, Any]]:
    support: dict[str, dict[str, Any]] = {
        ".txt": {
            "available": True,
            "dependency": None,
            "message": "Built-in text reader is available.",
        },
        ".pdf": {
            "available": _has_module("pypdf"),
            "dependency": "pypdf",
            "message": "Requires the 'pypdf' package.",
        },
        ".docx": {
            "available": _has_module("docx"),
            "dependency": "python-docx",
            "message": "Requires the 'python-docx' package.",
        },
        ".doc": {
            "available": sys.platform.startswith("win") and _has_module("win32com"),
            "dependency": "pywin32 + Microsoft Word",
            "message": "Requires Windows, pywin32, and a local Microsoft Word installation.",
        },
    }

    for extension, details in support.items():
        if details["available"]:
            details["status"] = "ready"
        else:
            details["status"] = "missing_dependency"
    return support


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
    collection_name = infer_collection_name(resolved_path)
    # Use full path as source to avoid collisions with same-named files in different directories
    metadata = {
        "source": str(resolved_path),
        "path": str(resolved_path),
        "filename": resolved_path.name,
        "extension": resolved_path.suffix.lower(),
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
