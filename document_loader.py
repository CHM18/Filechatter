"""Helpers for loading supported document types into plain text.

Readers fall into three groups:
- rich formats with dedicated parsers: .pdf, .docx, .doc, .xlsx, .pptx
- markup that is stripped to text: .html/.htm
- everything else (plain text, markdown, code, config) read as UTF-8 text

Extensions are organized into categories (Office / Text / Web / Data & Config
/ Code) that the web UI uses to offer grouped, then individual, selection.
"""
from __future__ import annotations

import base64
import html as html_module
import importlib.util
import logging
import re
from urllib.parse import urlparse
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import requests

import config

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


_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".avif": "image/avif",
    ".heic": "image/heic",
}


def _normalize_base_url(base_url: str | None) -> str:
    value = (base_url or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"http://{value.lstrip('/')}"
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return value.rstrip("/")


def _extract_image_exif(path: Path) -> dict[str, Any]:
    try:
        from PIL import ExifTags, Image
    except ImportError as exc:
        raise RuntimeError("Reading image EXIF requires the optional dependency 'Pillow'.") from exc

    with Image.open(path) as image:
        raw_exif = image.getexif()
        if not raw_exif:
            return {}
        name_by_id = ExifTags.TAGS
        normalized: dict[str, Any] = {}
        for key, value in raw_exif.items():
            tag_name = name_by_id.get(key, str(key))
            if tag_name in config.IMAGE_EXIF_FIELDS and value not in (None, "", (), [], {}):
                normalized[tag_name] = str(value)
        return normalized


def _summarize_image_with_llm(path: Path) -> str:
    base_url = _normalize_base_url(config.IMAGE_VISION_BASE_URL)
    if not base_url:
        raise RuntimeError(
            "Image vision base URL is empty or invalid. Set IMAGE_VISION_BASE_URL (or LM_STUDIO_URL)."
        )

    model = str(config.IMAGE_VISION_MODEL or "").strip()
    if not model:
        raise RuntimeError("Image vision model is empty. Set IMAGE_VISION_MODEL.")

    mime = _IMAGE_MIME.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    file_context = f"File name: {path.name}. Folder: {path.parent.name}."
    prompt = (
        "Describe this image for retrieval in a local RAG index. "
        "Include visible entities, scene, activities, text shown in the image, approximate place/time cues, "
        "and notable colors/objects. Keep it factual and concise in 4-8 sentences. "
        f"{file_context}"
    )

    headers = {"Content-Type": "application/json"}
    if config.IMAGE_VISION_API_KEY:
        headers["Authorization"] = f"Bearer {config.IMAGE_VISION_API_KEY}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            }
        ],
        "max_tokens": int(config.IMAGE_VISION_MAX_TOKENS),
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=max(int(config.IMAGE_VISION_TIMEOUT), 30),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not analyze image via vision model at {base_url}: {exc}") from exc

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Vision model returned no choices.")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict) and item.get("type") in {None, "text"}
        )
    summary = str(content).strip()
    if not summary:
        raise RuntimeError("Vision model returned an empty image summary.")
    return summary


def _read_image(path: Path) -> tuple[str, dict[str, Any]]:
    summary = _summarize_image_with_llm(path)
    exif = _extract_image_exif(path)

    lines = [
        f"Image file: {path.name}",
        f"Folder: {path.parent.name}",
        f"Full path: {path}",
        "Vision summary:",
        summary,
    ]
    if exif:
        lines.append("EXIF metadata:")
        for key in config.IMAGE_EXIF_FIELDS:
            value = exif.get(key)
            if value:
                lines.append(f"- {key}: {value}")

    metadata = {
        "content_kind": "image",
        "image_summary": summary,
        "image_exif": exif,
        "image_vision_model": config.IMAGE_VISION_MODEL,
    }
    return "\n".join(lines).strip(), metadata


# ----------------------------------------------------------------------
# Categories and reader registry
# ----------------------------------------------------------------------

FILE_TYPE_CATEGORIES: dict[str, list[str]] = {
    "Office": [".pdf", ".docx", ".doc", ".xlsx", ".pptx"],
    "Images": [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif", ".heic"],
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
    ".jpg": ("Pillow", "PIL"),
    ".jpeg": ("Pillow", "PIL"),
    ".png": ("Pillow", "PIL"),
    ".webp": ("Pillow", "PIL"),
    ".gif": ("Pillow", "PIL"),
    ".bmp": ("Pillow", "PIL"),
    ".tif": ("Pillow", "PIL"),
    ".tiff": ("Pillow", "PIL"),
    ".avif": ("Pillow", "PIL"),
    ".heic": ("Pillow", "PIL"),
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

    extension = resolved_path.suffix.lower()
    extra_metadata: dict[str, Any] = {}
    if extension in FILE_TYPE_CATEGORIES["Images"]:
        content, extra_metadata = _read_image(resolved_path)
    else:
        reader = READERS[extension]
        body = reader(resolved_path).strip()
        content = (
            f"Source file: {resolved_path.name}\n"
            f"Source folder: {resolved_path.parent.name}\n"
            f"Full path: {resolved_path}\n\n"
            f"{body}"
        ).strip()
    stat = resolved_path.stat()
    changed_epoch = float(stat.st_mtime)
    changed_at = datetime.fromtimestamp(changed_epoch, tz=timezone.utc).isoformat()
    collection_name = infer_collection_name(resolved_path)
    metadata = {
        "source": str(resolved_path),
        "path": str(resolved_path),
        "filename": resolved_path.name,
        "extension": extension,
        "file_changed_at": changed_at,
        "file_changed_epoch": changed_epoch,
        "collection_name": collection_name,
        "content_type": collection_name,
    }
    metadata.update(extra_metadata)
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
