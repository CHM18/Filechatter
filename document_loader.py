"""Helpers for loading supported document types into plain text.

Readers fall into three groups:
- rich formats with dedicated parsers: .pdf, .docx, .doc, .xlsx, .pptx
- markup that is stripped to text: .html/.htm
- everything else (plain text, markdown, code, config) read as UTF-8 text

Extensions are organized into categories (Office / Text / Web / Data & Config
/ Code) that the web UI uses to offer grouped, then individual, selection.

OCR (Tesseract, via pytesseract) is an opt-in, per-collection feature: when a
collection has ocr_enabled=True, scanned PDFs (no text layer) and image files
are run through OCR instead of / in addition to their normal reader, and
embedded pictures inside .docx/.pptx/.xlsx are OCR'd inline as well.
"""
from __future__ import annotations

import base64
import hashlib
import html as html_module
import importlib.util
import io
import logging
import re
import shutil
import warnings
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
# OCR (Tesseract via pytesseract) - optional, gated per collection
# ----------------------------------------------------------------------

_MIN_TEXT_LAYER_CHARS = 20  # below this, a PDF page/document is treated as "scanned"
_MAX_OCR_IMAGE_DIMENSION_PX = 3500  # cap OCR input size: bounds runtime/memory on huge scans
_OSD_PROBE_MAX_DIMENSION_PX = 1000  # orientation detection runs on a small downscaled probe
_MIN_EMBEDDED_OCR_IMAGE_DIMENSION_PX = 80
_MIN_OCR_WORD_CONFIDENCE = 50.0
_MIN_SINGLE_WORD_OCR_CONFIDENCE = 70.0


def _has_tesseract_binary() -> bool:
    configured = str(getattr(config, "OCR_TESSERACT_CMD", "") or "").strip()
    if configured:
        return Path(configured).is_file()
    return shutil.which("tesseract") is not None


def is_ocr_available() -> bool:
    """Whether OCR can actually run: pytesseract installed AND the tesseract binary is reachable."""
    return _has_module("pytesseract") and _has_tesseract_binary()


def _configured_tesseract():
    import pytesseract

    configured = str(getattr(config, "OCR_TESSERACT_CMD", "") or "").strip()
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
    return pytesseract


def _cap_image_dimensions(image):
    """Downscale oversized images before OCR to bound runtime/memory (e.g. huge historical scans)."""
    if max(image.size) > _MAX_OCR_IMAGE_DIMENSION_PX:
        scale = _MAX_OCR_IMAGE_DIMENSION_PX / max(image.size)
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        return image.resize(new_size)
    return image


def _correct_orientation(pytesseract, image):
    """Cheap OSD pass on a small downscaled probe to auto-correct sideways/upside-down scans."""
    probe = image
    if max(image.size) > _OSD_PROBE_MAX_DIMENSION_PX:
        scale = _OSD_PROBE_MAX_DIMENSION_PX / max(image.size)
        probe = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    try:
        osd = pytesseract.image_to_osd(probe, output_type=pytesseract.Output.DICT)
        rotation = int(osd.get("rotate", 0)) % 360
    except Exception as exc:
        # OSD needs a minimum amount of legible text; small/blank images routinely fail this.
        logger.debug("OSD orientation detection skipped: %s", exc)
        return image
    return image.rotate(-rotation, expand=True) if rotation else image


def _ocr_pil_image(image) -> str:
    if not is_ocr_available():
        raise RuntimeError(
            "OCR requires the optional dependency 'pytesseract' and a Tesseract OCR "
            "installation on the host (the binary cannot be installed via pip)."
        )
    pytesseract = _configured_tesseract()
    image = _correct_orientation(pytesseract, image)
    from PIL import ImageOps

    image = ImageOps.autocontrast(image.convert("L"))
    if max(image.size) < 1200:
        scale = min(2.0, 1600 / max(image.size))
        image = image.resize((int(image.width * scale), int(image.height * scale)))

    data = pytesseract.image_to_data(
        image,
        lang=config.OCR_LANGUAGE,
        output_type=pytesseract.Output.DICT,
    )
    lines: dict[tuple[str, str, str], list[str]] = {}
    accepted_confidences: list[float] = []
    for index, text in enumerate(data.get("text", [])):
        word = text.strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if confidence < _MIN_OCR_WORD_CONFIDENCE:
            continue
        line_key = tuple(
            str(data.get(key, ["0"] * len(data["text"]))[index])
            for key in ("block_num", "par_num", "line_num")
        )
        lines.setdefault(line_key, []).append(word)
        accepted_confidences.append(confidence)

    if not accepted_confidences:
        return ""
    if len(accepted_confidences) == 1 and accepted_confidences[0] < _MIN_SINGLE_WORD_OCR_CONFIDENCE:
        return ""
    return "\n".join(" ".join(words) for words in lines.values())


def _ocr_image_bytes(data: bytes) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("OCR requires the optional dependency 'Pillow'.") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            rgb = _cap_image_dimensions(image.convert("RGB"))
    return _ocr_pil_image(rgb)


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


def _read_pdf_pages(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Reading .pdf files requires the optional dependency 'pypdf'.") from exc
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def _read_pdf_with_ocr(path: Path, text_pages: list[str]) -> tuple[str, dict[str, Any]]:
    """OCR pages that need it: fully rasterize pages with no usable text layer, and
    additionally OCR any embedded raster images on otherwise-text pages (e.g. a
    background photo of a bill with a thin real text layer on top of it), without
    re-rendering/re-OCRing the whole page in that second case."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("Scanned-PDF OCR requires the optional dependency 'pymupdf'.") from exc

    parts: list[str] = []
    ocr_pages = 0
    ocr_image_count = 0
    seen_image_hashes: set[str] = set()
    document = fitz.open(str(path))
    try:
        for index, page in enumerate(document):
            existing_text = text_pages[index] if index < len(text_pages) else ""
            if len(existing_text.strip()) < _MIN_TEXT_LAYER_CHARS:
                # Likely a fully scanned page: OCR the whole rendered page. Cap the
                # render DPI so physically large pages (e.g. historical A3 scans)
                # don't blow up into huge, slow-to-OCR pixmaps.
                long_side_inches = max(page.rect.width, page.rect.height) / 72.0
                dpi = 300
                if long_side_inches > 0:
                    dpi = max(100, min(300, int(_MAX_OCR_IMAGE_DIMENSION_PX / long_side_inches)))
                pixmap = page.get_pixmap(dpi=dpi)
                try:
                    ocr_text = _ocr_image_bytes(pixmap.tobytes("png"))
                except Exception as exc:
                    logger.warning("OCR failed on %s page %d: %s", path, index + 1, exc)
                    ocr_text = existing_text
                else:
                    ocr_pages += 1
                parts.append(ocr_text)
                continue

            # Page already has a usable text layer; only OCR embedded raster
            # images (e.g. a background photo) to catch any extra text in them.
            page_parts = [existing_text]
            for image_info in page.get_images(full=True):
                xref = image_info[0]
                if max(image_info[2], image_info[3]) < _MIN_EMBEDDED_OCR_IMAGE_DIMENSION_PX:
                    continue
                try:
                    image_bytes = document.extract_image(xref)["image"]
                    image_hash = hashlib.sha256(image_bytes).hexdigest()
                    if image_hash in seen_image_hashes:
                        continue
                    seen_image_hashes.add(image_hash)
                    ocr_text = _ocr_image_bytes(image_bytes)
                except Exception as exc:
                    logger.warning("Embedded-image OCR failed on %s page %d: %s", path, index + 1, exc)
                    continue
                if ocr_text:
                    ocr_image_count += 1
                    page_parts.append(f"[Image OCR]: {ocr_text}")
            parts.append("\n".join(page_parts))
    finally:
        document.close()

    metadata: dict[str, Any] = {}
    if ocr_pages or ocr_image_count:
        metadata = {
            "extraction_method": "ocr" if ocr_pages == len(parts) and not ocr_image_count else "mixed",
            "ocr_used": True,
            "ocr_language": config.OCR_LANGUAGE,
            "ocr_pages": ocr_pages,
            "ocr_image_count": ocr_image_count,
        }
    return "\n\n".join(parts), metadata


def _read_docx(path: Path, ocr_enabled: bool = False) -> tuple[str, dict[str, Any]]:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise RuntimeError("Reading .docx files requires the optional dependency 'python-docx'.") from exc
    document = DocxDocument(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]

    ocr_image_count = 0
    if ocr_enabled and is_ocr_available():
        paragraphs = []
        blip_tag = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                paragraphs.append(paragraph.text)
            for blip in paragraph._element.findall(f".//{blip_tag}"):
                relationship_id = blip.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                )
                if not relationship_id:
                    continue
                try:
                    image_part = document.part.related_parts[relationship_id]
                    ocr_text = _ocr_image_bytes(image_part.blob)
                except Exception as exc:
                    logger.warning("Embedded-image OCR failed in %s: %s", path, exc)
                    continue
                if ocr_text:
                    ocr_image_count += 1
                    paragraphs.append(f"[Image OCR]: {ocr_text}")
        # Note: images embedded directly inside table cells are not walked here.

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))

    metadata: dict[str, Any] = {}
    if ocr_image_count:
        metadata = {
            "extraction_method": "mixed",
            "ocr_used": True,
            "ocr_language": config.OCR_LANGUAGE,
            "ocr_image_count": ocr_image_count,
        }
    return "\n".join(paragraphs), metadata


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


def _read_xlsx(path: Path, ocr_enabled: bool = False) -> tuple[str, dict[str, Any]]:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("Reading .xlsx files requires the optional dependency 'openpyxl'.") from exc

    # read_only mode is faster but doesn't expose embedded images, so only pay
    # for a full-load pass when OCR was actually requested and is available.
    use_ocr = ocr_enabled and is_ocr_available()
    workbook = openpyxl.load_workbook(str(path), read_only=not use_ocr, data_only=True)
    parts: list[str] = []
    ocr_image_count = 0
    try:
        for worksheet in workbook.worksheets:
            parts.append(f"# Sheet: {worksheet.title}")
            images_by_row: dict[int, list[Any]] = {}
            if use_ocr:
                for image in getattr(worksheet, "_images", []):
                    anchor_row = getattr(getattr(image, "anchor", None), "_from", None)
                    row_index = getattr(anchor_row, "row", None)
                    images_by_row.setdefault(row_index if row_index is not None else -1, []).append(image)

            for row_index, row in enumerate(worksheet.iter_rows(values_only=True)):
                cells = [str(cell) for cell in row if cell is not None]
                if cells:
                    parts.append(" | ".join(cells))
                for image in images_by_row.pop(row_index, []):
                    try:
                        ocr_text = _ocr_image_bytes(image._data())
                    except Exception as exc:
                        logger.warning("Embedded-image OCR failed in %s: %s", path, exc)
                        continue
                    if ocr_text:
                        ocr_image_count += 1
                        parts.append(f"[Image OCR @ row {row_index + 1}]: {ocr_text}")
            # Images anchored beyond the last data row (or with an unknown anchor).
            for remaining in images_by_row.values():
                for image in remaining:
                    try:
                        ocr_text = _ocr_image_bytes(image._data())
                    except Exception as exc:
                        logger.warning("Embedded-image OCR failed in %s: %s", path, exc)
                        continue
                    if ocr_text:
                        ocr_image_count += 1
                        parts.append(f"[Image OCR]: {ocr_text}")
    finally:
        workbook.close()

    metadata: dict[str, Any] = {}
    if ocr_image_count:
        metadata = {
            "extraction_method": "mixed",
            "ocr_used": True,
            "ocr_language": config.OCR_LANGUAGE,
            "ocr_image_count": ocr_image_count,
        }
    return "\n".join(parts), metadata


def _read_pptx(path: Path, ocr_enabled: bool = False) -> tuple[str, dict[str, Any]]:
    try:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except ImportError as exc:
        raise RuntimeError("Reading .pptx files requires the optional dependency 'python-pptx'.") from exc
    presentation = Presentation(str(path))
    parts: list[str] = []
    use_ocr = ocr_enabled and is_ocr_available()
    ocr_image_count = 0
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
            if use_ocr and shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    ocr_text = _ocr_image_bytes(shape.image.blob)
                except Exception as exc:
                    logger.warning("Embedded-image OCR failed in %s: %s", path, exc)
                    continue
                if ocr_text:
                    ocr_image_count += 1
                    parts.append(f"[Image OCR]: {ocr_text}")

    metadata: dict[str, Any] = {}
    if ocr_image_count:
        metadata = {
            "extraction_method": "mixed",
            "ocr_used": True,
            "ocr_language": config.OCR_LANGUAGE,
            "ocr_image_count": ocr_image_count,
        }
    return "\n".join(parts), metadata


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


def _image_bytes_for_vision(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    mime = _IMAGE_MIME.get(path.suffix.lower(), "image/jpeg")
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Reading image orientation requires the optional dependency 'Pillow'.") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            orientation = image.getexif().get(274)
            if orientation not in range(2, 9):
                return data, mime

            corrected = ImageOps.exif_transpose(image)
            output = io.BytesIO()
            save_options: dict[str, Any] = {"format": image.format}
            if image.format == "JPEG":
                save_options["quality"] = 95
            corrected.save(output, **save_options)
            return output.getvalue(), mime


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

    image_data, mime = _image_bytes_for_vision(path)
    encoded = base64.b64encode(image_data).decode("ascii")
    file_context = f"File name: {path.name}. Folder: {path.parent.name}."
    prompt = (
        "Describe this image for retrieval in a local RAG index. "
        "Include visible entities, scene, activities, text shown in the image, approximate place/time cues, "
        "and notable colors/objects. Keep the description factual and concise in 4-8 sentences. "
        "After the description, add a separate line exactly in this format: "
        "Searchable keywords: term one, term two, term three. "
        "Use concise searchable terms for visible entities, scene, activities, readable text, and notable colors/objects; "
        "include place/time cues only when supported by the image. Do not invent details. "
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


def _parse_image_analysis(response: str) -> tuple[str, list[str]]:
    sections = re.split(r"(?im)^\s*searchable keywords\s*:\s*", response, maxsplit=1)
    description = re.sub(r"(?im)^\s*description\s*:\s*", "", sections[0], count=1).strip()
    keywords: list[str] = []
    if len(sections) > 1:
        seen: set[str] = set()
        for keyword in re.split(r"[,;]", sections[1].splitlines()[0]):
            cleaned = keyword.strip(" \t-*.")
            normalized = cleaned.casefold()
            if cleaned and normalized not in seen:
                keywords.append(cleaned)
                seen.add(normalized)
    return description, keywords


def _read_image(path: Path) -> tuple[str, dict[str, Any]]:
    analysis = _summarize_image_with_llm(path)
    summary, keywords = _parse_image_analysis(analysis)
    exif = _extract_image_exif(path)

    lines = [
        f"Image file: {path.name}",
        f"Folder: {path.parent.name}",
        f"Full path: {path}",
        "Vision summary:",
        summary,
    ]
    if keywords:
        lines.append(f"Searchable keywords: {', '.join(keywords)}")
    if exif:
        lines.append("EXIF metadata:")
        for key in config.IMAGE_EXIF_FIELDS:
            value = exif.get(key)
            if value:
                lines.append(f"- {key}: {value}")

    metadata = {
        "content_kind": "image",
        "image_summary": summary,
        "image_keywords": keywords,
        "image_exif": exif,
        "image_vision_model": config.IMAGE_VISION_MODEL,
    }
    return "\n".join(lines).strip(), metadata


def _read_image_ocr(path: Path) -> tuple[str, dict[str, Any]]:
    ocr_text = _ocr_pil_image(_open_image_rgb(path))

    lines = [
        f"Image file: {path.name}",
        f"Folder: {path.parent.name}",
        f"Full path: {path}",
        "OCR text:",
        ocr_text or "(no text detected)",
    ]

    metadata = {
        "content_kind": "image",
        "extraction_method": "ocr",
        "ocr_used": True,
        "ocr_language": config.OCR_LANGUAGE,
    }
    return "\n".join(lines).strip(), metadata


def _open_image_rgb(path: Path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("OCR requires the optional dependency 'Pillow'.") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            return _cap_image_dimensions(image.convert("RGB"))


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
# Note: load_document() special-cases .pdf/.docx/.xlsx/.pptx directly (for OCR
# support) rather than calling through this dict; it stays here only for
# SUPPORTED_EXTENSIONS/availability bookkeeping below.
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


def load_document(path: Path, *, ocr_enabled: bool = False) -> LoadedDocument:
    resolved_path = path.expanduser().resolve()
    if not is_supported_file(resolved_path):
        raise ValueError(f"Unsupported file type: {resolved_path.suffix or '<none>'}")

    extension = resolved_path.suffix.lower()
    extra_metadata: dict[str, Any] = {}
    use_ocr = ocr_enabled and is_ocr_available()

    if extension in FILE_TYPE_CATEGORIES["Images"]:
        if use_ocr:
            content, extra_metadata = _read_image_ocr(resolved_path)
        else:
            content, extra_metadata = _read_image(resolved_path)
            if ocr_enabled:
                extra_metadata["ocr_requested_but_unavailable"] = True
    elif extension == ".pdf":
        text_pages = _read_pdf_pages(resolved_path)
        if use_ocr:
            body, extra_metadata = _read_pdf_with_ocr(resolved_path, text_pages)
        else:
            body = "\n\n".join(text_pages)
        if ocr_enabled and not is_ocr_available():
            extra_metadata["ocr_requested_but_unavailable"] = True
        content = (
            f"Source file: {resolved_path.name}\n"
            f"Source folder: {resolved_path.parent.name}\n"
            f"Full path: {resolved_path}\n\n"
            f"{body.strip()}"
        ).strip()
    elif extension in (".docx", ".xlsx", ".pptx"):
        reader = {".docx": _read_docx, ".xlsx": _read_xlsx, ".pptx": _read_pptx}[extension]
        body, extra_metadata = reader(resolved_path, ocr_enabled=use_ocr)
        if ocr_enabled and not is_ocr_available():
            extra_metadata["ocr_requested_but_unavailable"] = True
        content = (
            f"Source file: {resolved_path.name}\n"
            f"Source folder: {resolved_path.parent.name}\n"
            f"Full path: {resolved_path}\n\n"
            f"{body.strip()}"
        ).strip()
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
    metadata.setdefault("ocr_used", False)
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
