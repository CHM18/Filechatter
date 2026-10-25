"""Extract user-written XMP / IPTC metadata (title, description, keywords, people, place) from images,
filter junk values and remove information that is duplicated across XMP, IPTC and EXIF."""
from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import config

SUPPORTED_FIELDS = ("Title", "Description", "Keywords", "People", "City", "Country", "CreatorTool")

_IPTC = {
    (2, 5): "Title", (2, 25): "Keywords", (2, 90): "City", (2, 101): "Country", (2, 120): "Description",
}
_NS = {
    "http://purl.org/dc/elements/1.1/": "dc",
    "http://ns.adobe.com/xap/1.0/": "xmp",
    "http://ns.adobe.com/photoshop/1.0/": "photoshop",
    "http://www.metadataworkinggroup.com/schemas/regions/": "mwg-rs",
}
_XMP = {
    "dc:title": "Title", "dc:description": "Description", "dc:subject": "Keywords",
    "photoshop:City": "City", "photoshop:Country": "Country", "xmp:CreatorTool": "CreatorTool",
}
_RDF_LI = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"
_MWG_NAME = "{http://www.metadataworkinggroup.com/schemas/regions/}Name"

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Generic junk for free text (title / description): bare numbers, file names, camera file counters
_JUNK_TEXT = [
    re.compile(r"^\W*$"),
    re.compile(r"^\d+$"),
    re.compile(r"^(img|dsc|dscn|dscf|pict|p|_mg|mvi|dcim|sam|pic)[_\- ]?\d+(\.\w+)?$", re.I),
    re.compile(r"\.(jpe?g|tiff?|png|heic|cr2|nef|arw|dng)$", re.I),
]


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", _CONTROL_CHARS.sub("", str(text))).strip()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def is_ignored(text: str) -> bool:
    """True if the value is on the user's ignore list (IMAGE_METADATA_IGNORE_VALUES)."""
    return normalize(text) in {normalize(item) for item in config.IMAGE_METADATA_IGNORE_VALUES}


def _is_junk_text(text: str) -> bool:
    return is_ignored(text) or any(pattern.search(text) for pattern in _JUNK_TEXT)


def _qualified(name: str) -> str | None:
    if not name.startswith("{"):
        return None
    uri, local = name[1:].split("}", 1)
    return f"{_NS[uri]}:{local}" if uri in _NS else None


def _li_texts(element: ET.Element) -> list[str]:
    items = [_clean(li.text or "") for li in element.iter(_RDF_LI)] or [_clean(element.text or "")]
    return [item for item in items if item]


def _xmp_packet(image: Any) -> str:
    raw = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp") or b""
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="ignore")


def _parse_xmp(packet: str) -> dict[str, list[str]]:
    if not packet.strip():
        return {}
    try:
        root = ET.fromstring(packet.strip().strip("\x00"))
    except ET.ParseError:
        return {}
    found: dict[str, list[str]] = {}

    def add(field: str, values: list[str]) -> None:
        found.setdefault(field, []).extend(value for value in values if value)

    for element in root.iter():
        qualified = _qualified(element.tag)
        if qualified in _XMP:
            add(_XMP[qualified], _li_texts(element))
        elif qualified == "mwg-rs:Regions":
            names = [_clean(e.attrib[_MWG_NAME]) for e in element.iter() if _MWG_NAME in e.attrib]
            names += [_clean(e.text or "") for e in element.iter() if e.tag == _MWG_NAME]
            add("People", names)
        for key, value in element.attrib.items():
            attribute = _qualified(key)
            if attribute in _XMP:
                add(_XMP[attribute], [_clean(value)])
    return found


def _decode_iptc(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return _clean(value.decode("utf-8"))
        except UnicodeDecodeError:
            return _clean(value.decode("cp1252", errors="ignore"))
    return _clean(value)


def _parse_iptc(image: Any) -> dict[str, list[str]]:
    from PIL import IptcImagePlugin

    try:
        info = IptcImagePlugin.getiptcinfo(image) or {}
    except Exception:
        return {}
    found: dict[str, list[str]] = {}
    for key, field in _IPTC.items():
        raw = info.get(key)
        if raw is None:
            continue
        values = [_decode_iptc(item) for item in (raw if isinstance(raw, list) else [raw])]
        values = [value for value in values if value]
        if values:
            found[field] = values
    return found


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = normalize(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def extract_xmp_iptc(path: Path) -> dict[str, Any]:
    """Return the configured (IMAGE_XMP_FIELDS) fields, junk-filtered and merged across XMP and IPTC.

    Result keys: title, description (str), keywords, people (list), city, country, creator_tool (str).
    Only non-empty values are included."""
    wanted = {field for field in config.IMAGE_XMP_FIELDS if field in SUPPORTED_FIELDS}
    if not wanted:
        return {}
    try:
        from PIL import Image

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as image:
                xmp = _parse_xmp(_xmp_packet(image))
                iptc = _parse_iptc(image) if image.format in {"JPEG", "TIFF"} else {}
    except Exception:
        return {}

    def merged(field: str) -> list[str]:
        return _unique([*xmp.get(field, []), *iptc.get(field, [])])

    result: dict[str, Any] = {}
    for field, key in (("Title", "title"), ("Description", "description")):
        if field in wanted:
            texts = [text for text in merged(field) if not _is_junk_text(text)]
            if texts:
                result[key] = _drop_contained(texts)
    if "Keywords" in wanted:
        keywords = [k for k in merged("Keywords") if not is_ignored(k) and not re.fullmatch(r"\W*", k)]
        if keywords:
            result["keywords"] = keywords
    if "People" in wanted:
        people = [name for name in merged("People") if not is_ignored(name)]
        if people:
            result["people"] = people
    for field in ("City", "Country"):
        if field in wanted:
            values = [value for value in merged(field) if not is_ignored(value)]
            if values:
                result[field.lower()] = values[0]
    if "CreatorTool" in wanted and xmp.get("CreatorTool"):
        result["creator_tool"] = xmp["CreatorTool"][0]

    # Title/description: keep strings, not lists
    for key in ("title", "description"):
        if key in result:
            result[key] = " | ".join(result[key])
    return result


def _drop_contained(texts: list[str]) -> list[str]:
    """Drop a text that is fully contained in another (same caption stored in several places)."""
    keep: list[str] = []
    for text in texts:
        if any(normalize(text) in normalize(other) for other in texts if other is not text
               and len(normalize(other)) > len(normalize(text))):
            continue
        keep.append(text)
    return keep


def deduplicate_against(texts: dict[str, str], exif: dict[str, Any]) -> dict[str, Any]:
    """Remove EXIF ImageDescription when it is junk or repeats the XMP/IPTC title or description.

    Returns the (possibly reduced) copy of the EXIF dict."""
    description = exif.get("ImageDescription")
    if description in (None, ""):
        return exif
    text = _clean(description)
    cleaned = dict(exif)
    others = [normalize(value) for value in texts.values() if value]
    key = normalize(text)
    repeated = any(key == other or (len(key) >= 15 and key in other) for other in others)
    if is_ignored(text) or _is_junk_text(text) or repeated:
        cleaned.pop("ImageDescription")
    else:
        cleaned["ImageDescription"] = text
    return cleaned


def dedupe_keywords(*groups: list[str]) -> list[list[str]]:
    """Case-insensitive de-duplication across keyword groups; earlier groups win."""
    seen: set[str] = set()
    result: list[list[str]] = []
    for group in groups:
        kept = []
        for keyword in group:
            key = normalize(keyword)
            if key and key not in seen:
                seen.add(key)
                kept.append(keyword)
        result.append(kept)
    return result
