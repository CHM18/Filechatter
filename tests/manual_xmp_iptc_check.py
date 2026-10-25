#!/usr/bin/env python3
"""Inspect XMP / IPTC metadata of a photo folder: which fields are filled, with what values, and how much of it
looks like junk (camera default text, file names, numbers only, ...).

Usage (from the project root):
    python tests/manual_xmp_iptc_check.py D:\\Fotos
    python tests/manual_xmp_iptc_check.py D:\\Fotos -o values.xlsx --top 15
    python tests/manual_xmp_iptc_check.py D:\\Fotos -o per_photo.csv --per-photo
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, IptcImagePlugin

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp", ".heic", ".heif"}

IPTC_FIELDS = {
    (2, 5): "Title", (2, 25): "Keywords", (2, 80): "Byline", (2, 90): "City",
    (2, 101): "Country", (2, 105): "Headline", (2, 120): "Caption",
}
NS = {
    "http://purl.org/dc/elements/1.1/": "dc",
    "http://ns.adobe.com/xap/1.0/": "xmp",
    "http://ns.adobe.com/lightroom/1.0/": "lr",
    "http://ns.adobe.com/photoshop/1.0/": "photoshop",
    "http://www.metadataworkinggroup.com/schemas/regions/": "mwg-rs",
}
XMP_FIELDS = {
    "dc:title": "Title", "dc:description": "Description", "dc:subject": "Keywords",
    "lr:hierarchicalSubject": "LightroomKeywords", "xmp:Rating": "Rating", "xmp:Label": "Label",
    "photoshop:City": "City", "photoshop:Country": "Country", "xmp:CreatorTool": "CreatorTool",
}
RDF_LI = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"

JUNK_PATTERNS = [
    re.compile(r"^\W*$"),                                        # empty / punctuation only
    re.compile(r"^\d+$"),                                        # numbers only
    re.compile(r"^(img|dsc|dscn|dscf|pict|p|_mg|mvi|dcim|sam|pic)[_\- ]?\d+(\.\w+)?$", re.I),  # file names
    re.compile(r"\.(jpe?g|tiff?|png|heic|cr2|nef|arw|dng)$", re.I),
    re.compile(r"(digital camera|^sony dsc|^kodak|^olympus|^canon$|^nikon$|^untitled|^unbenannt|^default|"
               r"^created by|^generated|^scanned|^(no|kein)\b)", re.I),
    re.compile(r"^[0-9a-f]{16,}$", re.I),                        # hashes / ids
]


def xmp_packet(data: bytes) -> str:
    start = data.find(b"<x:xmpmeta")
    if start < 0:
        return ""
    end = data.find(b"</x:xmpmeta>", start)
    return data[start:end + 12].decode("utf-8", errors="ignore") if end > 0 else ""


def _qualified(name: str) -> str | None:
    if not name.startswith("{"):
        return None
    uri, local = name[1:].split("}", 1)
    return f"{NS[uri]}:{local}" if uri in NS else None


def _texts(element: ET.Element) -> list[str]:
    items = [(li.text or "").strip() for li in element.iter(RDF_LI)]
    items = items or [(element.text or "").strip()]
    return [item for item in items if item]


def xmp_values(packet: str) -> dict[str, list[str]]:
    if not packet:
        return {}
    try:
        root = ET.fromstring(packet)
    except ET.ParseError:
        return {}
    found: dict[str, list[str]] = defaultdict(list)
    for element in root.iter():
        qualified = _qualified(element.tag)
        if qualified in XMP_FIELDS:
            found[XMP_FIELDS[qualified]].extend(_texts(element))
        elif qualified == "mwg-rs:Regions":
            names = [a for e in element.iter() for k, a in e.attrib.items()
                     if k == "{http://www.metadataworkinggroup.com/schemas/regions/}Name" and a.strip()]
            found["FaceRegions"].extend(names or ["(regions without names)"])
        for key, value in element.attrib.items():
            qualified_attr = _qualified(key)
            if qualified_attr in XMP_FIELDS and value.strip():
                found[XMP_FIELDS[qualified_attr]].append(value.strip())
    return {key: list(dict.fromkeys(values)) for key, values in found.items() if values}


def iptc_values(path: Path) -> dict[str, list[str]]:
    try:
        with Image.open(path) as image:
            info = IptcImagePlugin.getiptcinfo(image) or {}
    except Exception:
        return {}
    found: dict[str, list[str]] = {}
    for key, label in IPTC_FIELDS.items():
        raw = info.get(key)
        items = raw if isinstance(raw, list) else [raw]
        texts = []
        for item in items:
            if isinstance(item, bytes):
                try:
                    item = item.decode("utf-8")
                except UnicodeDecodeError:
                    item = item.decode("latin-1")
            if item and str(item).strip():
                texts.append(str(item).strip())
        if texts:
            found[label] = texts
    return found


NON_TEXT_FIELDS = ("Rating", "Label")


def is_junk(value: str, field: str = "") -> bool:
    if field.endswith(NON_TEXT_FIELDS):
        return False
    return any(pattern.search(value.strip()) for pattern in JUNK_PATTERNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect XMP / IPTC values of a photo folder")
    parser.add_argument("folder", type=Path)
    parser.add_argument("-o", "--output", help=".csv or .xlsx: one row per field+value (or per photo with --per-photo)")
    parser.add_argument("--per-photo", action="store_true", help="write one row per photo instead of per value")
    parser.add_argument("--top", type=int, default=10, help="most frequent values shown per field")
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    pattern = "*" if args.no_recursive else "**/*"
    files = sorted(p for p in args.folder.glob(pattern) if p.is_file() and p.suffix.lower() in EXTENSIONS)

    values: dict[str, Counter[str]] = defaultdict(Counter)
    photos: dict[str, int] = Counter()
    per_photo = []
    for path in files:
        try:
            sources = {"XMP": xmp_values(xmp_packet(path.read_bytes())), "IPTC": iptc_values(path)}
        except OSError as exc:
            print(f"ERROR {path}: {exc}", file=sys.stderr)
            continue
        row = {"file": str(path.relative_to(args.folder))}
        for source, fields in sources.items():
            for label, texts in fields.items():
                field = f"{source} {label}"
                photos[field] += 1
                values[field].update(texts)
                row[field] = " | ".join(texts)
        per_photo.append(row)

    summary = []
    print(f"{len(files)} images\n")
    for field, counter in sorted(values.items(), key=lambda item: -photos[item[0]]):
        total = sum(counter.values())
        junk = sum(count for value, count in counter.items() if is_junk(value, field))
        print(f"== {field}: {photos[field]} photos, {len(counter)} distinct values, "
              f"{100 * junk / total:.0f}% look like junk")
        for value, count in counter.most_common(args.top):
            print(f"   {count:6d}  {'[junk] ' if is_junk(value, field) else ''}{value[:90]}")
        for value, count in counter.items():
            summary.append({"field": field, "value": value, "count": count, "junk": is_junk(value, field)})
        print()

    if args.output:
        rows = per_photo if args.per_photo else sorted(summary, key=lambda r: (r["field"], -r["count"]))
        columns = list(dict.fromkeys(key for row in rows for key in row))
        if args.output.lower().endswith(".xlsx"):
            from openpyxl import Workbook
            from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

            workbook = Workbook()
            sheet = workbook.active
            sheet.append([ILLEGAL_CHARACTERS_RE.sub("", column) for column in columns])
            for row in rows:
                sheet.append([ILLEGAL_CHARACTERS_RE.sub("", str(row.get(column, ""))) for column in columns])
            for line in sheet.iter_rows(min_row=2):
                for cell in line:
                    cell.number_format = "@"
            workbook.save(args.output)
        else:
            with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)
        print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
