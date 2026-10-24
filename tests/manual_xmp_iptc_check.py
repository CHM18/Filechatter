#!/usr/bin/env python3
"""Check which photos in a folder carry XMP and/or IPTC metadata, and which useful fields are filled.

Usage (from the project root):
    python tests/manual_xmp_iptc_check.py D:\\Fotos
    python tests/manual_xmp_iptc_check.py D:\\Fotos -o xmp_iptc.csv --no-recursive
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

from PIL import Image, IptcImagePlugin

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp", ".heic", ".heif"}

# IPTC-IIM record 2 datasets worth indexing
IPTC_FIELDS = {
    (2, 5): "Title", (2, 25): "Keywords", (2, 80): "Byline", (2, 90): "City",
    (2, 101): "Country", (2, 105): "Headline", (2, 120): "Caption",
}
# XMP properties worth indexing (matched on the raw packet, so no XML library is needed)
XMP_FIELDS = {
    "dc:title": "Title", "dc:description": "Description", "dc:subject": "Keywords",
    "lr:hierarchicalSubject": "LightroomKeywords", "xmp:Rating": "Rating", "xmp:Label": "Label",
    "MP:RegionInfo": "FaceRegions(MS)", "mwg-rs:Regions": "FaceRegions(MWG)",
    "photoshop:City": "City", "photoshop:Country": "Country", "xmp:CreatorTool": "CreatorTool",
}


def xmp_packet(data: bytes) -> str:
    start = data.find(b"<x:xmpmeta")
    if start < 0:
        return ""
    end = data.find(b"</x:xmpmeta>", start)
    return data[start:end + 12].decode("utf-8", errors="ignore") if end > 0 else ""


def xmp_fields(packet: str) -> list[str]:
    found = []
    for tag, label in XMP_FIELDS.items():
        # element form (<tag>...</tag>, <tag/> excluded) or attribute form (tag="value")
        if re.search(rf"<{re.escape(tag)}[ >][^/]", packet) or re.search(rf'\b{re.escape(tag)}="[^"]+"', packet):
            found.append(label)
    return found


def iptc_fields(path: Path) -> list[str]:
    try:
        with Image.open(path) as image:
            info = IptcImagePlugin.getiptcinfo(image) or {}
    except Exception:
        return []
    return [label for key, label in IPTC_FIELDS.items() if info.get(key)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Find photos with XMP / IPTC metadata")
    parser.add_argument("folder", type=Path)
    parser.add_argument("-o", "--output", help="optional CSV with one row per photo")
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    pattern = "*" if args.no_recursive else "**/*"
    files = sorted(p for p in args.folder.glob(pattern) if p.is_file() and p.suffix.lower() in EXTENSIONS)

    with_xmp = with_iptc = 0
    counts: Counter[str] = Counter()
    rows = []
    for path in files:
        try:
            xmp = xmp_fields(xmp_packet(path.read_bytes()))
            iptc = iptc_fields(path)
        except OSError as exc:
            print(f"ERROR {path}: {exc}", file=sys.stderr)
            continue
        with_xmp += bool(xmp)
        with_iptc += bool(iptc)
        counts.update(f"XMP {label}" for label in xmp)
        counts.update(f"IPTC {label}" for label in iptc)
        rows.append({"file": str(path.relative_to(args.folder)), "xmp": ", ".join(xmp), "iptc": ", ".join(iptc)})

    print(f"{len(files)} images; {with_xmp} with useful XMP fields; {with_iptc} with useful IPTC fields")
    for name, count in counts.most_common():
        print(f"  {count:6d}  {name}")

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file", "xmp", "iptc"], delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
