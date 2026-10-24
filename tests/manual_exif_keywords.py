#!/usr/bin/env python3
"""Dump the EXIF values and derived keywords Filechatter would store for a photo folder.

Uses the real extraction code and your .env (IMAGE_EXIF_FIELDS, IMAGE_EXIF_GEOCODE, ...).
No vision model and no indexing is involved.

Usage (from the project root):
    python tests/manual_exif_keywords.py D:\\Fotos -o exif_keywords.csv
    python tests/manual_exif_keywords.py D:\\Fotos --no-recursive --delimiter ,
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

import config  # noqa: E402
from document_loader import _extract_image_exif  # noqa: E402
from exif_keywords import capture_datetime, derive_exif_keywords  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", message="Metadata Warning")

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp", ".heic", ".heif"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump EXIF values and derived keywords of a photo folder")
    parser.add_argument("folder", type=Path)
    parser.add_argument("-o", "--output", default="exif_keywords.csv")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    pattern = "*" if args.no_recursive else "**/*"
    files = sorted(p for p in args.folder.glob(pattern) if p.is_file() and p.suffix.lower() in EXTENSIONS)
    columns = ["file", "capture_time", "keywords", *config.IMAGE_EXIF_FIELDS]

    errors = 0
    with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=args.delimiter, extrasaction="ignore")
        writer.writeheader()
        for path in files:
            try:
                exif = _extract_image_exif(path)
            except Exception as exc:
                errors += 1
                print(f"ERROR {path}: {exc}", file=sys.stderr)
                continue
            moment = capture_datetime(exif)
            row = {
                "file": str(path.relative_to(args.folder)),
                "capture_time": moment.isoformat() if moment else "",
                "keywords": ", ".join(derive_exif_keywords(exif)),
                **{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                   for key, value in exif.items()},
            }
            writer.writerow(row)

    print(f"{len(files)} images, {errors} errors -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
