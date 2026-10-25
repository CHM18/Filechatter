#!/usr/bin/env python3
"""Run the real vision analysis on a photo folder and dump the results to CSV/Excel (no indexing).

Use it to evaluate vision models and prompt settings on a sample. Results are cached per
(model, context on/off, file, mtime) in a JSON file so an interrupted run can be resumed
and finished photos are not analysed twice.

Usage (from the project root, vision server running):
    python tests/manual_vision_report.py D:\\Fotos -o vision.xlsx --limit 50
    python tests/manual_vision_report.py D:\\Fotos -o vision.xlsx --limit 50 --models qwen3-vl-8b,gemma-3-4b
    python tests/manual_vision_report.py D:\\Fotos -o vision.xlsx --limit 50 --no-context
    python tests/manual_vision_report.py D:\\Fotos -o vision.xlsx --random --seed 1 --limit 50
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

import config  # noqa: E402
import document_loader as dl  # noqa: E402
from xmp_iptc import extract_xmp_iptc  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", message="Metadata Warning")

EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp", ".heic", ".heif"}
COLUMNS = ["file", "model", "context", "seconds", "headline", "description", "keywords",
           "context_sent", "user_title", "user_description", "user_keywords", "error"]


def write_xlsx(path: str, rows: list[dict]) -> None:
    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(COLUMNS)
    for row in rows:
        sheet.append([ILLEGAL_CHARACTERS_RE.sub("", str(row.get(column, ""))) for column in COLUMNS])
    for line in sheet.iter_rows(min_row=2):
        for cell in line:
            cell.number_format = "@"
    workbook.save(path)


def analyse(path: Path, model: str, use_context: bool) -> dict:
    config.IMAGE_VISION_MODEL = model
    config.IMAGE_VISION_CONTEXT = use_context
    exif: dict = {}
    user_meta: dict = {}
    context = ""
    started = time.time()
    headline = description = ""
    keywords: list[str] = []
    error = ""
    try:
        exif = dl._extract_image_exif(path)
        user_meta = extract_xmp_iptc(path)
        context = dl._vision_context(path, exif, user_meta) if use_context else ""
        headline, description, keywords = dl._parse_image_analysis(
            dl._summarize_image_with_llm(path, context=context)
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "model": model,
        "context": "yes" if use_context else "no",
        "seconds": f"{time.time() - started:.1f}",
        "headline": headline,
        "description": description,
        "keywords": ", ".join(keywords),
        "context_sent": context.replace("\n", " | "),
        "user_title": user_meta.get("title", ""),
        "user_description": user_meta.get("description", ""),
        "user_keywords": ", ".join(user_meta.get("keywords", [])),
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump vision analysis results of a photo folder")
    parser.add_argument("folder", type=Path)
    parser.add_argument("-o", "--output", default="vision_report.xlsx", help="CSV, or .xlsx for Excel")
    parser.add_argument("--limit", type=int, default=0, help="Analyse at most N photos (0 = all)")
    parser.add_argument("--random", action="store_true", help="Pick the photos randomly instead of sorted")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", default="", help="Comma-separated model names to compare (default: IMAGE_VISION_MODEL)")
    parser.add_argument("--no-context", action="store_true", help="Do not send EXIF/XMP context to the model")
    parser.add_argument("--both-contexts", action="store_true", help="Run each photo with and without context")
    parser.add_argument("--cache", default="vision_report_cache.json")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    pattern = "*" if args.no_recursive else "**/*"
    files = sorted(p for p in args.folder.glob(pattern) if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if args.random:
        random.Random(args.seed).shuffle(files)
    if args.limit:
        files = files[: args.limit]

    models = [m.strip() for m in args.models.split(",") if m.strip()] or [config.IMAGE_VISION_MODEL]
    contexts = [True, False] if args.both_contexts else [not args.no_context]

    cache_path = Path(args.cache)
    cache: dict[str, dict] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    rows: list[dict] = []
    total = len(files) * len(models) * len(contexts)
    done = 0
    for path in files:
        relative = str(path.relative_to(args.folder))
        for model in models:
            for use_context in contexts:
                done += 1
                key = f"{model}|{use_context}|{path}|{path.stat().st_mtime}"
                if key not in cache or cache[key].get("error"):
                    cache[key] = analyse(path, model, use_context)
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                    print(f"[{done}/{total}] {relative} ({model}) {cache[key]['seconds']}s")
                rows.append({"file": relative, **cache[key]})

    if args.output.lower().endswith(".xlsx"):
        write_xlsx(args.output, rows)
    else:
        with open(args.output, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter=args.delimiter, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    errors = sum(1 for row in rows if row.get("error"))
    print(f"{len(files)} photos, {len(rows)} results, {errors} errors -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
