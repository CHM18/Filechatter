"""Manual OCR benchmark against real sample scans in tests/test_data.

Not picked up by pytest (name doesn't match test_*.py / *_test.py) because it
needs a real Tesseract install and produces human-eyeballed output rather than
assertions. Run directly:

    python tests/manual_ocr_benchmark.py [filename ...]

With no arguments it runs every supported file in tests/test_data; pass one or
more filenames to only run those.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import document_loader as dl

TESTDATA_DIR = Path(__file__).resolve().parent / "test_data"


def run(paths: list[Path]) -> None:
    if not dl.is_ocr_available():
        print("Tesseract is not available (see README 'OCR setup') - aborting.")
        return

    for path in paths:
        if path.suffix.lower() not in dl.SUPPORTED_EXTENSIONS:
            print(f"{path.name}: SKIP (unsupported extension)")
            continue
        start = time.time()
        try:
            loaded = dl.load_document(path, ocr_enabled=True)
        except Exception as exc:
            print(f"{path.name}: ERROR {exc}")
            continue
        elapsed = time.time() - start
        meta = loaded.metadata
        print(f"--- {path.name} ({elapsed:.1f}s) ---")
        print(
            "  ocr_used=", meta.get("ocr_used"),
            "method=", meta.get("extraction_method"),
            "pages=", meta.get("ocr_pages"),
            "images=", meta.get("ocr_image_count"),
        )
        print("  content_len=", len(loaded.content))
        print("  snippet:", loaded.content[:300].replace("\n", " | "))
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run([TESTDATA_DIR / name for name in sys.argv[1:]])
    else:
        run(sorted(TESTDATA_DIR.iterdir()))
