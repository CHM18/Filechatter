"""Re-scan behaviour for photos: skip unchanged analysed images, re-analyse on fingerprint change."""
from __future__ import annotations

from PIL import Image

import document_loader as dl
import runtime
from ingest_jobs import IngestJob


def _run_job(folder):
    manager = runtime.jobs()
    job = IngestJob(job_id="j", path=str(folder), recursive=True, total_files=0, collection_name="all")
    manager._jobs[job.job_id] = job
    manager._run(job.job_id)
    return job


def test_rescan_skips_unchanged_and_reanalyses_on_fingerprint_change(data_env, tmp_path, monkeypatch):
    folder = tmp_path / "pics"
    folder.mkdir()
    Image.new("RGB", (8, 8)).save(folder / "a.jpg")
    calls = []
    monkeypatch.setattr(
        dl, "_summarize_image_with_llm",
        lambda p, **_kw: calls.append(p) or "Headline: Test\nDescription: Ein Bild.\nSearchable keywords: Bild, picture",
    )
    monkeypatch.setattr(dl.config, "IMAGE_SKIP_ANALYZED", True)
    monkeypatch.setattr(dl.config, "IMAGE_VISION_MODEL", "model-a")

    first = _run_job(folder)
    assert (first.succeeded_files, first.skipped_unchanged, len(calls)) == (1, 0, 1)

    second = _run_job(folder)
    assert (second.succeeded_files, second.skipped_unchanged, len(calls)) == (0, 1, 1)

    monkeypatch.setattr(dl.config, "IMAGE_VISION_MODEL", "model-b")
    third = _run_job(folder)
    assert (third.succeeded_files, third.skipped_unchanged, len(calls)) == (1, 0, 2)

    store = runtime.collections().get_store("documentation")
    assert store.chunk_count >= 1
