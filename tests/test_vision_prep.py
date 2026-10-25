"""Tests for image preparation, prompt and response parsing for the vision model."""
import io

import pytest
from PIL import Image

import document_loader as dl


@pytest.fixture(autouse=True)
def _no_downscale(monkeypatch):
    monkeypatch.setattr(dl.config, "IMAGE_VISION_MAX_SIDE", 0)


@pytest.mark.parametrize("orientation,expected", [(2, (4, 2)), (3, (4, 2)), (4, (4, 2)),
                                                  (5, (2, 4)), (6, (2, 4)), (7, (2, 4)), (8, (2, 4))])
def test_all_exif_orientations_are_corrected(tmp_path, orientation, expected):
    path = tmp_path / "o.jpg"
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (4, 2), "red").save(path, exif=exif)

    data, mime = dl._image_bytes_for_vision(path)

    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == expected


def test_16bit_tiff_is_sent_as_png(tmp_path):
    path = tmp_path / "hdr.tif"
    Image.new("I;16", (8, 4), 40000).save(path)

    data, mime = dl._image_bytes_for_vision(path)

    assert mime == "image/png"
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "PNG"
        assert image.size == (8, 4)


def test_heic_is_readable_and_sent_as_png(tmp_path):
    pytest.importorskip("pillow_heif")
    path = tmp_path / "photo.heic"
    exif = Image.Exif()
    exif[271] = "Apple"
    Image.new("RGB", (40, 20), "red").save(path, format="HEIF", exif=exif)

    data, mime = dl._image_bytes_for_vision(path)

    assert mime == "image/png"
    assert dl._extract_image_exif(path).get("Make") == "Apple"


def test_large_image_is_downscaled(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.config, "IMAGE_VISION_MAX_SIDE", 100)
    path = tmp_path / "big.jpg"
    Image.new("RGB", (400, 200), "green").save(path)

    data, mime = dl._image_bytes_for_vision(path)

    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (100, 50)


def test_parse_headline_description_and_keywords():
    headline, description, keywords = dl._parse_image_analysis(
        "Headline: Hund am Strand\nDescription: Ein Hund rennt.\nSearchable keywords: Hund, dog, Strand"
    )
    assert headline == "Hund am Strand"
    assert description == "Ein Hund rennt."
    assert keywords == ["Hund", "dog", "Strand"]


def test_parse_without_headline():
    headline, description, keywords = dl._parse_image_analysis("Ein Bild.\nSearchable keywords: Bild")
    assert (headline, description, keywords) == ("", "Ein Bild.", ["Bild"])


def test_prompt_includes_context_only_when_enabled(tmp_path, monkeypatch):
    path = tmp_path / "a.jpg"
    monkeypatch.setattr(dl.config, "IMAGE_VISION_CONTEXT", True)
    assert "Place: Wien" in dl._build_vision_prompt(path, "Place: Wien")
    monkeypatch.setattr(dl.config, "IMAGE_VISION_CONTEXT", False)
    assert "Wien" not in dl._build_vision_prompt(path, "Place: Wien")


def _fake_vision_post(monkeypatch, message, finish_reason="stop"):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": message, "finish_reason": finish_reason}]}

    monkeypatch.setattr(dl.config, "IMAGE_VISION_BASE_URL", "http://localhost:1234")
    monkeypatch.setattr(dl.config, "IMAGE_VISION_MODEL", "m")
    monkeypatch.setattr(dl.requests, "post", lambda _u, **k: captured.update(k) or Response())
    return captured


def _small_image(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(path)
    return path


@pytest.mark.parametrize("reproducible,expected", [(True, {"temperature": 0, "seed": 7}), (False, {"temperature": 0.2})])
def test_vision_follows_reproducible_switch(tmp_path, monkeypatch, reproducible, expected):
    import runtime

    class Settings:
        def get(self):
            return {"llm": {"reproducible": reproducible, "seed": 7}}

    monkeypatch.setattr(runtime, "settings", lambda: Settings())
    captured = _fake_vision_post(monkeypatch, {"content": "ok"})

    dl._summarize_image_with_llm(_small_image(tmp_path))

    payload = captured["json"]
    assert {k: payload[k] for k in expected} == expected
    assert ("seed" in payload) == reproducible


def test_thinking_model_out_of_tokens_gives_helpful_error(tmp_path, monkeypatch):
    _fake_vision_post(monkeypatch, {"content": "", "reasoning_content": "hmm"}, finish_reason="length")
    with pytest.raises(RuntimeError, match="IMAGE_VISION_MAX_TOKENS"):
        dl._summarize_image_with_llm(_small_image(tmp_path))


def test_think_blocks_are_stripped(tmp_path, monkeypatch):
    _fake_vision_post(monkeypatch, {"content": "<think>reasoning</think>\nHeadline: Ein Hund"})
    assert dl._summarize_image_with_llm(_small_image(tmp_path)) == "Headline: Ein Hund"


def test_fingerprint_changes_with_model(monkeypatch):
    monkeypatch.setattr(dl.config, "IMAGE_VISION_MODEL", "a")
    first = dl.image_analysis_fingerprint()
    monkeypatch.setattr(dl.config, "IMAGE_VISION_MODEL", "b")
    assert dl.image_analysis_fingerprint() != first
