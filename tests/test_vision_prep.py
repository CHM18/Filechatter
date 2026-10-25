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


def test_fingerprint_changes_with_model(monkeypatch):
    monkeypatch.setattr(dl.config, "IMAGE_VISION_MODEL", "a")
    first = dl.image_analysis_fingerprint()
    monkeypatch.setattr(dl.config, "IMAGE_VISION_MODEL", "b")
    assert dl.image_analysis_fingerprint() != first
