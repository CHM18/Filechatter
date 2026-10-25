"""Tests for xmp_iptc and its integration into the image reader."""
from __future__ import annotations

import struct

import pytest
from PIL import Image

import document_loader as dl
import xmp_iptc


def _xmp(**parts: str) -> bytes:
    body = "".join(parts.values())
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        'xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/">'
        f'<rdf:Description xmp:CreatorTool="Adobe Photoshop Lightroom 6.8 (Windows)">{body}</rdf:Description>'
        "</rdf:RDF></x:xmpmeta>"
    ).encode("utf-8")


TITLE = '<dc:title><rdf:Alt><rdf:li xml:lang="x-default">Fledermaus im Keller</rdf:li></rdf:Alt></dc:title>'
SUBJECT = "<dc:subject><rdf:Bag><rdf:li>Fledermaus</rdf:li><rdf:li>Keller</rdf:li></rdf:Bag></dc:subject>"
DESC = '<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{}</rdf:li></rdf:Alt></dc:description>'
REGIONS = (
    '<mwg-rs:Regions><mwg-rs:RegionList><rdf:Bag>'
    '<rdf:li mwg-rs:Name="Anna" mwg-rs:Type="Face"/></rdf:Bag></mwg-rs:RegionList></mwg-rs:Regions>'
)


def _iptc_jpeg(path, records: dict[int, list[bytes]], xmp: bytes | None = None) -> None:
    Image.new("RGB", (8, 8)).save(path, xmp=xmp) if xmp else Image.new("RGB", (8, 8)).save(path)
    data = b"".join(
        b"\x1c\x02" + bytes([dataset]) + struct.pack(">H", len(value)) + value
        for dataset, values in records.items() for value in values
    )
    if len(data) % 2:
        data += b"\x00"
    block = b"8BIM" + b"\x04\x04" + b"\x00\x00" + struct.pack(">I", len(data)) + data
    payload = b"Photoshop 3.0\x00" + block
    segment = b"\xff\xed" + struct.pack(">H", len(payload) + 2) + payload
    raw = path.read_bytes()
    path.write_bytes(raw[:2] + segment + raw[2:])


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setattr(xmp_iptc.config, "IMAGE_XMP_FIELDS", ["Title", "Description", "Keywords", "People", "City", "Country", "CreatorTool"])
    monkeypatch.setattr(xmp_iptc.config, "IMAGE_METADATA_IGNORE_VALUES", ["My beautiful picture", "Version 2", "Anafi 1.4.1", "MINOLTA DIGITAL CAMERA"])


def test_xmp_title_description_keywords_people_and_creator_tool(tmp_path):
    path = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(path, xmp=_xmp(t=TITLE, s=SUBJECT, d=DESC.format("Eine Zwergfledermaus"), r=REGIONS))
    assert xmp_iptc.extract_xmp_iptc(path) == {
        "title": "Fledermaus im Keller",
        "description": "Eine Zwergfledermaus",
        "keywords": ["Fledermaus", "Keller"],
        "people": ["Anna"],
        "creator_tool": "Adobe Photoshop Lightroom 6.8 (Windows)",
    }


def test_iptc_is_read_and_merged_with_xmp_without_duplicates(tmp_path):
    path = tmp_path / "b.jpg"
    _iptc_jpeg(
        path,
        {5: ["fledermaus im keller".encode()], 25: ["Keller".encode(), "Höhle".encode()],
         120: ["Eine Zwergfledermaus".encode()], 90: ["Zürich".encode()], 101: ["Schweiz".encode()]},
        xmp=_xmp(t=TITLE, s=SUBJECT),
    )
    result = xmp_iptc.extract_xmp_iptc(path)
    assert result["title"] == "Fledermaus im Keller"
    assert result["keywords"] == ["Fledermaus", "Keller", "Höhle"]
    assert result["description"] == "Eine Zwergfledermaus"
    assert (result["city"], result["country"]) == ("Zürich", "Schweiz")


def test_iptc_latin1_is_decoded(tmp_path):
    path = tmp_path / "c.jpg"
    _iptc_jpeg(path, {25: ["Höhle".encode("cp1252")]})
    assert xmp_iptc.extract_xmp_iptc(path)["keywords"] == ["Höhle"]


@pytest.mark.parametrize("junk", ["My beautiful picture", "version 2", "Anafi 1.4.1", "MINOLTA DIGITAL CAMERA ",
                                  "IMG_1234.jpg", "20190"])
def test_junk_descriptions_are_dropped(tmp_path, junk):
    path = tmp_path / "d.jpg"
    Image.new("RGB", (8, 8)).save(path, xmp=_xmp(d=DESC.format(junk)))
    assert "description" not in xmp_iptc.extract_xmp_iptc(path)


def test_control_characters_are_removed(tmp_path):
    path = tmp_path / "e.jpg"
    _iptc_jpeg(path, {25: [b"Berg\x00"]})
    assert xmp_iptc.extract_xmp_iptc(path)["keywords"] == ["Berg"]


def test_disabled_without_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(xmp_iptc.config, "IMAGE_XMP_FIELDS", [])
    path = tmp_path / "f.jpg"
    Image.new("RGB", (8, 8)).save(path, xmp=_xmp(t=TITLE))
    assert xmp_iptc.extract_xmp_iptc(path) == {}


def test_broken_file_gives_empty_result(tmp_path):
    path = tmp_path / "g.jpg"
    path.write_bytes(b"not an image")
    assert xmp_iptc.extract_xmp_iptc(path) == {}


@pytest.mark.parametrize(
    "description, kept",
    [
        ("MINOLTA DIGITAL CAMERA", False),
        ("Eine Zwergfledermaus", False),  # same as XMP description
        ("Eine Zwergfledermaus im Keller", True),  # longer than the XMP description: keeps the extra detail
        ("Anderer Text", True),
    ],
)
def test_exif_image_description_deduplicated(description, kept):
    exif = {"ImageDescription": description, "Make": "Canon"}
    result = xmp_iptc.deduplicate_against({"title": "Fledermaus im Keller", "description": "Eine Zwergfledermaus"}, exif)
    assert ("ImageDescription" in result) is kept
    assert result["Make"] == "Canon"


def test_dedupe_keywords_across_groups():
    assert xmp_iptc.dedupe_keywords(["Sommer", "Berg"], ["sommer", "2020"], ["BERG", "See"]) == [
        ["Sommer", "Berg"], ["2020"], ["See"],
    ]


def test_read_image_integrates_and_deduplicates(tmp_path, monkeypatch):
    path = tmp_path / "h.jpg"
    Image.new("RGB", (8, 8)).save(path, xmp=_xmp(t=TITLE, s=SUBJECT, d=DESC.format("Eine Zwergfledermaus"), r=REGIONS))
    monkeypatch.setattr(dl, "_summarize_image_with_llm", lambda _p, **_kw: "Description: Eine Fledermaus.\nSearchable keywords: keller, Nacht")
    monkeypatch.setattr(dl, "_extract_image_exif", lambda _p: {"ImageDescription": "Eine Zwergfledermaus", "Software": "NIKON D850 Ver.1.20"})
    monkeypatch.setattr(dl.config, "IMAGE_EXIF_FIELDS", ["ImageDescription", "Software"])
    monkeypatch.setattr(dl.config, "IMAGE_EXIF_DERIVE_KEYWORDS", True)
    monkeypatch.setattr(dl.config, "IMAGE_KEYWORD_LANGUAGES", ["en"])

    text, metadata = dl._read_image(path)

    assert "Title: Fledermaus im Keller" in text
    assert "Description: Eine Zwergfledermaus" in text
    assert "Tags: Fledermaus, Keller" in text
    assert "People: Anna" in text
    assert "- ImageDescription" not in text
    assert "Searchable keywords: Nacht" in text
    assert "edited with Adobe Lightroom" in text
    assert metadata["image_people"] == ["Anna"]
    assert metadata["image_user_keywords"] == ["Fledermaus", "Keller"]
