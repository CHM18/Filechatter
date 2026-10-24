"""Tests for exif_keywords."""
from __future__ import annotations

from datetime import datetime

import pytest

import exif_keywords as ek


@pytest.fixture(autouse=True)
def no_geocoding(monkeypatch):
    monkeypatch.setattr(ek.config, "IMAGE_EXIF_GEOCODE", False)
    monkeypatch.setattr(ek.config, "IMAGE_KEYWORD_LANGUAGES", ["en"])


def test_keywords_are_bilingual_by_default_languages(monkeypatch):
    monkeypatch.setattr(ek.config, "IMAGE_KEYWORD_LANGUAGES", ["en", "de"])
    keywords = ek.derive_exif_keywords(
        {"DateTimeOriginal": "2020:07:04 08:15:00", "ExifImageWidth": 4000, "ExifImageHeight": 3000,
         "Software": "Picasa", "Model": "X100"}
    )
    assert keywords == [
        "landscape format", "Querformat", "2020", "July", "Juli", "summer", "Sommer", "Saturday", "Samstag",
        "morning", "Morgen", "edited", "bearbeitet", "edited with Picasa", "bearbeitet mit Picasa", "X100",
    ]


def test_german_only_and_unknown_language_fallback(monkeypatch):
    monkeypatch.setattr(ek.config, "IMAGE_KEYWORD_LANGUAGES", ["de"])
    assert "Sommer" in ek.derive_exif_keywords({"DateTimeOriginal": "2020:07:04 08:15:00"})
    monkeypatch.setattr(ek.config, "IMAGE_KEYWORD_LANGUAGES", ["fr"])
    assert "summer" in ek.derive_exif_keywords({"DateTimeOriginal": "2020:07:04 08:15:00"})


def test_empty_exif_gives_no_keywords():
    assert ek.derive_exif_keywords({}) == []


@pytest.mark.parametrize(
    "exif, expected",
    [
        ({"ExifImageWidth": 4000, "ExifImageHeight": 3000}, "landscape format"),
        ({"ExifImageWidth": 3000, "ExifImageHeight": 4000}, "portrait format"),
        ({"ExifImageWidth": 4000, "ExifImageHeight": 3000, "Orientation": 6}, "portrait format"),
        ({"ExifImageWidth": 3000, "ExifImageHeight": 3010}, "square format"),
    ],
)
def test_format_uses_orientation(exif, expected):
    assert expected in ek.derive_exif_keywords(exif)


@pytest.mark.parametrize(
    "degrees, expected",
    [(0, "north"), (44, "north-east"), (180, "south"), (350, "north"), (360, "north")],
)
def test_camera_direction(degrees, expected):
    assert f"camera facing {expected}" in ek.derive_exif_keywords({"GPS:GPSImgDirection": degrees})


def test_date_keywords_and_hemisphere():
    exif = {"DateTimeOriginal": "2020:07:04 08:15:00"}
    assert ek.derive_exif_keywords(exif)[:5] == ["2020", "July", "summer", "Saturday", "morning"]
    south = {**exif, "GPS:GPSLatitude": 33.9, "GPS:GPSLatitudeRef": "S", "GPS:GPSLongitude": 18.4,
             "GPS:GPSLongitudeRef": "E"}
    assert "winter" in ek.derive_exif_keywords(south)


def test_capture_datetime_includes_subseconds():
    moment = ek.capture_datetime({"DateTimeOriginal": "2020:07:04 08:15:00", "SubsecTimeOriginal": "07"})
    assert moment == datetime(2020, 7, 4, 8, 15, 0, 70000)
    assert ek.capture_datetime({"DateTimeOriginal": "garbage"}) is None


def test_gps_position_signs_and_rejects_null_island():
    exif = {"GPS:GPSLatitude": 10.5, "GPS:GPSLatitudeRef": "S", "GPS:GPSLongitude": 20.0, "GPS:GPSLongitudeRef": "W"}
    assert ek.gps_position(exif) == (-10.5, -20.0)
    assert ek.gps_position({"GPS:GPSLatitude": 0, "GPS:GPSLongitude": 0}) is None


@pytest.mark.parametrize(
    "software, expected",
    [
        ("Adobe Photoshop Elements 13.0 (Windows)", ["edited", "edited with Adobe Photoshop Elements"]),
        ("Adobe Photoshop CS2 Macintosh", ["edited", "edited with Adobe Photoshop"]),
        ("Picasa", ["edited", "edited with Picasa"]),
        ("IJ Scan Utility", ["scanned"]),
        ("QuickTime 7.7.3", []),
        ("Photos 3.0", []),
        ("V100-01", []),
    ],
)
def test_software_whitelist(software, expected):
    assert ek.derive_exif_keywords({"Software": software}) == expected


def test_equipment_and_shooting_keywords():
    keywords = ek.derive_exif_keywords(
        {
            "Make": "Canon", "Model": "Canon EOS R5", "LensModel": "RF24-70mm",
            "FocalLengthIn35mmFilm": 24, "FNumber": 2.0, "ExposureTime": "1/10", "ISOSpeedRatings": 3200,
            "Flash": "flash fired", "SubjectDistanceRange": "macro", "ExposureProgram": "manual",
        }
    )
    assert keywords == [
        "Canon EOS R5", "RF24-70mm", "wide angle", "shallow depth of field", "long exposure",
        "low light", "with flash", "macro", "manual exposure",
    ]


def test_altitude_keywords():
    assert "high altitude" in ek.derive_exif_keywords({"GPS:GPSAltitude": 2400, "GPS:GPSAltitudeRef": 0})
    assert "below sea level" in ek.derive_exif_keywords({"GPS:GPSAltitude": 30, "GPS:GPSAltitudeRef": 1})


def test_locate_with_offline_geocoder(monkeypatch):
    pytest.importorskip("reverse_geocoder")
    monkeypatch.setattr(ek.config, "IMAGE_EXIF_GEOCODE", True)
    monkeypatch.setattr(ek.config, "IMAGE_KEYWORD_LANGUAGES", ["en", "de"])
    keywords = ek.derive_exif_keywords(
        {"GPS:GPSLatitude": 48.1374, "GPS:GPSLatitudeRef": "N", "GPS:GPSLongitude": 11.5755,
         "GPS:GPSLongitudeRef": "E"}
    )
    assert "Munich" in keywords
    assert "Bavaria" in keywords
    assert "Germany" in keywords and "Deutschland" in keywords


def test_far_city_is_not_reported(monkeypatch):
    class Fake:
        @staticmethod
        def search(_points, mode=1):
            return [{"lat": "50.0", "lon": "10.0", "name": "Faraway", "admin1": "Region", "admin2": "", "cc": "DE"}]

    monkeypatch.setattr(ek, "_geocoder", Fake)
    monkeypatch.setattr(ek, "_geocoder_failed", False)
    keywords = ek.locate(40.0, 5.0)
    assert "Faraway" not in keywords and "Region" in keywords
