"""Derive searchable keywords from normalized EXIF values (see document_loader._extract_image_exif)."""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

import config

_COMPASS = [
    {"en": "north", "de": "Norden"},
    {"en": "north-east", "de": "Nordosten"},
    {"en": "east", "de": "Osten"},
    {"en": "south-east", "de": "Südosten"},
    {"en": "south", "de": "Süden"},
    {"en": "south-west", "de": "Südwesten"},
    {"en": "west", "de": "Westen"},
    {"en": "north-west", "de": "Nordwesten"},
]
_MONTHS = [
    ("January", "Januar"), ("February", "Februar"), ("March", "März"), ("April", "April"),
    ("May", "Mai"), ("June", "Juni"), ("July", "Juli"), ("August", "August"),
    ("September", "September"), ("October", "Oktober"), ("November", "November"), ("December", "Dezember"),
]
_WEEKDAYS = [
    ("Monday", "Montag"), ("Tuesday", "Dienstag"), ("Wednesday", "Mittwoch"), ("Thursday", "Donnerstag"),
    ("Friday", "Freitag"), ("Saturday", "Samstag"), ("Sunday", "Sonntag"),
]
_SEASONS = {
    "winter": ("winter", "Winter"), "spring": ("spring", "Frühling"),
    "summer": ("summer", "Sommer"), "autumn": ("autumn", "Herbst"),
}
_SEASONS_NORTH = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
_SEASONS_SOUTH = {m: {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}[s]
                  for m, s in _SEASONS_NORTH.items()}


def _t(en: str, de: str) -> dict[str, str]:
    return {"en": en, "de": de}


def _flatten(items: list[Any]) -> list[str]:
    """Expand per-language keywords into the configured languages and drop duplicates."""
    languages = [lang for lang in config.IMAGE_KEYWORD_LANGUAGES if lang in {"en", "de"}] or ["en"]
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        texts = [item] if isinstance(item, str) else [item.get(lang) for lang in languages]
        for text in texts:
            if text and text.casefold() not in seen:
                seen.add(text.casefold())
                result.append(text)
    return result

# (substring, label); first match wins, so more specific names come first.
_EDITING_SOFTWARE = [
    ("photoshop elements", "Adobe Photoshop Elements"),
    ("elements organizer", "Adobe Photoshop Elements"),
    ("photoshop", "Adobe Photoshop"),
    ("lightroom", "Adobe Lightroom"),
    ("picasa", "Picasa"),
    ("windows photo gallery", "Windows Photo Gallery"),
    ("digital photo professional", "Canon Digital Photo Professional"),
    ("capture one", "Capture One"),
    ("gimp", "GIMP"),
    ("snapseed", "Snapseed"),
    ("coachware", "COACHWare"),
]
_SCAN_SOFTWARE = ("ij scan utility",)

_COUNTRIES = {
    "CH": "Switzerland / Schweiz", "DE": "Germany / Deutschland", "AT": "Austria / Österreich",
    "LI": "Liechtenstein", "FR": "France / Frankreich", "IT": "Italy / Italien", "ES": "Spain / Spanien",
    "PT": "Portugal", "GB": "United Kingdom / Großbritannien", "IE": "Ireland / Irland",
    "NL": "Netherlands / Niederlande", "BE": "Belgium / Belgien", "LU": "Luxembourg / Luxemburg",
    "DK": "Denmark / Dänemark", "SE": "Sweden / Schweden", "NO": "Norway / Norwegen",
    "FI": "Finland / Finnland", "IS": "Iceland / Island", "PL": "Poland / Polen",
    "CZ": "Czechia / Tschechien", "SK": "Slovakia / Slowakei", "HU": "Hungary / Ungarn",
    "SI": "Slovenia / Slowenien", "HR": "Croatia / Kroatien", "GR": "Greece / Griechenland",
    "TR": "Turkey / Türkei", "US": "United States / USA", "CA": "Canada / Kanada",
    "MX": "Mexico / Mexiko", "BR": "Brazil / Brasilien", "AR": "Argentina / Argentinien",
    "CL": "Chile", "PE": "Peru", "CN": "China", "JP": "Japan", "KR": "South Korea / Südkorea",
    "TH": "Thailand", "VN": "Vietnam", "IN": "India / Indien", "AU": "Australia / Australien",
    "NZ": "New Zealand / Neuseeland", "ZA": "South Africa / Südafrika", "EG": "Egypt / Ägypten",
    "MA": "Morocco / Marokko", "AE": "United Arab Emirates / Vereinigte Arabische Emirate",
}

_MAX_CITY_DISTANCE_KM = 25.0
_geocoder: Any = None
_geocoder_failed = False


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
        if match and float(match.group(2)):
            return float(match.group(1)) / float(match.group(2))
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _signed_coordinate(exif: dict[str, Any], name: str, negative_ref: str) -> float | None:
    value = _number(exif.get(f"GPS:{name}"))
    if value is None:
        return None
    ref = str(exif.get(f"GPS:{name}Ref", "")).strip().upper()
    return -value if ref == negative_ref else value


def gps_position(exif: dict[str, Any]) -> tuple[float, float] | None:
    latitude = _signed_coordinate(exif, "GPSLatitude", "S")
    longitude = _signed_coordinate(exif, "GPSLongitude", "W")
    if latitude is None or longitude is None:
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    if latitude == 0 and longitude == 0:  # cameras without a fix often write 0/0
        return None
    return latitude, longitude


def capture_datetime(exif: dict[str, Any]) -> datetime | None:
    """Local capture time incl. sub-seconds, usable for ordering burst images."""
    raw = str(exif.get("DateTimeOriginal", "")).strip()
    try:
        moment = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    subsec = re.sub(r"\D", "", str(exif.get("SubsecTimeOriginal", "")))[:6]
    if subsec:
        moment = moment.replace(microsecond=int(subsec.ljust(6, "0")))
    return moment


def _format_hint(exif: dict[str, Any]) -> list[Any]:
    width, height = _number(exif.get("ExifImageWidth")), _number(exif.get("ExifImageHeight"))
    if not width or not height:
        return []
    orientation = _number(exif.get("Orientation"))
    if orientation is not None and 5 <= orientation <= 8:
        width, height = height, width
    if abs(width / height - 1) <= 0.05:
        return [_t("square format", "Quadratisches Format")]
    if width > height:
        return [_t("landscape format", "Querformat")]
    return [_t("portrait format", "Hochformat")]


def _direction(exif: dict[str, Any]) -> list[Any]:
    degrees = _number(exif.get("GPS:GPSImgDirection"))
    if degrees is None or not 0 <= degrees <= 360:
        return []
    point = _COMPASS[int((degrees + 22.5) // 45) % 8]
    return [_t(f"camera facing {point['en']}", f"Kamera blickt nach {point['de']}")]


def _date_keywords(exif: dict[str, Any], latitude: float | None) -> list[Any]:
    moment = capture_datetime(exif)
    if moment is None:
        return []
    seasons = _SEASONS_SOUTH if latitude is not None and latitude < 0 else _SEASONS_NORTH
    if moment.hour < 5 or moment.hour >= 21:
        part_of_day = _t("night", "Nacht")
    elif moment.hour < 12:
        part_of_day = _t("morning", "Morgen")
    elif moment.hour < 17:
        part_of_day = _t("afternoon", "Nachmittag")
    else:
        part_of_day = _t("evening", "Abend")
    return [
        str(moment.year),
        _t(*_MONTHS[moment.month - 1]),
        _t(*_SEASONS[seasons[moment.month]]),
        _t(*_WEEKDAYS[moment.weekday()]),
        part_of_day,
    ]


def _software_keywords(exif: dict[str, Any]) -> list[Any]:
    software = str(exif.get("Software", "")).casefold()
    if not software:
        return []
    if any(token in software for token in _SCAN_SOFTWARE):
        return [_t("scanned", "gescannt")]
    for token, label in _EDITING_SOFTWARE:
        if token in software:
            return [_t("edited", "bearbeitet"), _t(f"edited with {label}", f"bearbeitet mit {label}")]
    return []


def _equipment_keywords(exif: dict[str, Any]) -> list[Any]:
    keywords: list[Any] = []
    make, model = str(exif.get("Make", "")).strip(), str(exif.get("Model", "")).strip()
    if model:
        keywords.append(model if model.casefold().startswith(make.casefold()) else f"{make} {model}".strip())
    elif make:
        keywords.append(make)
    lens = str(exif.get("LensModel", "")).strip()
    if lens:
        keywords.append(lens)
    return keywords


def _shooting_keywords(exif: dict[str, Any]) -> list[Any]:
    keywords: list[Any] = []
    focal = _number(exif.get("FocalLengthIn35mmFilm"))
    if focal:
        if focal < 35:
            keywords.append(_t("wide angle", "Weitwinkel"))
        elif focal > 200:
            keywords.append(_t("super telephoto", "Super-Tele"))
        elif focal > 85:
            keywords.append(_t("telephoto", "Teleobjektiv"))
    aperture = _number(exif.get("FNumber"))
    if aperture and aperture <= 2.8:
        keywords.append(_t("shallow depth of field", "geringe Schärfentiefe"))
    exposure = _number(exif.get("ExposureTime"))
    if exposure and exposure >= 1 / 30:
        keywords.append(_t("long exposure", "Langzeitbelichtung"))
    iso = _number(exif.get("ISOSpeedRatings"))
    if iso and iso >= 1600:
        keywords.append(_t("low light", "wenig Licht"))
    if str(exif.get("Flash", "")) == "flash fired":
        keywords.append(_t("with flash", "mit Blitz"))
    if exif.get("SceneCaptureType") == "night scene":
        keywords.append(_t("night scene", "Nachtaufnahme"))
    if exif.get("SubjectDistanceRange") == "macro":
        keywords.append(_t("macro", "Makro"))
    if "manual" in (str(exif.get("ExposureProgram", "")), str(exif.get("ExposureMode", ""))) or \
            exif.get("ExposureMode") == "manual exposure":
        keywords.append(_t("manual exposure", "manuelle Belichtung"))
    if exif.get("ExposureMode") == "auto bracket":
        keywords.append(_t("exposure bracketing", "Belichtungsreihe"))
    return keywords


def _altitude_keywords(exif: dict[str, Any]) -> list[Any]:
    altitude = _number(exif.get("GPS:GPSAltitude"))
    if altitude is None:
        return []
    if _number(exif.get("GPS:GPSAltitudeRef")) == 1:
        altitude = -altitude
    if altitude < -1:
        return [_t("below sea level", "unter dem Meeresspiegel")]
    return [_t("high altitude", "große Höhe")] if altitude >= 1500 else []


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((phi2 - phi1) / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def locate(latitude: float, longitude: float) -> list[str]:
    """Offline reverse geocoding (GeoNames cities via the optional 'reverse_geocoder' package)."""
    global _geocoder, _geocoder_failed
    if _geocoder_failed:
        return []
    if _geocoder is None:
        try:
            import reverse_geocoder
        except ImportError:
            _geocoder_failed = True
            return []
        _geocoder = reverse_geocoder
    try:
        hit = _geocoder.search([(latitude, longitude)], mode=1)[0]
    except Exception:
        return []

    keywords: list[Any] = []
    try:
        near = _distance_km(latitude, longitude, float(hit["lat"]), float(hit["lon"])) <= _MAX_CITY_DISTANCE_KM
    except (KeyError, TypeError, ValueError):
        near = False
    if near and hit.get("name"):
        keywords.append(hit["name"])
    for key in ("admin2", "admin1"):
        if hit.get(key):
            keywords.append(hit[key])
    country = str(hit.get("cc", "")).upper()
    if country in _COUNTRIES:
        names = _COUNTRIES[country].split(" / ")
        keywords.append(_t(names[0], names[-1]))
    elif country:
        keywords.append(country)
    return _flatten(keywords)


def derive_exif_keywords(exif: dict[str, Any]) -> list[str]:
    """Return de-duplicated keywords (in IMAGE_KEYWORD_LANGUAGES) derived from the EXIF values present."""
    if not exif:
        return []
    position = gps_position(exif)
    keywords: list[Any] = [
        *_format_hint(exif),
        *_date_keywords(exif, position[0] if position else None),
        *_direction(exif),
        *_software_keywords(exif),
        *_equipment_keywords(exif),
        *_shooting_keywords(exif),
        *_altitude_keywords(exif),
    ]
    if position and config.IMAGE_EXIF_GEOCODE:
        keywords.extend(locate(*position))
    return _flatten(keywords)
