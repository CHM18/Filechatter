#!/usr/bin/env python3
"""
Analysiert die Exif-Tags aller Fotos in einem Verzeichnis (inkl. Unterverzeichnisse)
und schreibt eine CSV-Liste aller vorhandenen, nicht-leeren Tags.

Pro Zeile: Tag-Name; Anzahl Fotos; Wert 1; Wert 2; ... (max. 25 verschiedene Werte der ersten Treffer)

Installation:  pip install pillow
Optional (HEIC/HEIF): pip install pillow-heif

Aufruf:
    python exif_tag_analyse.py /pfad/zu/fotos
    python exif_tag_analyse.py /pfad/zu/fotos -o ergebnis.csv --max-values 25 --delimiter ,
"""

import argparse
import csv
import os
import sys
import warnings
from collections import OrderedDict

from PIL import ExifTags, Image, UnidentifiedImageError

# Es werden nur Metadaten gelesen, keine Pixel -> Pixel-Limit ist hier unkritisch.
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
# Harmlose Warnungen bei herstellerspezifisch fehlerhaften Tags (z. B. IPTC Tag 33723)
warnings.filterwarnings("ignore", message="Metadata Warning")

HEIF_AVAILABLE = False
try:  # HEIC/HEIF-Unterstützung: pip install pillow-heif
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    pass

EXTENSIONS = {
    ".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp", ".heic", ".heif",
}

MAX_VALUE_LEN = 300  # überlange Werte (z. B. MakerNote) werden gekürzt


def is_empty(text: str) -> bool:
    """Leer = nur Whitespace / Null-Bytes / leere Klammern."""
    return text.replace("\x00", "").strip() in ("", "()", "[]", "{}", "b''")


def value_to_str(value) -> str:
    """Wandelt einen Exif-Wert in einen lesbaren String um."""
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8").replace("\x00", "").strip()
            if text and text.isprintable():
                return text
        except UnicodeDecodeError:
            pass
        return f"<{len(value)} bytes>" if value.strip(b"\x00") else ""
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(value_to_str(v) for v in value) + ")"
    if isinstance(value, dict):
        return str(value)
    text = str(value).replace("\x00", "").strip()
    return text


def read_exif(path: str) -> dict:
    """Liefert {tagname: wertstring} für alle nicht-leeren Tags eines Bildes."""
    result = {}
    with Image.open(path) as img:
        exif = img.getexif()
        if not exif and img.info.get("exif"):
            # Fallback (u. a. HEIC): rohe Exif-Bytes selbst parsen
            exif = Image.Exif()
            exif.load(img.info["exif"])
        if not exif:
            return result

        def add(prefix, tag_dict, tag_id, value):
            name = tag_dict.get(tag_id, f"Unknown_0x{tag_id:04X}")
            text = value_to_str(value)
            if is_empty(text):
                return
            if len(text) > MAX_VALUE_LEN:
                text = text[:MAX_VALUE_LEN] + "…"
            result[prefix + name] = text

        # IFD0 (Hauptbild-Tags; Verweise auf Sub-IFDs überspringen)
        sub_ifd_pointers = {0x8769, 0x8825, 0xA005}
        for tag_id, value in exif.items():
            if tag_id in sub_ifd_pointers:
                continue
            add("", ExifTags.TAGS, tag_id, value)

        # Exif-IFD
        try:
            for tag_id, value in exif.get_ifd(ExifTags.IFD.Exif).items():
                add("", ExifTags.TAGS, tag_id, value)
        except Exception:
            pass

        # GPS-IFD
        try:
            for tag_id, value in exif.get_ifd(ExifTags.IFD.GPSInfo).items():
                add("GPS:", ExifTags.GPSTAGS, tag_id, value)
        except Exception:
            pass

        # Interoperability-IFD
        try:
            for tag_id, value in exif.get_ifd(ExifTags.IFD.Interop).items():
                add("Interop:", ExifTags.INTEROP_TAGS if hasattr(ExifTags, "INTEROP_TAGS")
                    else ExifTags.TAGS, tag_id, value)
        except Exception:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Exif-Tag-Statistik für ein Fotoverzeichnis")
    parser.add_argument("directory", help="Wurzelverzeichnis mit Fotos")
    parser.add_argument("-o", "--output", default="exif_tags.csv", help="Ausgabe-CSV (Standard: exif_tags.csv)")
    parser.add_argument("--max-values", type=int, default=25,
                        help="Max. Anzahl verschiedener Beispielwerte pro Tag (Standard: 25)")
    parser.add_argument("--delimiter", default=";", help="CSV-Trennzeichen (Standard: ';' für deutsches Excel)")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        sys.exit(f"Verzeichnis nicht gefunden: {args.directory}")

    counts = {}                # tag -> Anzahl Fotos
    samples = OrderedDict()    # tag -> Liste der ersten Werte
    n_files = n_errors = n_without_exif = 0
    heif_warned = False

    for root, _dirs, files in os.walk(args.directory):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTENSIONS:
                continue
            if ext in (".heic", ".heif") and not HEIF_AVAILABLE:
                if not heif_warned:
                    print("Warnung: HEIC/HEIF-Dateien gefunden, aber 'pillow-heif' ist nicht "
                          "installiert (pip install pillow-heif). Diese Dateien werden übersprungen.",
                          file=sys.stderr)
                    heif_warned = True
                n_errors += 1
                continue
            path = os.path.join(root, fname)
            n_files += 1
            try:
                tags = read_exif(path)
            except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
                n_errors += 1
                print(f"Warnung: {path} nicht lesbar ({e})", file=sys.stderr)
                continue

            if not tags:
                n_without_exif += 1
                continue

            for tag, value in tags.items():
                counts[tag] = counts.get(tag, 0) + 1
                lst = samples.setdefault(tag, [])
                if len(lst) < args.max_values and value not in lst:
                    lst.append(value)

            if n_files % 500 == 0:
                print(f"... {n_files} Dateien verarbeitet", file=sys.stderr)

    # Sortierung: häufigste Tags zuerst, dann alphabetisch
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=args.delimiter)
        for tag, n in rows:
            writer.writerow([tag, n] + samples[tag])

    print(
        f"Fertig: {n_files} Bilddateien, {n_without_exif} ohne Exif, {n_errors} Fehler, "
        f"{len(rows)} verschiedene Tags -> {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()