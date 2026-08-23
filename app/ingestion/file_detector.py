"""File-type detection. Deliberately dumb — extension + magic-byte sniff, no ML here."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

FileFormat = Literal["pdf", "docx", "pptx", "image", "unknown"]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def detect_file_format(path: Path) -> FileFormat:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix == ".pptx":
        return "pptx"
    if suffix in _IMAGE_EXTS:
        return "image"
    return "unknown"
