"""
DOCX -> PDF conversion, OS-dispatched.

DOCX has no fixed page concept until something actually paginates it (Word's
live layout engine, at print/render time). Docling's native DOCX handler
never renders/paginates (confirmed empirically: doc.pages == {} and every
item's .prov == [] for docx input) — the only way to get real page
boundaries is to render the DOCX to a paginated PDF first.

Windows (dev): docx2pdf, which drives a real installed/licensed MS Word via
COM. Linux (deploy target, e.g. AWS): LibreOffice headless, which needs no
license and is the standard cross-platform renderer for this exact step.
Each LibreOffice call gets an isolated user-profile dir so concurrent
conversions (multiple workers) don't collide/lock on a shared profile.
"""

from __future__ import annotations

import platform
import subprocess
import tempfile
import uuid
from pathlib import Path

DOCX_TO_PDF_TIMEOUT_SECONDS = 120.0


def convert_docx_to_pdf(docx_path: Path, output_dir: Path) -> Path:
    """Render docx_path to a PDF inside output_dir and return the PDF's path."""
    docx_path = Path(docx_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (docx_path.stem + ".pdf")

    current_os = platform.system()
    if current_os == "Windows":
        _convert_windows(docx_path, output_dir)
    elif current_os == "Linux":
        _convert_linux(docx_path, output_dir)
    else:
        raise OSError(
            f"DOCX->PDF conversion is not supported on {current_os!r} (Windows/Linux only)."
        )

    if not out_path.exists():
        raise RuntimeError(f"DOCX->PDF conversion did not produce the expected output: {out_path}")
    return out_path


def _convert_windows(docx_path: Path, output_dir: Path) -> None:
    try:
        from docx2pdf import convert
    except ImportError as exc:
        raise RuntimeError(
            "docx2pdf is required for DOCX->PDF conversion on Windows "
            "(pip install docx2pdf) and needs a licensed MS Word install."
        ) from exc
    convert(str(docx_path), str(output_dir))


def _convert_linux(docx_path: Path, output_dir: Path) -> None:
    profile_dir = Path(tempfile.gettempdir()) / f"lo_profile_{uuid.uuid4().hex}"
    cmd = [
        "libreoffice",
        "--headless",
        f"--env:UserInstallation=file://{profile_dir.as_posix()}",
        "--convert-to", "pdf",
        "--outdir", str(output_dir),
        str(docx_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=DOCX_TO_PDF_TIMEOUT_SECONDS, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "LibreOffice ('libreoffice'/'soffice') is required for DOCX->PDF conversion on "
            "Linux and was not found on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"LibreOffice conversion of {docx_path.name} exceeded {DOCX_TO_PDF_TIMEOUT_SECONDS:.0f}s."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"LibreOffice conversion of {docx_path.name} failed: {stderr}") from exc
