"""
Cheap first-pass document profiling (PyMuPDF only — no heavy parser touched yet).
This is the "is a text layer available?" cost-control branch: the profile decides
which extraction engine the deterministic router sends the document to, before
any expensive parsing happens.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from app.config.settings import settings
from app.ingestion.docx_converter import convert_docx_to_pdf
from app.ingestion.file_detector import FileFormat, detect_file_format
from app.observability.operation_logger import log_operation

try:
    # import fitz  # PyMuPDF
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    fitz = None


class DocumentProfile(BaseModel):
    source_path: str
    file_type: FileFormat
    page_count: int = 0
    has_native_text: bool = False
    avg_chars_per_page: float = 0.0
    likely_scanned: bool = False
    converted_pdf_path: str | None = None  # docx only — set once it's rendered to a real, paginated PDF


def profile_document(path: Path) -> DocumentProfile:
    with log_operation("profile_document", path=path.name):
        file_type = detect_file_format(path)

        if file_type == "pdf":
            return _profile_pdf(path)

        if file_type == "image":
            return DocumentProfile(
                source_path=str(path),
                file_type="image",
                page_count=1,
                has_native_text=False,
                likely_scanned=True,
            )

        if file_type == "docx":
            return _profile_docx(path)

        if file_type == "pptx":
            return DocumentProfile(source_path=str(path), file_type=file_type, has_native_text=True)

        return DocumentProfile(source_path=str(path), file_type="unknown")


def _profile_docx(path: Path) -> DocumentProfile:
    """DOCX has no page concept until rendered (see docx_converter.py docstring) —
    render to PDF first, then reuse the real PDF profiler for page_count/
    has_native_text instead of guessing. converted_pdf_path is kept on the
    profile so a later extraction step can reuse it instead of re-converting."""
    converted_dir = settings.data_dir / "converted"
    pdf_path = convert_docx_to_pdf(path, converted_dir)
    pdf_profile = _profile_pdf(pdf_path)
    return DocumentProfile(
        source_path=str(path),
        file_type="docx",
        page_count=pdf_profile.page_count,
        has_native_text=pdf_profile.has_native_text,
        avg_chars_per_page=pdf_profile.avg_chars_per_page,
        likely_scanned=pdf_profile.likely_scanned,
        converted_pdf_path=str(pdf_path),
    )


def _profile_pdf(path: Path) -> DocumentProfile:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF profiling but is not installed.")

    doc = fitz.open(str(path))
    try:
        page_count = doc.page_count
        char_counts = [len(page.get_text("text")) for page in doc]
    finally:
        doc.close()

    total_chars = sum(char_counts)
    avg_chars = total_chars / page_count if page_count else 0.0
    # Heuristic: a genuine native text layer produces meaningfully more than
    # a stray watermark/footer per page.
    has_native_text = avg_chars > 20

    return DocumentProfile(
        source_path=str(path),
        file_type="pdf",
        page_count=page_count,
        has_native_text=has_native_text,
        avg_chars_per_page=avg_chars,
        likely_scanned=not has_native_text,
    )
