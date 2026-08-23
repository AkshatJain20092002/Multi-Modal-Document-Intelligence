"""
Deterministic (no-LLM) parser-selection router. File type + native-text availability
decides which extraction engine runs — this is a plain lookup, never a model call,
per the accepted plan: "parser selection is deterministic; only genuinely ambiguous
downstream decisions reach OpenAI."
"""

from __future__ import annotations

import logging
from typing import Literal

from app.ingestion.profiler import DocumentProfile

logger = logging.getLogger(__name__)

ExtractionPath = Literal[
    "pymupdf_first",
    "ocr_layout_first",
    "docling",
    "ocr_chain",
    "fallback",
]


def select_extraction_path(profile: DocumentProfile) -> ExtractionPath:
    if profile.file_type == "pdf":
        # Native-text PDFs no longer get the PyMuPDF fast path: confirmed by
        # direct testing that some PDF fonts encode symbols (e.g. "√") with no
        # valid Unicode mapping, so any text-layer read (PyMuPDF included)
        # silently drops them, and PyMuPDF's own LLM-based table extraction
        # proved unreliable (failed on 3/5 pages of a real test document). All
        # PDFs now go through DoclingEngine with forced full-page OCR
        # (docling_engine.py), which recovers those symbols and gives real
        # element typing (headings/tables/images) that PyMuPDF never provided.
        path: ExtractionPath = "ocr_layout_first"
    elif profile.file_type == "docx":
        # docx has no native page concept — profiler.py already rendered it to a
        # real, paginated PDF (profile.converted_pdf_path) before this runs, so
        # from here it's routed exactly like a PDF, on that converted file.
        path = "ocr_layout_first"
    elif profile.file_type == "pptx":
        path = "docling"
    elif profile.file_type == "image":
        path = "ocr_chain"
    else:
        path = "fallback"

    logger.info(
        "operation.decision op=select_extraction_path file_type=%s has_native_text=%s -> path=%s",
        profile.file_type, profile.has_native_text, path,
    )
    return path
