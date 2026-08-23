"""
Docling extraction engine — the canonical structural parser per the accepted plan.
Handles PDF/DOCX/PPTX/image inputs uniformly and produces a DoclingDocument (already
a Pydantic representation with text/tables/pictures/hierarchy/layout/provenance),
which docling_convert.py maps into our own normalized schema.

OCR backend is configurable and swappable without touching any calling code —
this is what lets OCRChainEngine try RapidOCR first, then EasyOCR, without
duplicating the Docling wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from app.config.settings import settings
from app.extraction.base import ExtractionEngine
from app.extraction.docling_convert import docling_document_to_normalized
from app.ingestion.file_detector import detect_file_format
from app.normalization.schema import Document

OcrBackend = Literal["rapidocr", "easyocr"]


class DoclingEngine(ExtractionEngine):
    def __init__(self, ocr_backend: OcrBackend = "rapidocr") -> None:
        self.ocr_backend = ocr_backend
        self.name = f"docling[{ocr_backend}]"

    def extract(self, path: Path, document_id: str) -> Document:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import (
            DocumentConverter,
            ImageFormatOption,
            PdfFormatOption,
        )

        file_format = detect_file_format(path)
        # NOTE: only pdf/image go through PdfPipelineOptions (OCR + page/picture
        # image generation) — docx/pptx are parsed natively by Docling from their
        # XML content, no OCR or page-rasterization involved, so they get no
        # format_options override and just use Docling's own defaults for that
        # format. Previously this branch routed anything non-image through
        # InputFormat.PDF/PdfFormatOption unconditionally, which would silently
        # try to parse a .docx as a PDF and break.
        format_options: dict = {}
        if file_format in ("pdf", "image"):
            ocr_options = RapidOcrOptions() if self.ocr_backend == "rapidocr" else EasyOcrOptions()
            pipeline_options = PdfPipelineOptions()
            pipeline_options.ocr_options = ocr_options
            # Without these, Docling silently emits picture/formula items with no
            # image data, which the converter then has to drop entirely. This is
            # pure geometric cropping (page image -> bbox crop), no model
            # inference — deliberately NOT using do_formula_enrichment, which
            # pulls in a VLM (CodeFormulaV2) and is far too slow on CPU for a
            # per-page default. LaTeX transcription, if ever needed, should be a
            # cheap on-demand OpenAI vision call per cropped formula later, not a
            # bulk local model run here — see app/extraction/vision_llm_engine.py
            # for the existing tiered-OpenAI pattern this would follow.
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True

            input_format = InputFormat.IMAGE if file_format == "image" else InputFormat.PDF
            format_option_cls = ImageFormatOption if file_format == "image" else PdfFormatOption
            format_options = {input_format: format_option_cls(pipeline_options=pipeline_options)}
        elif file_format == "docx":
            format_options = {}  # Docling defaults handle docx natively
        elif file_format == "pptx":
            format_options = {}  # Docling defaults handle pptx natively
        else:
            raise ValueError(f"DoclingEngine cannot handle file format '{file_format}' for {path}")

        converter = DocumentConverter(format_options=format_options)
        result = converter.convert(str(path))

        assets_dir = settings.data_dir / "assets" / document_id
        return docling_document_to_normalized(
            result.document,
            document_id=document_id,
            source_path=str(path),
            source_format=file_format,
            parser_name=self.name,
            assets_dir=assets_dir,
        )
