"""
PyMuPDF extraction engine — the cheap, fast, native-text path. Used when the
profiler found a genuine text layer, so no OCR/heavy layout model is needed.

Column detection is a simple x-midpoint heuristic (not a real layout model);
Docling/Marker are the engines responsible for hard multi-column cases — this
engine intentionally stays cheap and is only routed to when the profile says
the page is straightforward.

Image and table handling here port the metadata-chunking approach from the
futureInnovationChatbot reference repo (metadataChunking/), adapted to this
repo's local-storage/schema/LLM-router conventions instead of its S3 + raw
OpenAI client usage:
- Images: crop bytes saved to disk (asset_path) with tiny-noise filtering and
  "Figure N"/"Fig. N" caption matching, instead of bbox-only placeholders.
- Tables: whole-document LLM extraction with cross-page continuation merge
  (Tier 1, only runs when OPENAI_API_KEY is configured — skipped otherwise,
  consistent with milestone 1 not requiring an LLM key to run at all).
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import settings
from app.extraction.base import ExtractionEngine
from app.extraction.image_extractor import ImageExtractor
from app.extraction.table_extractor import TableExtractor
from app.normalization.schema import (
    BoundingBox,
    Document,
    DocumentElement,
    ElementType,
    Page,
    Provenance,
    TableData,
)

try:
    # import fitz  # PyMuPDF
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    fitz = None

logger = logging.getLogger(__name__)


class PyMuPDFEngine(ExtractionEngine):
    name = "pymupdf"

    def __init__(self) -> None:
        self.image_extractor = ImageExtractor()

    def extract(self, path: Path, document_id: str) -> Document:
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is not installed.")

        pdf = fitz.open(str(path))
        document = Document(id=document_id, source_path=str(path), source_format="pdf")
        assets_dir = settings.data_dir / "assets" / document_id

        try:
            for page_index in range(pdf.page_count):
                pdf_page = pdf[page_index]
                page_number = page_index + 1
                page = Page(
                    page_number=page_number,
                    width=pdf_page.rect.width,
                    height=pdf_page.rect.height,
                    has_native_text=True,
                )

                blocks = pdf_page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
                ordered_blocks = self._reading_order(blocks, page.width or 0)

                for order_index, block in enumerate(ordered_blocks):
                    x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                    text = text.strip()
                    if not text:
                        continue
                    element = DocumentElement(
                        type=ElementType.TEXT,
                        content=text,
                        reading_order=order_index,
                        provenance=[
                            Provenance(
                                document_id=document_id,
                                page_number=page_number,
                                bbox=BoundingBox(x1=x0, y1=y0, x2=x1, y2=y1),
                                parser=self.name,
                                element_id="",  # filled after element id is known
                            )
                        ],
                    )
                    element.provenance[0].element_id = element.id
                    document.add_element(element)
                    page.element_ids.append(element.id)

                # Images: crop saved to disk (asset_path) + "Figure N" caption matching,
                # with tiny-bbox/icon noise filtered out.
                page_images = self.image_extractor.extract_page_images(
                    pdf, pdf_page, page_number, assets_dir
                )
                for image_index, img in enumerate(page_images):
                    x0, y0, x1, y1 = img["bbox"]
                    element = DocumentElement(
                        id=img["element_id"],
                        type=ElementType.IMAGE,
                        asset_path=img["asset_path"],
                        caption=img["caption"],
                        reading_order=len(ordered_blocks) + image_index,
                        provenance=[
                            Provenance(
                                document_id=document_id,
                                page_number=page_number,
                                bbox=BoundingBox(x1=x0, y1=y0, x2=x1, y2=y1),
                                parser=self.name,
                                element_id=img["element_id"],
                            )
                        ],
                    )
                    document.add_element(element)
                    page.element_ids.append(element.id)

                document.pages.append(page)

            self._extract_tables(pdf, document)
        finally:
            pdf.close()

        return document

    def _extract_tables(self, pdf, document: Document) -> None:
        """Whole-document LLM table extraction with cross-page merge. Skipped
        entirely (not an error) when no OpenAI key is configured — table
        extraction is a Tier 1 escalation, not a requirement to run milestone 1."""
        if not settings.openai_api_key:
            logger.info("table_extraction.skipped reason=no_openai_api_key")
            return

        pages_by_number = {page.page_number: page for page in document.pages}
        tables = TableExtractor().extract(pdf)
        for table in tables:
            page_start = table["page_start"]
            page = pages_by_number.get(page_start)
            if page is None:
                continue
            element = DocumentElement(
                type=ElementType.TABLE,
                table_data=TableData(headers=table["headers"], rows=table["rows"]),
                caption=table["caption"],
                reading_order=len(page.element_ids),
                provenance=[
                    Provenance(
                        document_id=document.id,
                        page_number=page_start,
                        parser=self.name,
                        element_id="",
                    )
                ],
            )
            element.provenance[0].element_id = element.id
            document.add_element(element)
            page.element_ids.append(element.id)

    @staticmethod
    def _reading_order(blocks: list, page_width: float) -> list:
        """Two-column heuristic: split blocks left/right of the page midpoint,
        sort each column top-to-bottom, then concatenate left-column-first.
        Falls back to a single top-to-bottom sort when there's no clear split.
        """
        if not blocks or not page_width:
            return sorted(blocks, key=lambda b: (b[1], b[0]))

        midpoint = page_width / 2
        left = [b for b in blocks if ((b[0] + b[2]) / 2) < midpoint]
        right = [b for b in blocks if ((b[0] + b[2]) / 2) >= midpoint]

        # If the "right column" is basically the whole page (single column doc),
        # treat it as one column instead.
        if len(left) < 2 or len(right) < 2:
            return sorted(blocks, key=lambda b: (b[1], b[0]))

        left.sort(key=lambda b: b[1])
        right.sort(key=lambda b: b[1])
        return left + right
