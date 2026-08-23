"""
Page-wise structured view of a normalized Document — a pure reshape, not a new
extraction path. Every engine (PyMuPDF/Docling/VisionLLM, and now docx via the
converted-PDF route) already writes into the same Document/DocumentElement/Page
schema regardless of source format, so this same bucketing works identically
for a PDF, a DOCX, or a scanned image: no format-specific branches needed here.

Buckets elements per page by ElementType into text/headings/tables/images/
equations/captions, plus an `other` bucket (running headers/footers, page
numbers, decorative, unknown) so nothing is silently dropped, matching the
"crop, don't drop" principle already used for images/equations elsewhere.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.normalization.schema import BoundingBox, Document, ElementType, TableData

_HEADING_TYPES = {ElementType.HEADING, ElementType.SECTION_HEADING}
_TABLE_TYPES = {ElementType.TABLE}
_IMAGE_TYPES = {ElementType.IMAGE, ElementType.DIAGRAM}
_EQUATION_TYPES = {ElementType.EQUATION}
_CAPTION_TYPES = {ElementType.CAPTION}
_TEXT_TYPES = {
    ElementType.TEXT,
    ElementType.PARAGRAPH,
    ElementType.QUESTION_STEM,
    ElementType.QUESTION_SUBPART,
    ElementType.MCQ_OPTION,
    ElementType.ANSWER,
    ElementType.ANSWER_SUBPART,
    ElementType.EXPLANATION,
    ElementType.RELATED_THEORY,
    ElementType.LEARNING_OBJECTIVE,
}
# RUNNING_HEADER_FOOTER, PAGE_NUMBER, DECORATIVE, UNKNOWN fall through to `other`.


class PageElementView(BaseModel):
    element_id: str
    type: str
    content: str | None = None
    reading_order: int | None = None
    bbox: BoundingBox | None = None
    table_data: TableData | None = None
    asset_path: str | None = None
    latex: str | None = None
    caption: str | None = None


class PageMetadata(BaseModel):
    document_id: str
    page_number: int
    width: float | None = None
    height: float | None = None
    has_native_text: bool = False
    column_count: int | None = None
    image_path: str | None = None
    source_format: str
    parsers: list[str] = Field(default_factory=list)


class PageStructured(BaseModel):
    page: int
    text: list[PageElementView] = Field(default_factory=list)
    headings: list[PageElementView] = Field(default_factory=list)
    tables: list[PageElementView] = Field(default_factory=list)
    images: list[PageElementView] = Field(default_factory=list)
    equations: list[PageElementView] = Field(default_factory=list)
    captions: list[PageElementView] = Field(default_factory=list)
    other: list[PageElementView] = Field(default_factory=list)
    metadata: PageMetadata


class PagewiseDocument(BaseModel):
    document_id: str
    source_path: str
    source_format: str
    pages: list[PageStructured] = Field(default_factory=list)


def _to_view(element_id: str, document: Document) -> PageElementView:
    element = document.elements[element_id]
    bbox = element.provenance[0].bbox if element.provenance else None
    return PageElementView(
        element_id=element.id,
        type=element.type.value,
        content=element.content,
        reading_order=element.reading_order,
        bbox=bbox,
        table_data=element.table_data,
        asset_path=element.asset_path,
        latex=element.latex,
        caption=element.caption,
    )


def document_to_pagewise(document: Document) -> PagewiseDocument:
    pages: list[PageStructured] = []

    for page in document.pages:
        bucketed = PageStructured(
            page=page.page_number,
            metadata=PageMetadata(
                document_id=document.id,
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                has_native_text=page.has_native_text,
                column_count=page.column_count,
                image_path=page.image_path,
                source_format=document.source_format,
                parsers=sorted({
                    prov.parser
                    for element_id in page.element_ids
                    for prov in document.elements[element_id].provenance
                }),
            ),
        )

        for element_id in page.element_ids:
            element_type = document.elements[element_id].type
            view = _to_view(element_id, document)

            if element_type in _HEADING_TYPES:
                bucketed.headings.append(view)
            elif element_type in _TABLE_TYPES:
                bucketed.tables.append(view)
            elif element_type in _IMAGE_TYPES:
                bucketed.images.append(view)
            elif element_type in _EQUATION_TYPES:
                bucketed.equations.append(view)
            elif element_type in _CAPTION_TYPES:
                bucketed.captions.append(view)
            elif element_type in _TEXT_TYPES:
                bucketed.text.append(view)
            else:
                bucketed.other.append(view)

        pages.append(bucketed)

    return PagewiseDocument(
        document_id=document.id,
        source_path=document.source_path,
        source_format=document.source_format,
        pages=pages,
    )
