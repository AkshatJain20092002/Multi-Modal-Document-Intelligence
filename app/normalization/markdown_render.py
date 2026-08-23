"""
Page-wise Markdown rendering — a human-readable view derived directly from the
canonical Document (independent of page_view.py's type-bucketed structured
JSON, which loses reading order by design). Per the "JSON is canonical,
Markdown is derived" principle: this is a projection, never a data source —
nothing downstream should read Markdown back into the pipeline.

Element-type -> Markdown mapping:
    HEADING / SECTION_HEADING                    -> "## content"
    TABLE                                        -> Markdown pipe table (table_data)
    IMAGE / DIAGRAM                               -> ![caption](asset_path)
    EQUATION                                      -> $$latex$$, else an image link, else a note
    CAPTION                                       -> *content*
    MCQ_OPTION                                    -> "- content" bullet
    RUNNING_HEADER_FOOTER / PAGE_NUMBER / DECORATIVE -> HTML comment (kept, not shown as
                                                          content — nothing is silently dropped)
    everything else (TEXT/PARAGRAPH/QUESTION_*/ANSWER*/EXPLANATION/
    RELATED_THEORY/LEARNING_OBJECTIVE/UNKNOWN)    -> plain paragraph
"""

from __future__ import annotations

from pathlib import PurePath

from pydantic import BaseModel, Field

from app.normalization.schema import Document, DocumentElement, ElementType, TableData

_SILENT_TYPES = {ElementType.RUNNING_HEADER_FOOTER, ElementType.PAGE_NUMBER, ElementType.DECORATIVE}


class PageMarkdown(BaseModel):
    page_number: int
    markdown: str


class MarkdownDocument(BaseModel):
    document_id: str
    source_path: str
    source_format: str
    pages: list[PageMarkdown] = Field(default_factory=list)

    def combined(self) -> str:
        """Single concatenated Markdown string, page breaks marked with an anchor comment."""
        parts = [f"<!-- page {page.page_number} -->\n\n{page.markdown}" for page in self.pages]
        return "\n\n---\n\n".join(parts)


def _render_table(table_data: TableData) -> str:
    if not table_data.headers and not table_data.rows:
        return ""
    headers = table_data.headers
    if not headers and table_data.rows:
        headers = [f"col_{i}" for i in range(len(table_data.rows[0]))]
    lines: list[str] = []
    if headers:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in table_data.rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _as_link_path(asset_path: str) -> str:
    """Markdown/HTML links need forward slashes regardless of OS — asset_path
    is stored via pathlib with the platform's native separator, so this is
    purely a rendering-time normalization, not a change to the stored path."""
    return PurePath(asset_path).as_posix()


def _render_element(element: DocumentElement) -> str | None:
    if element.type in (ElementType.HEADING, ElementType.SECTION_HEADING):
        return f"## {element.content}" if element.content else None

    if element.type == ElementType.TABLE:
        table_md = _render_table(element.table_data) if element.table_data else ""
        if element.caption:
            return f"*{element.caption}*\n\n{table_md}" if table_md else f"*{element.caption}*"
        return table_md or None

    if element.type in (ElementType.IMAGE, ElementType.DIAGRAM):
        alt = element.caption or element.type.value
        if element.asset_path:
            return f"![{alt}]({_as_link_path(element.asset_path)})"
        return f"*[{alt} — no crop recovered]*"

    if element.type == ElementType.EQUATION:
        if element.latex:
            return f"$$\n{element.latex}\n$$"
        if element.asset_path:
            return f"![equation]({_as_link_path(element.asset_path)})"
        return "*[equation — nothing recovered]*"

    if element.type == ElementType.CAPTION:
        return f"*{element.content}*" if element.content else None

    if element.type == ElementType.MCQ_OPTION:
        return f"- {element.content}" if element.content else None

    if element.type in _SILENT_TYPES:
        preview = (element.content or element.asset_path or "")[:60]
        return f"<!-- {element.type.value}: {preview} -->"

    return element.content or None


def document_to_markdown(document: Document) -> MarkdownDocument:
    pages: list[PageMarkdown] = []
    for page in document.pages:
        blocks: list[str] = []
        for element_id in page.element_ids:
            rendered = _render_element(document.elements[element_id])
            if rendered:
                blocks.append(rendered)
        pages.append(PageMarkdown(page_number=page.page_number, markdown="\n\n".join(blocks)))

    return MarkdownDocument(
        document_id=document.id,
        source_path=document.source_path,
        source_format=document.source_format,
        pages=pages,
    )
