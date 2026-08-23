"""
Deterministic P-1/P/P+1 page windows, built directly from a MarkdownDocument's
own page list — no vector DB, no embeddings, no search. For exhaustive
whole-document extraction, every page is its own window center, unlike
context_builder.py's query-driven path, which needs a Qdrant search just to
know which pages are relevant in the first place. Page order is already
known here, so search is unnecessary overhead.

Reuses DocumentContext/PageContext from context_builder.py rather than
redefining an equivalent shape, so answer_extractor.py works against either
path's output without caring which one built it.
"""

from __future__ import annotations

from collections.abc import Iterator

from app.normalization.markdown_render import MarkdownDocument
from app.retrieval.context_builder import DocumentContext, PageContext


def page_windows(
    markdown_document: MarkdownDocument, *, backward: int = 1, forward: int = 1
) -> Iterator[DocumentContext]:
    """Yields one DocumentContext per page, in page order, each with that page
    as matched_page_numbers=[page_number] and a [page-backward, page+forward]
    window clipped at the document's actual first/last page — same edge-case
    behavior already validated in context_builder.py (page 1 never reaches for
    page 0, the last page never reaches past the end)."""
    pages_by_number = {page.page_number: page for page in markdown_document.pages}
    if not pages_by_number:
        return

    min_page = min(pages_by_number)
    max_page = max(pages_by_number)

    for center in sorted(pages_by_number):
        lo = max(min_page, center - backward)
        hi = min(max_page, center + forward)
        window_pages = [
            PageContext(page_number=n, text=pages_by_number[n].markdown)
            for n in range(lo, hi + 1)
            if n in pages_by_number  # tolerate a gap in page numbering
        ]
        yield DocumentContext(
            document_id=markdown_document.document_id,
            matched_page_numbers=[center],
            pages=window_pages,
        )
