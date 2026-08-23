"""
Context builder — turns vector-search hits into page-ordered context blocks:

    Vector DB -> Retrieval -> Page ref -> P-1 / P / P+1 (clipped at doc edges)

A retrieved hit is a *chunk*, not necessarily a whole page (a dense page can
split into several chunks — see chunker.py), so "current page" here always
means every chunk belonging to that page_number, reassembled in chunk_index
order — never just the one chunk that happened to match the query.

Edge cases handled explicitly:
    - first page  -> window is [1, 2]            (no page 0)
    - last page   -> window is [N-1, N]           (no page N+1)
    - final assembly is always sorted by page_number (then chunk_index within
      a page) — retrieval score decides *inclusion*, never final order.

Multiple top-k hits are grouped by document_id first; within a document,
overlapping windows from different hits are unioned/deduped rather than
producing duplicate/overlapping context blocks (e.g. hits on page 3 and page
4 union into one [2,3,4,5] window, not two separate windows). Across
documents, one context block is built per document — pages are never
interleaved across documents by score.

Page bounds (needed to detect "is this the last page") are derived by
scrolling that document's own Qdrant collection rather than kept as separate
metadata — one source of truth, no sidecar to fall out of sync.
"""

from __future__ import annotations

import collections

from pydantic import BaseModel, Field
from qdrant_client.http import models as qmodels

from app.config.settings import settings
from app.retrieval.vector_store import collection_name, get_client, search


class PageContext(BaseModel):
    page_number: int
    text: str  # that page's chunks, rejoined in chunk_index order


class DocumentContext(BaseModel):
    document_id: str
    matched_page_numbers: list[int]  # pages that actually matched the query, pre-expansion
    pages: list[PageContext] = Field(default_factory=list)  # the expanded, deduped, ordered window

    def combined_text(self) -> str:
        parts = [f"<!-- page {page.page_number} -->\n\n{page.text}" for page in self.pages]
        return "\n\n---\n\n".join(parts)


def _page_bounds(client, document_id: str) -> tuple[int, int] | None:
    name = collection_name(document_id)
    if not client.collection_exists(name):
        return None
    points, _ = client.scroll(
        collection_name=name,
        limit=10_000,
        with_payload=["page_number"],
        with_vectors=False,
    )
    if not points:
        return None
    page_numbers = [p.payload["page_number"] for p in points]
    return min(page_numbers), max(page_numbers)


def _window_for_page(page_number: int, bounds: tuple[int, int], *, backward: int, forward: int) -> set[int]:
    min_page, max_page = bounds
    lo = max(min_page, page_number - backward)
    hi = min(max_page, page_number + forward)
    return set(range(lo, hi + 1))


def _clip_to_ceiling(pages: set[int], matched_pages: list[int], ceiling: int) -> set[int]:
    if len(pages) <= ceiling:
        return pages
    # Keep the pages closest to any matched page; ties broken by page number
    # for deterministic output.
    def distance(page: int) -> int:
        return min(abs(page - m) for m in matched_pages)

    closest = sorted(pages, key=lambda p: (distance(p), p))[:ceiling]
    return set(closest)


def _fetch_pages(client, document_id: str, page_numbers: set[int]) -> list[PageContext]:
    if not page_numbers:
        return []
    points, _ = client.scroll(
        collection_name=collection_name(document_id),
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="page_number", match=qmodels.MatchAny(any=sorted(page_numbers)))]
        ),
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )

    by_page: dict[int, list[dict]] = collections.defaultdict(list)
    for point in points:
        by_page[point.payload["page_number"]].append(point.payload)

    pages: list[PageContext] = []
    for page_number in sorted(by_page):
        chunks = sorted(by_page[page_number], key=lambda payload: payload["chunk_index"])
        text = "\n\n".join(chunk["text"] for chunk in chunks)
        pages.append(PageContext(page_number=page_number, text=text))
    return pages


def build_context(
    query_text: str,
    *,
    top_k: int = 5,
    document_id: str | None = None,
    window_backward: int | None = None,
    window_forward: int | None = None,
) -> list[DocumentContext]:
    backward = window_backward if window_backward is not None else settings.page_window_backward
    forward = window_forward if window_forward is not None else settings.page_window_default
    ceiling = settings.page_window_ceiling

    hits = search(query_text, top_k=top_k, document_id=document_id)
    if not hits:
        return []

    matched_pages_by_doc: dict[str, set[int]] = collections.defaultdict(set)
    for hit in hits:
        matched_pages_by_doc[hit.payload["document_id"]].add(hit.payload["page_number"])

    client = get_client()
    results: list[DocumentContext] = []

    for doc_id, matched_pages in matched_pages_by_doc.items():
        bounds = _page_bounds(client, doc_id)
        if bounds is None:
            continue

        window: set[int] = set()
        for page_number in matched_pages:
            window |= _window_for_page(page_number, bounds, backward=backward, forward=forward)

        window = _clip_to_ceiling(window, sorted(matched_pages), ceiling)
        pages = _fetch_pages(client, doc_id, window)

        results.append(
            DocumentContext(
                document_id=doc_id,
                matched_page_numbers=sorted(matched_pages),
                pages=pages,
            )
        )

    return results
