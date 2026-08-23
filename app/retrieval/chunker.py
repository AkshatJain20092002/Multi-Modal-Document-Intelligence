"""
MarkdownDocument -> list[Chunk]. Default unit is one page; a page's Markdown
that exceeds the embedding model's token budget (bge-small-en-v1.5 has a hard
512-token limit — settings.chunk_max_tokens defaults to 480, leaving
headroom) is split at paragraph boundaries instead of being silently
truncated at embed time. A single paragraph that alone exceeds the budget
(rare) falls back to a word-level split so nothing is ever silently dropped.

Token counting uses the embedding model's own tokenizer (loaded once, cached)
rather than a char/4 approximation, so the budget is enforced exactly.
"""

from __future__ import annotations

import functools

from transformers import AutoTokenizer

from app.config.settings import settings
from app.normalization.markdown_render import MarkdownDocument
from app.retrieval.schema import Chunk


@functools.lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(settings.embedding_model_name)


def _token_count(text: str, *, add_special_tokens: bool = True) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=add_special_tokens))


def _split_oversized_block(block: str, max_tokens: int) -> list[str]:
    """Last-resort word-level split for a single paragraph that alone exceeds
    max_tokens — rare, but must not silently truncate."""
    words = block.split(" ")
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and _token_count(candidate, add_special_tokens=False) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _chunk_page_text(text: str, max_tokens: int) -> list[str]:
    blocks = [b for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return []

    chunks: list[str] = []
    current_blocks: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current_blocks, current_tokens
        if current_blocks:
            chunks.append("\n\n".join(current_blocks))
        current_blocks = []
        current_tokens = 0

    for block in blocks:
        block_tokens = _token_count(block, add_special_tokens=False)
        if block_tokens > max_tokens:
            flush()
            chunks.extend(_split_oversized_block(block, max_tokens))
            continue
        if current_blocks and current_tokens + block_tokens > max_tokens:
            flush()
        current_blocks.append(block)
        current_tokens += block_tokens

    flush()
    return chunks


def markdown_document_to_chunks(markdown_document: MarkdownDocument) -> list[Chunk]:
    max_tokens = settings.chunk_max_tokens
    chunks: list[Chunk] = []

    for page in markdown_document.pages:
        pieces = _chunk_page_text(page.markdown, max_tokens)
        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    document_id=markdown_document.document_id,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=piece,
                    char_count=len(piece),
                    token_count=_token_count(piece),
                    source_format=markdown_document.source_format,
                    embedding_model=settings.embedding_model_name,
                )
            )

    return chunks
