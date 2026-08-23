"""
Chunk — the retrieval unit. One chunk is normally one page's rendered
Markdown; a page whose Markdown exceeds the embedding model's token budget
is split into multiple chunks at paragraph boundaries (see chunker.py), so
page_number + chunk_index together identify a chunk's position within a
page, not id alone. page_number is always kept, so P-1/P/P+1 window
retrieval later is just "chunks whose page_number is in the window" — no
special case for a page that got split into more than one chunk.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.normalization.schema import new_element_id


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: new_element_id("chunk"))
    document_id: str
    page_number: int
    chunk_index: int  # 0-based position within the page (0 if the page wasn't split)
    text: str
    char_count: int
    token_count: int  # exact, from the embedding model's own tokenizer
    source_format: str
    embedding_model: str
