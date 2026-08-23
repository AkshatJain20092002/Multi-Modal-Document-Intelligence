"""
Thin wrapper around the local sentence-transformers embedding model
(BAAI/bge-small-en-v1.5 by default, per settings.embedding_model_name) —
local/CPU, no API key, no network call.

BGE models are trained for *asymmetric* retrieval: passages/documents are
embedded plain, but queries need an instruction prefix to land in the same
vector space as what the model was trained to match against. embed_passages()
is what chunking/ingestion uses now; embed_queries() is for the retrieval
step (later milestone), kept here so both live next to the one model load.
"""

from __future__ import annotations

import functools

from sentence_transformers import SentenceTransformer

from app.config.settings import settings

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@functools.lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


def embedding_dimension() -> int:
    return _model().get_sentence_embedding_dimension()


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed chunk/document text — no instruction prefix (bge convention)."""
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def embed_queries(texts: list[str]) -> list[list[float]]:
    """Embed a search query — bge requires this instruction prefix so the
    query lands in the same space as passages embedded via embed_passages()."""
    prefixed = [_QUERY_INSTRUCTION + t for t in texts]
    return _model().encode(prefixed, normalize_embeddings=True, show_progress_bar=False).tolist()
