"""
Local (embedded, no server) Qdrant wrapper — settings.qdrant_path is a local
directory, not a network address.

One Qdrant collection PER DOCUMENT, named after that document's document_id
(sanitized). This reverses an earlier shared-collection design on purpose:
the user wants each document's chunks physically isolated — task_page-0016,
Questionnaire, and SAS_Maths-Class-9_p6-10 each get their own collection,
not just a payload filter within one shared collection.

Point ids: Chunk.id is a readable "chunk_<hex>" string, but Qdrant point ids
must be an unsigned int or a UUID. uuid5, deterministic from chunk.id, is
used as the actual point id — re-upserting the same chunk updates its
existing point instead of creating a duplicate. chunk.id is kept in the
payload for traceability back to the Chunk record.
"""

from __future__ import annotations

import collections
import re
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config.settings import settings
from app.retrieval.embedder import embed_passages, embed_queries, embedding_dimension
from app.retrieval.schema import Chunk

_COLLECTION_PREFIX = "doc_"
_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_POINT_ID_NAMESPACE = uuid.UUID("6f6b6f9a-6b1a-4b6b-9c8a-9b3f7a5c9e10")


def collection_name(document_id: str) -> str:
    """One collection per document. Qdrant collection names are restricted to
    a safe charset here (document_id could contain characters Qdrant doesn't
    accept), so this is deterministic but not always == document_id verbatim."""
    safe = _UNSAFE_CHARS.sub("_", document_id)
    return f"{_COLLECTION_PREFIX}{safe}"


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def get_client() -> QdrantClient:
    return QdrantClient(path=str(settings.qdrant_path))


def list_document_collections(client: QdrantClient | None = None) -> list[str]:
    client = client or get_client()
    return [c.name for c in client.get_collections().collections if c.name.startswith(_COLLECTION_PREFIX)]


def ensure_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=embedding_dimension(), distance=qmodels.Distance.COSINE),
    )


def upsert_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    client = get_client()
    by_document: dict[str, list[Chunk]] = collections.defaultdict(list)
    for chunk in chunks:
        by_document[chunk.document_id].append(chunk)

    total = 0
    for document_id, doc_chunks in by_document.items():
        name = collection_name(document_id)
        ensure_collection(client, name)
        vectors = embed_passages([chunk.text for chunk in doc_chunks])

        points = [
            qmodels.PointStruct(
                id=_point_id(chunk.id),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "char_count": chunk.char_count,
                    "token_count": chunk.token_count,
                    "source_format": chunk.source_format,
                    "embedding_model": chunk.embedding_model,
                },
            )
            for chunk, vector in zip(doc_chunks, vectors)
        ]
        client.upsert(collection_name=name, points=points)
        total += len(points)

    return total


def search(query_text: str, *, top_k: int = 5, document_id: str | None = None) -> list[qmodels.ScoredPoint]:
    client = get_client()
    [query_vector] = embed_queries([query_text])

    if document_id is not None:
        name = collection_name(document_id)
        if not client.collection_exists(name):
            return []
        return client.query_points(collection_name=name, query=query_vector, limit=top_k).points

    # No document_id given: search across every document's collection and merge by score, since there's no single collection to query anymore.
    all_hits: list[qmodels.ScoredPoint] = []
    for name in list_document_collections(client):
        all_hits.extend(client.query_points(collection_name=name, query=query_vector, limit=top_k).points)
    all_hits.sort(key=lambda point: point.score, reverse=True)
    return all_hits[:top_k]
