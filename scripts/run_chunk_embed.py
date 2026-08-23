"""
Chunk + Embed — reads the page-wise Markdown manifest (scripts/export_markdown.py's
output) and:
  1. splits each page into token-budget-safe chunks (paragraph-boundary aware,
     app/retrieval/chunker.py)
  2. embeds each chunk locally (BAAI/bge-small-en-v1.5 by default — no API
     key, no network call, app/retrieval/embedder.py)
  3. upserts into the local Qdrant store at settings.qdrant_path
     (app/retrieval/vector_store.py)

Usage:
    python scripts/run_chunk_embed.py --markdown-json output/task_page-0016.markdown.json
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.normalization.markdown_render import MarkdownDocument
from app.retrieval.chunker import markdown_document_to_chunks
from app.retrieval.vector_store import collection_name, upsert_chunks


def run(markdown_json_path: Path, output_path: Path) -> None:
    markdown_document = MarkdownDocument.model_validate_json(markdown_json_path.read_text(encoding="utf-8"))
    chunks = markdown_document_to_chunks(markdown_document)

    per_page = collections.Counter(chunk.page_number for chunk in chunks)
    print(f"\ndocument_id={markdown_document.document_id} pages={len(markdown_document.pages)} chunks={len(chunks)}")
    for page_number in sorted(per_page):
        print(f"  page {page_number:>3}: {per_page[page_number]} chunk(s)")

    print("\nFirst 5 chunks:")
    for chunk in chunks[:5]:
        preview = chunk.text[:70].replace("\n", " ")
        print(f"  [{chunk.page_number}:{chunk.chunk_index}] tokens={chunk.token_count:<4} :: {preview!r}")

    output_path.write_text(
        "[" + ",\n".join(chunk.model_dump_json() for chunk in chunks) + "]",
        encoding="utf-8",
    )
    print(f"\nWrote {len(chunks)} chunks to {output_path}")

    print("\nEmbedding + upserting to local Qdrant...")
    count = upsert_chunks(chunks)
    print(f"Upserted {count} point(s) into the '{collection_name(markdown_document.document_id)}' collection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-json", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    output = args.output or args.markdown_json.with_suffix("").with_suffix(".chunks.json")
    run(args.markdown_json, output)
