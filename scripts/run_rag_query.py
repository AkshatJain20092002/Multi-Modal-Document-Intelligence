"""
Standalone RAG query answering CLI — ask a question, get an answer sourced
from whatever's already been chunked+embedded into Qdrant (scripts/run_chunk_embed.py).
Not tied to Milestone 2 — see app/retrieval/rag_query.py.

Usage:
    python scripts/run_rag_query.py --query "What is the frequency ratio?"
    python scripts/run_rag_query.py --query "..." --document-id task_page-0016
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.observability.logging_config import configure_logging
from app.retrieval.rag_query import answer_query


def run(query: str, document_id: str | None, top_k: int) -> None:
    results = answer_query(query, document_id=document_id, top_k=top_k)

    if not results:
        print("No matching document/chunks found for this query.")
        return

    for result in results:
        print(f"\ndocument_id={result.document_id}  context_pages_used={result.context_pages_used}")
        if result.found:
            print(f"  FOUND (confidence={result.confidence:.2f}, source_pages={result.source_pages})")
            if result.matched_question_text:
                print(f"  matched question: {result.matched_question_text!r}")
            print(f"  answer: {result.answer_text!r}")
        else:
            print(f"  NOT FOUND in this document (confidence={result.confidence:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--document-id", default=None, type=str)
    parser.add_argument("--top-k", default=5, type=int)
    args = parser.parse_args()

    configure_logging()
    run(args.query, args.document_id, args.top_k)
