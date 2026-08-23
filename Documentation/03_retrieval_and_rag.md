# Retrieval and RAG

Covers `app/retrieval/chunker.py`, `embedder.py`, `vector_store.py`,
`context_builder.py`, and the standalone `rag_query.py` tool. These are the pieces
that make P-1/P/P+1 page-window retrieval possible — used both by the ad-hoc query
tool in this doc, and (via a different, search-free path) by the Question Bank
pipeline documented in `04_question_bank_pipeline.md`.

## Chunking — page-level, with a real token-limit safety valve

Default unit is one page's rendered Markdown. `BAAI/bge-small-en-v1.5` has a hard
**512-token limit** — confirmed by testing that this isn't a rare edge case: a
single dense MCQ page came out to ~900 tokens, already over the limit. So a page
that exceeds the budget (`settings.chunk_max_tokens`, default 480 — headroom under
512) is split at paragraph boundaries instead of being silently truncated at embed
time; a single paragraph that alone exceeds the budget falls back to a word-level
split. Token counting uses the embedding model's own tokenizer, not a char/4
approximation. `page_number` is kept on every resulting chunk (via `chunk_index`
distinguishing pieces of a split page), so P-1/P/P+1 window retrieval later is just
"chunks whose page_number is in the window" — no special case for a split page.

## Embedding — local, no API key

`BAAI/bge-small-en-v1.5` via `sentence-transformers`, chosen over OpenAI's
embeddings API deliberately: free, no network call, and retrieval-tuned (trained
for asymmetric query/passage matching) rather than a general-purpose
sentence-similarity model like `all-MiniLM-L6-v2`. `embed_passages()` (no prefix)
is used for chunking/ingestion; `embed_queries()` (adds the bge instruction prefix)
is for the retrieval step — both live in `embedder.py` next to the one model load.

## Vector store — one Qdrant collection per document

```python
collection_name(document_id) -> "doc_<sanitized_id>"
```

This is a **reversal of an earlier design**: the first version used one shared
collection with `document_id` as a payload filter. Changed on explicit request —
each document's chunks are now physically isolated (`doc_task_page-0016`,
`doc_Questionnaire`, `doc_SAS_Maths-Class-9_p6-10`, ...), confirmed via
`get_collections()` after re-embedding all sample documents. `search(document_id=None)`
merges results across every `doc_*` collection client-side, since Qdrant's local
mode has no native cross-collection search in one call.

Local/embedded mode — `settings.qdrant_path` is a directory, not a network address.
Point ids are `uuid5`, deterministic from `Chunk.id`, so re-upserting the same
chunk updates its existing point rather than duplicating it.

## Context building — the actual P-1/P/P+1 logic

`context_builder.py`'s `build_context(query, top_k, document_id=None)`:

1. Vector search (`vector_store.search`) returns chunk hits.
2. Hits are grouped by `document_id` — a query with no `document_id` filter can
   match multiple documents; one `DocumentContext` is built per document, pages
   are never interleaved across documents by score.
3. For each document, page bounds (`min`/`max` page number) are derived by
   scrolling that document's own Qdrant collection — no separate metadata store to
   fall out of sync.
4. Each matched page's window (`[page-1, page+1]`, clipped at the document's real
   edges) is computed, then **unioned** across all matched pages in that document
   — e.g. hits on page 2 and page 5 union into `[1,2,3] ∪ [4,5] = [1,2,3,4,5]`,
   not two separate, possibly-overlapping context blocks. Verified edge cases on a
   real 5-page document: `page1 -> [1,2]` (no page 0), `page5 -> [4,5]` (no page 6).
5. The union is clipped to `settings.page_window_ceiling` (default 5) if it grew
   too large from multiple far-apart hits, keeping the pages closest to any
   matched page.
6. Every page in the final window is reassembled from **all** its chunks (not
   just the one that matched), in `chunk_index` order — a retrieved hit is a
   chunk, not necessarily a whole page.
7. Final assembly is always page-order, never retrieval-score order.

## `rag_query.py` — standalone, ad-hoc question answering

Deliberately **not tied to Milestone 2's schema** — no reuse of
`Question`/`Answer` from `app/normalization/schema.py`, no synthesized elements.
`answer_query(query, document_id=None, top_k=5) -> list[RAGAnswer]` — one
`RAGAnswer` per document the search matched.

The prompt went through one real, confirmed fix: an early version returned
`found=true` for "what is the height of the container" when the document only said
"the height is 0.15 m more than the width" (no width given) — a related fact, not
an answer. Tightened with an explicit rule and worked example distinguishing "a
direct answer" from "an on-topic fact requiring further reasoning" — re-verified
against the same failing case (now correctly `NOT FOUND`) and the known-good cases
(still correctly `FOUND`). The same rigor was reused, not just copy-pasted, in
`answer_extractor.py` for the Question Bank pipeline.

```bash
python scripts/run_rag_query.py --query "What is the frequency ratio?" --document-id task_page-0016
```

**Architecturally separate from the Question Bank pipeline** — both read the same
Qdrant collections, but this answers one question on demand; the Question Bank
pipeline (`page_window.py` onward) never calls vector search at all for its
primary path, only during escalation for uncertain items (see
`04_question_bank_pipeline.md`).
