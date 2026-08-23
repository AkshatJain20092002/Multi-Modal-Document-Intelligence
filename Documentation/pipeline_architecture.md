# Pipeline Architecture

Full top-down flow of the current pipeline, one command end to end via
`scripts/run_full_pipeline.py`. Each stage's real file is named so this diagram
stays checkable against the code, not just a description of intent.

```
                                   INPUT
                        sample_data/ (pdf, docx, image)
                                     |
                                     v
                     scripts/run_pipeline.py  (Milestone 1)
        +--------------------+-----------------------+--------------------+
        |                    |                       |                    |
    native PDF             DOCX                  scanned PDF        image folder
  (PyMuPDFEngine)   (docx_converter.py:        / standalone image  (build_document_from_
                     OS-dispatched docx2pdf/   (OCRChainEngine:     image_folder -- every
                     LibreOffice -> routed      Docling+RapidOCR     page gets a unique
                     through the PDF path)      -> Docling+EasyOCR   scratch dir so no two
                                                 -> VisionLLM         pages' assets collide;
                                                 fallback)            real page numbers 1..N)
        +--------------------+-----------------------+--------------------+
                                     |
                                     v
                    output/<doc_id>.json  (canonical Document)
                    page + bbox + reading order + element type
                                     |
                      +--------------+---------------+
                      v                               v
      app/normalization/page_view.py      app/normalization/markdown_render.py
      -> PagewiseDocument                 -> MarkdownDocument
      (bucketed by type: text/            (reading-order preserved,
       headings/tables/images/             one .md per page)
       equations/captions/metadata)               |
      export_pagewise.py                  export_markdown.py
                                     output/<doc_id>.markdown.json
                                                   |
                                    +--------------+--------------+
                                    v                              v
                     app/retrieval/chunker.py        app/retrieval/page_window.py
                     (1 page = 1 chunk, split         (deterministic P-1/P/P+1 per
                      only if > bge-small's            page, NO vector search --
                      512-token limit)                 page order already known)
                                    |                              |
                                    v                              v
                     app/retrieval/embedder.py        app/retrieval/question_detector.py
                     (BAAI/bge-small-en-v1.5,          detect_questions_on_page_with_context()
                      local, no API key)               - page P = only source of NEW questions
                                    |                   - small [CONTEXT ONLY] peek into P-1/P+1
                                    v                     boundaries, used only to complete a
                     app/retrieval/vector_store.py        question that spills across a page break
                     Qdrant, ONE COLLECTION PER            - gpt-4o-mini
                     DOCUMENT: doc_<sanitized_id>                        |
                     scripts/run_chunk_embed.py                          v
                                    |                    app/retrieval/answer_extractor.py
                                    |                    extract_answers_for_questions()
                                    |                     - builds each question's own P-1/P/P+1
                                    |                       window (page_window.py)
                                    |                     - gpt-4o-mini, verbatim-only,
                                    |                       answer_status: exact/multi_source/
                                    |                       inferred/not_found
                                    |                                    |
                                    |                                    v
                                    |                    app/retrieval/validator.py
                                    |                    deterministic (NO LLM) -- good / uncertain
                                    |                                    |
                                    |                       +------------+------------+
                                    |                     good                   uncertain
                                    |                       |                        |
                                    |                       |                        v
                                    |                       |        app/retrieval/escalation.py
                                    +---------------------->|<----- targeted RAG: real vector
                        (Qdrant search used here,           |        search via context_builder.
                         only for uncertain retries)         |        build_context() against that
                                                              |        doc's own Qdrant collection --
                                                              |        STILL gpt-4o-mini, never
                                                              |        escalated to gpt-4o
                                                              |                        |
                                                              +-----------+------------+
                                                                          v
                                                    app/retrieval/question_bank_export.py
                                                                          |
                                                                          v
                                                output/<doc_id>.question_bank.xlsx  (.csv also available)
                                                            (the "Question Bank")
                                       scripts/run_question_bank.py  (this stage alone)
                                       scripts/run_full_pipeline.py (all stages, one command)


  -- separate, independent consumer of the same Qdrant store -- NOT part of the pipeline above --

                     app/retrieval/context_builder.py
                     query -> Qdrant search -> group by document -> union/clip
                     overlapping P-1/P/P+1 windows -> page-ordered context
                                    |
                                    v
                     app/retrieval/rag_query.py
                     answer_query(query, document_id=None)
                     "does this document verbatim-answer THIS question?"
                     found=true/false, never fabricates
                     scripts/run_rag_query.py
```

## Two things this diagram makes visible on purpose

**`page_view.py` / `PagewiseDocument` is a live but unused side branch.** It
produces a real, correct type-bucketed JSON (all headings together, all tables
together, etc.) — kept because it's useful for page-level structural inspection —
but nothing in the chunk/embed/detect/answer path reads it. Markdown rendering
reads the canonical `Document` directly instead, because reading order (which the
detect/answer stages depend on to correctly attribute an answer to its question)
would be lost if it were built from the type-bucketed groups. See
`02_normalization_and_views.md` for the full reasoning.

**`rag_query.py` is architecturally separate from the Question Bank pipeline.**
Both read the same per-document Qdrant collections, but `rag_query.py` answers one
ad-hoc question on demand (a "chat with your document" use case), while
`question_bank_export.py`'s pipeline exhaustively walks every page of a document.
They were deliberately kept decoupled — see `03_retrieval_and_rag.md`.

## One-command entry point

```bash
python scripts/run_full_pipeline.py --input <file-or-folder> [--doc-id ...]
```

Runs every stage in this diagram in order, by calling each stage script's own
`run()` function directly (not a subprocess). Each stage can also be run alone —
see the README's "Individual pipeline stages" section — which is the faster way to
debug one stage without re-running everything before it.
