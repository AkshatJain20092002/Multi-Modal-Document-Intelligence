# Normalization and Views

## The canonical representation

`app/normalization/schema.py`'s `Document` is the single backbone every
extraction engine (PyMuPDF, Docling, VisionLLM) writes into, regardless of source
format. This is what makes the pipeline format-agnostic in practice, not just in
name — a PDF, a DOCX, and a scanned image all land in the exact same shape:

```
Document
  id, source_path, source_format ("pdf"/"docx"/"pptx"/"image")
  pages: [Page]              page_number, width/height, has_native_text, element_ids (reading order)
  elements: {id: DocumentElement}   type, content, table_data, latex, asset_path, provenance
```

Every element carries `Provenance` (document_id, page_number, bbox, parser,
parser_confidence, source_crop_path) — the design rule is "never lose the path
back to the source pixel." `Answer.answer_status` is always one of
`exact`/`multi_source`/`inferred`/`not_found` — extraction, not generation.

## Two independent derived views

Two separate projections are built off the same canonical `Document` — deliberately
**not chained** into each other:

```
Document (canonical)
   +--> page_view.py       -> PagewiseDocument   (bucketed by element type)
   +--> markdown_render.py -> MarkdownDocument    (reading order preserved)
```

### `markdown_render.py` — the one that matters downstream

Renders each page's elements, in reading order, into Markdown: headings become
`## `, tables become Markdown pipe tables, images/equations become links (or an
inline `$$latex$$` block when recovered), MCQ options become bullets, running
headers/footers/page numbers become HTML comments (kept for traceability, not shown
as content). This is what every downstream stage — chunking, embedding, question
detection, answer extraction — actually reads.

**Why reading order matters concretely, not just in principle:** question
detection and answer extraction both depend on a question and its answer sitting
near each other in the natural print sequence to correctly attribute one to the
other. A real bug found during testing (`task_page-0020`) involved a page reusing
the same `(A)/(B)/(C)` labels for two separate question blocks — the only thing that
makes correct disambiguation possible at all is that the real answer sits in
reading-order proximity to its actual question, not just sharing a label.

### `page_view.py` — kept, but not wired into the pipeline

Buckets each page's elements by type into `text`/`headings`/`tables`/`images`/
`equations`/`captions`/`other`, plus per-page `metadata`. Useful for structural
inspection ("does this page have a table? how many images?") — but a genuinely
different kind of artifact from a coherent, orderable passage. Nothing in the
chunk/embed/detect/answer path reads it.

**This was a deliberate decision, not an oversight — reconsidered explicitly once
during this project and kept as-is.** The alternative (`Document -> PagewiseDocument
-> Markdown`) was evaluated and rejected: reconstructing true reading order from
type-bucketed groups is extra work and error-prone, and changing `page_view.py` to
preserve order too would defeat the point of bucketing by type in the first place.
Embeddings also specifically want coherent excerpts a human would recognize as one
passage, not "every table on this page, concatenated." If a real consumer for
`PagewiseDocument` shows up later (structural filtering — "only search pages with a
table" — was floated as a middle-ground idea: pull a few counts from it into Qdrant
payload metadata without changing what gets embedded), it's still available and
correct; it's just not on the critical path today.

## Running the views

```bash
python scripts/export_markdown.py --doc-json output/my_doc.json
python scripts/export_pagewise.py --doc-json output/my_doc.json
```

`export_markdown.py` writes per-page `.md` files, a combined `full.md`, and a
`<doc_id>.markdown.json` manifest that everything downstream actually consumes.
`export_pagewise.py` writes `<doc_id>.pagewise.json` for standalone inspection.
