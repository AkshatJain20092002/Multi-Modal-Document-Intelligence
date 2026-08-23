# Multi-Modal Document Intelligence

Extracts every question and answer from an educational document — PDF, DOCX, or
scanned page images — into a structured, ready-to-use Question Bank (Excel/CSV).

## About the Project

Textbooks, worksheets, and question banks mostly exist as unstructured PDFs,
scanned images, or Word documents. Turning that into something searchable,
gradeable, or reusable for building an actual question bank today usually means
manual data entry, or a brittle template built for one specific publisher's layout.

This project does that extraction automatically, without assuming anything about
the document's publisher, layout, or format. Feed it a PDF, a scanned image, or a
DOCX file, and it returns a clean, structured spreadsheet of every question and its
answer — with page-level provenance for every entry.

The core design principle is **extraction, not generation**: every answer is
tagged with how confidently it was sourced —

- `exact` — the answer is explicitly stated in the document
- `multi_source` — assembled from information across more than one page
- `inferred` — reasonably deduced, but not explicitly stated (always flagged as such)
- `not_found` — no answer could be determined from the document

Nothing is ever fabricated and presented as if it came from the source document.

## Features

- **Format-agnostic ingestion** — native or scanned PDF, DOCX, and scanned page
  images, all through the same pipeline
- **Local-first OCR** with a cascading fallback chain (RapidOCR → EasyOCR →
  vision-LLM), keeping cost and dependency footprint low
- **Page-accurate structure extraction** — page numbers, bounding boxes, reading
  order, and element types (text, headings, tables, images, equations) preserved
  for every page, including DOCX (rendered to a real paginated PDF first, since
  DOCX has no page concept on its own)
- **Page-boundary-aware question detection** — a question that spans two pages
  isn't lost or duplicated
- **Verbatim answer extraction** with previous/current/next-page context, so an
  answer sitting on a nearby page still gets found
- **Automatic validation and retry** — low-confidence or inconsistent extractions
  are flagged and automatically retried with a broader search before being
  accepted
- **Excel/CSV export** — a clean, reviewable Question Bank as the final output
- **Ad-hoc semantic Q&A** — a standalone tool to ask any question against an
  already-ingested document and get a verbatim, sourced answer (or an honest "not
  found")
- **Local vector search** — embeddings and vector storage run entirely locally, no
  API key needed for retrieval itself

## Tech Stack

| | |
|---|---|
| Language | Python 3.11+ |
| Document parsing / OCR | Docling, RapidOCR, EasyOCR, PyMuPDF |
| LLM | OpenAI (`gpt-4o-mini` primary, `gpt-4o` available for escalation) |
| Vector search | QdrantDB (embedded/local mode) |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) |
| Data modeling | Pydantic / Langchain|
| Export | openpyxl (Excel) |
| DOCX → PDF rendering | `docx2pdf` (Windows) / LibreOffice headless (Linux) |

## Project Structure

```
app/
  ingestion/      file-type detection, page profiling, DOCX -> PDF conversion
  extraction/     OCR/parsing engines and the router that selects between them
  normalization/  the canonical document schema and its derived views
  retrieval/      chunking, embeddings, vector search, question/answer extraction
  llm/            OpenAI call routing
  config/         all runtime settings (env-driven)
scripts/          CLI entry points — see Usage below
Documentation/    architecture diagrams and design notes
```

## Installation & Setup

Requires Python 3.11+.

```bash
git clone <repo-url>
cd Multi-Modal-Document-Intelligence
pip install -r requirements.txt
```

The first run downloads a couple of models automatically (a layout model, the
embedding model), so expect it to be slower than subsequent runs.

**DOCX input** additionally needs one of the following, since DOCX has no native
page concept and must be rendered to a real PDF first:

- **Windows** — a licensed MS Word install (`docx2pdf` is already in
  `requirements.txt` and drives it via COM)
- **Linux** — LibreOffice headless: `apt-get install libreoffice`

PDF and image input need neither of these.

## Environment Variables

Copy the example file and fill in your key:

```bash
cp .env.example .env
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | Question detection, answer extraction, vision-OCR fallback |
| `OPENAI_EXTRACTION_MODEL` | No | `gpt-4o-mini` | Primary model |
| `OPENAI_REASONING_FALLBACK` | No | `gpt-4o` | Escalation model |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |

Everything else is configured in `app/config/settings.py` with working defaults —
no other environment variables are required to run the pipeline.

## Usage

Run the full pipeline on a single file or a folder of page images:

```bash
python scripts/run_full_pipeline.py --input path/to/document.pdf
python scripts/run_full_pipeline.py --input path/to/scanned_pages/   # folder = one multi-page document
```

Output: `output/<doc_id>.question_bank.xlsx`.

Run just the ad-hoc Q&A tool against an already-ingested document:

```bash
python scripts/run_rag_query.py --query "What is the boiling point of water?" --document-id my_doc
```

Individual pipeline stages can also be run on their own — useful for inspecting
intermediate output or debugging a single step:

```bash
python scripts/run_pipeline.py --input path/to/document.pdf --doc-id my_doc   # extraction
python scripts/export_markdown.py --doc-json output/my_doc.json               # Markdown rendering
python scripts/export_pagewise.py --doc-json output/my_doc.json               # Structured JSON rendering
python scripts/run_chunk_embed.py --markdown-json output/my_doc.markdown.json # chunk + embed
python scripts/run_question_bank.py --markdown-json output/my_doc.markdown.json # detect -> answer -> export
```

### Example output

| question_number | question_text | answer_status | answer_text |
|---|---|---|---|
| 1 | Two sound waves in air have a wavelength ratio 3:7. Their frequency ratio will be: | exact | 7 : 3 |
| 3 | What are the media through which sound can travel? | exact | Solid, liquid and gas |

Every row also carries page number, detection/answer confidence, source pages, and
a validation verdict.

## Documentation

Architecture diagrams and detailed design notes for each stage of the pipeline
live in [`Documentation/`](Documentation/), starting with
[`pipeline_architecture.md`](Documentation/pipeline_architecture.md).

## Future Improvements

- Handle nested/reused question labels on a page more robustly (a page reusing the
  same "(A)/(B)/(C)" for two unrelated question sets can currently confuse answer
  attribution)
- Full-scale validation across a large, varied multi-page document set
- PPTX support
- A lightweight review UI for the generated Question Bank

## License

Personal/private project — no license specified.
