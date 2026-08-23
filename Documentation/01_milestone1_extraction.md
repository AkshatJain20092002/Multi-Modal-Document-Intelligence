# Milestone 1 — Extraction Layer

Status: **finalized**, approved after hands-on testing against sample page images,
a DOCX, and PDFs (both native-text and scanned). This is the reference for what the
extraction layer actually does, what it deliberately doesn't do yet, and how to
run/test it.

## What Milestone 1 covers

Not question detection, answer linking, or orchestration — it proves the
normalized document representation (page + bbox + reading order + element-type
separation, including tables/images/equations) is stable across engines and file
types, before anything is built on top of it.

Pipeline shape: `profile -> select engine (deterministic router) -> extract ->
normalize into common Pydantic Document -> build relationship graph -> JSON`.

## Extraction engines

| Engine | File types | Role |
|---|---|---|
| `PyMuPDFEngine` | PDF with native text layer | Cheap/fast path when OCR isn't needed |
| `DoclingEngine(ocr_backend="rapidocr")` | PDF (scanned), image, DOCX, PPTX | **Primary OCR engine** |
| `DoclingEngine(ocr_backend="easyocr")` | same | Fallback if RapidOCR fails or exceeds its time budget |
| `VisionLLMEngine` (gpt-4o-mini via `LLMRouter`) | image (single page) | Last-resort fallback, structured JSON output |
| `OCRChainEngine` | orchestrates the three above | `app/extraction/ocr_chain.py` |

### OCR chain fallback order and timeouts

```
Docling+RapidOCR (base, NO timeout)
    -> Docling+EasyOCR (fallback, settings.ocr_engine_timeout_seconds budget, default 60s)
    -> vision-LLM (fallback, no timeout — nothing left to fall back to)
```

RapidOCR is given no time budget deliberately: it's the primary precisely because
it's fast and dependency-light, so a slow run signals something's actually wrong
with that page. EasyOCR (heavier, pure PyTorch) gets the timeout since it's more
plausibly slow-but-working depending on hardware. Timeout is enforced by running
the engine in a worker thread and calling `future.result(timeout=...)` — Python
can't forcibly kill a running thread, so a timed-out EasyOCR run keeps using CPU in
the background until it finishes on its own (accepted tradeoff: favors moving on
quickly over strict resource cleanup).

### Vision-LLM fallback: structured, not a text blob

Returns structured JSON (OpenAI Structured Outputs, `response_format=json_schema`,
strict mode) with separate elements per type, not a whole-page text dump. A strict
system prompt instructs verbatim transcription — no paraphrasing, no "fixing"
apparent inconsistencies, no guessing illegible characters.

**Known limitation, confirmed by testing:** the model can still misread
similar-looking characters (e.g. "II" read as "I") — a genuine vision-perception
limit at this tier, not something the prompt can fully correct. Also observed:
duplicated content (an equation appearing both inline and as its own element) and
occasional dropped content blocks. Accepted for V1 since this fallback only fires
rarely (both local OCR engines have to fail first).

## DOCX handling

Two separate fixes here, at different times:

**1. File-type routing fix (original Milestone 1 testing).** `DoclingEngine.extract()`
previously routed any non-image file through `InputFormat.PDF`/`PdfFormatOption`
unconditionally — a `.docx`/`.pptx` input would be silently parsed as if it were a
PDF and break. Fixed: PDF/image inputs get `PdfPipelineOptions` (OCR + page/picture
image generation); DOCX/PPTX get Docling's own native handlers.

**2. DOCX pagination fix (later).** DOCX genuinely has no page concept at the file
level — confirmed empirically: for a real DOCX input, `docling_doc.pages == {}` and
every item's `.prov == []`. Docling's native DOCX handler parses XML directly and
never renders/paginates, so every element fell into `page_number = 1`. Fixed via
`app/ingestion/docx_converter.py`: DOCX is rendered to a real, paginated PDF first
(Windows: `docx2pdf`, needs a licensed MS Word install; Linux: LibreOffice headless,
no license needed — this is the actual deploy-target path), then routed through the
same PDF pipeline as a native PDF (`router.py`: docx now follows
`pymupdf_first`/`ocr_layout_first` based on the rendered PDF's real
`has_native_text`, not straight to Docling's page-less native handler). Extraction
runs against the converted PDF; `Document.source_path`/`source_format` are restored
to the original `.docx` identity afterward (`scripts/run_pipeline.py`).

## Picture and equation elements: crop, don't drop

- `generate_page_images = True` / `generate_picture_images = True` — pure geometric
  bbox cropping from the rendered page image, no model inference.
- Both `picture` and `formula` items are cropped and saved; nothing is silently
  lost even when no text/LaTeX was recognized.
- **Deliberately not using** `do_formula_enrichment=True` (Docling's VLM formula
  recognizer) — too slow locally, same class of problem as Surya/Marker. `latex`
  stays `None` for formula crops; if ever needed, the intended path is an
  on-demand OpenAI vision call per cropped formula, not a bulk local model run.
- Tables/pictures/equations are kept even with no recovered content (bbox +
  provenance alone is meaningful); only empty plain-text elements are dropped.

## Asset storage layout, and the multi-page-folder overwrite bug

```
output/assets/<document_id>/page_<N>/picture_1.png
output/assets/<document_id>/page_<N>/formula_1.png
output/assets/<document_id>/page_<N>/img_1.jpg      (PyMuPDF native-image path)
```

**Confirmed and fixed bug:** when a folder of page images (e.g. `sample_data`) is
ingested as one multi-page document (`build_document_from_image_folder` in
`scripts/run_pipeline.py`), each page is OCR'd via a separate `engine.extract()`
call — and `DoclingEngine` has no idea that call is "page 20 of 30"; it always
numbers a standalone single-image input as "page 1" internally, so every page's
crops landed in the *same* `page_1` folder, and same-named crops from later pages
silently overwrote earlier pages' crops (real data loss, not just wrong labels).
Fixed by giving each page a unique scratch `document_id` during its own
`extract()` call (forcing Docling to write into a distinct, non-colliding folder),
then relocating those crops into the real `assets/<doc_id>/page_<N>/` folder and
fixing up `Provenance.document_id`/`page_number` afterward. Verified against a real
3-page test: every page's assets survive, correctly attributed, no leftover scratch
folders.

## Known gaps carried forward

- **Reading order can break on dense 2×2 MCQ option grids** — confirmed on a real
  page: options (a)/(c) read correctly in sequence, but (b)/(d) get displaced to
  the end of the page's element stream. Root cause is Docling's own reading-order
  model, not this codebase.
- **No cross-page table continuation/header inheritance** — a table split across a
  page boundary is captured correctly per-page, but the continuation page gets
  generic positional headers instead of inheriting the real header row.
- **Only tested against one real document family in depth** (a 30-page scanned
  science workbook) plus spot-checks on a maths PDF and a DOCX questionnaire. The
  original plan's exit bar (10-20 varied documents/publishers) hasn't been run.
- `VisionLLMEngine` assumes a single-image input — it can't open a raw multi-page
  PDF directly (only matters if the OCR chain is invoked on a PDF path and both
  Docling backends fail).

## How to run

Single-engine smoke test against one file:

```bash
python scripts/test_extraction_engine.py --engine docling_rapidocr --image sample_data/task_page-0016.jpg
python scripts/test_extraction_engine.py --engine vision_llm       --image sample_data/task_page-0016.jpg
python scripts/test_extraction_engine.py --engine pymupdf          --image some_native_text.pdf
```

Full Milestone 1 pipeline (folder of images, or a single PDF/DOCX):

```bash
python scripts/run_pipeline.py --input sample_data --doc-id my_doc
```

## Config knobs

- `settings.ocr_engine_timeout_seconds` (default `60.0`) — EasyOCR's fallback
  budget in `ocr_chain.py`.
- `settings.openai_extraction_model` / `settings.openai_reasoning_fallback` —
  Tier 1/2 model names for `LLMRouter`.
