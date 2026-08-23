"""
Reshapes an existing Milestone-1 normalized Document JSON (from run_pipeline.py
or test_extraction_engine.py) into the page-wise structured view:

    page
    ├── text
    ├── headings
    ├── tables
    ├── images
    ├── equations
    ├── captions
    ├── other       (running headers/footers, page numbers, decorative, unknown)
    └── metadata

Pure reshape, no re-extraction — same output shape regardless of whether the
source Document JSON came from a PDF, DOCX, or scanned image, since every
engine already writes into the same normalized schema.

Usage:
    python scripts/export_pagewise.py --doc-json output/task_page-0016.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.normalization.page_view import document_to_pagewise
from app.normalization.schema import Document


def run(doc_json_path: Path, output_path: Path) -> None:
    document = Document.model_validate_json(doc_json_path.read_text(encoding="utf-8"))
    pagewise = document_to_pagewise(document)

    print(f"\ndocument_id={pagewise.document_id} source_format={pagewise.source_format} pages={len(pagewise.pages)}")
    for page in pagewise.pages:
        print(
            f"  page {page.page:>3}: text={len(page.text):<3} headings={len(page.headings):<2} "
            f"tables={len(page.tables):<2} images={len(page.images):<2} "
            f"equations={len(page.equations):<2} captions={len(page.captions):<2} other={len(page.other):<2}"
        )

    output_path.write_text(pagewise.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nWrote page-wise structured JSON to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-json", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    output = args.output or args.doc_json.with_suffix(".pagewise.json")
    run(args.doc_json, output)
