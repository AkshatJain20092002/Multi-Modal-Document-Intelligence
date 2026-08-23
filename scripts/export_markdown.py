"""
Renders an existing Milestone-1 normalized Document JSON (from run_pipeline.py
or test_extraction_engine.py) into page-wise Markdown — the human-readable
projection that sits alongside the canonical JSON, per the "JSON is canonical,
Markdown is derived" principle. Feeds the next stage (chunk + embed).

Usage:
    python scripts/export_markdown.py --doc-json output/task_page-0016.json

Writes:
    output/markdown/<doc_id>/page_<NNN>.md   one file per page
    output/markdown/<doc_id>/full.md         combined, page-break comments
    output/<doc_id>.markdown.json            manifest (document_id/source_path/
                                              source_format/pages[]) for the
                                              next stage to load directly
                                              instead of re-deriving it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.normalization.markdown_render import document_to_markdown
from app.normalization.schema import Document


def run(doc_json_path: Path, output_dir: Path) -> None:
    document = Document.model_validate_json(doc_json_path.read_text(encoding="utf-8"))
    rendered = document_to_markdown(document)

    doc_id = rendered.document_id
    markdown_dir = output_dir / "markdown" / doc_id
    markdown_dir.mkdir(parents=True, exist_ok=True)

    print(f"\ndocument_id={doc_id} source_format={rendered.source_format} pages={len(rendered.pages)}")
    for page in rendered.pages:
        page_path = markdown_dir / f"page_{page.page_number:03d}.md"
        page_path.write_text(page.markdown, encoding="utf-8")
        print(f"  page {page.page_number:>3}: {len(page.markdown):>6} chars -> {page_path}")

    full_path = markdown_dir / "full.md"
    full_path.write_text(rendered.combined(), encoding="utf-8")

    manifest_path = output_dir / f"{doc_id}.markdown.json"
    manifest_path.write_text(rendered.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nWrote combined Markdown to {full_path}")
    print(f"Wrote manifest JSON to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-json", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("output"), type=Path)
    args = parser.parse_args()

    run(args.doc_json, args.output_dir)
