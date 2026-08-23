"""
Unified entry point — input a single file (pdf/docx/image) or a folder of
page images, run the entire pipeline end to end, and get back an Excel
question bank. Chains together the same building blocks each standalone
script already uses (Milestone 1 extraction -> Markdown export -> pagewise
JSON export -> chunk + embed -> detect -> answer -> validate -> escalate ->
xlsx), calling each stage's own run() function directly rather than
shelling out to a subprocess.

Usage:
    python scripts/run_full_pipeline.py --input sample_data/task_page-0016.jpg
    python scripts/run_full_pipeline.py --input sample_data --doc-id task_pages_full
    python scripts/run_full_pipeline.py --input sample_data/Questionnaire.docx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.export_markdown as export_markdown_stage
import scripts.export_pagewise as export_pagewise_stage
import scripts.run_chunk_embed as run_chunk_embed_stage
import scripts.run_pipeline as run_pipeline_stage
from app.normalization.markdown_render import MarkdownDocument
from app.observability.logging_config import configure_logging
from app.retrieval.answer_extractor import extract_answers_for_questions
from app.retrieval.escalation import escalate_all_uncertain
from app.retrieval.question_bank_export import write_question_bank_xlsx
from app.retrieval.question_detector import detect_questions_on_page_with_context
from app.retrieval.validator import validate_answered_questions


def run(input_path: Path, document_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Stage 1/5: Milestone 1 extraction ===")
    run_pipeline_stage.run(input_path, document_id, output_dir)
    doc_json_path = output_dir / f"{document_id}.json"

    print("\n=== Stage 2/5: Markdown export ===")
    export_markdown_stage.run(doc_json_path, output_dir)
    markdown_json_path = output_dir / f"{document_id}.markdown.json"

    print("\n=== Stage 3/5: Pagewise JSON export ===")
    pagewise_path = output_dir / f"{document_id}.pagewise.json"
    export_pagewise_stage.run(doc_json_path, pagewise_path)

    print("\n=== Stage 4/5: Chunk + embed ===")
    chunks_path = output_dir / f"{document_id}.chunks.json"
    run_chunk_embed_stage.run(markdown_json_path, chunks_path)

    print("\n=== Stage 5/5: Question bank (detect -> answer -> validate -> escalate) ===")
    markdown_document = MarkdownDocument.model_validate_json(markdown_json_path.read_text(encoding="utf-8"))

    all_questions = []
    for page in markdown_document.pages:
        detected = detect_questions_on_page_with_context(markdown_document, page.page_number)
        all_questions.extend(detected)
        print(f"  page {page.page_number}: detected {len(detected)} question(s)")

    print(f"\nTotal detected: {len(all_questions)}. Extracting answers...")
    answered = extract_answers_for_questions(markdown_document, all_questions)

    print("Validating...")
    validated = validate_answered_questions(answered)
    uncertain_count = sum(1 for r in validated if r.verdict == "uncertain")
    print(f"  good={len(validated) - uncertain_count} uncertain={uncertain_count}")

    print("Escalating uncertain items via targeted RAG (gpt-4o-mini, real vector search)...")
    final = escalate_all_uncertain(validated)
    still_uncertain = sum(1 for r in final if r.verdict == "uncertain")
    print(f"  after escalation: good={len(final) - still_uncertain} uncertain={still_uncertain}")

    xlsx_path = output_dir / f"{document_id}.question_bank.xlsx"
    write_question_bank_xlsx(final, xlsx_path)
    print(f"\nWrote question bank to {xlsx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--output-dir", default=Path("output"), type=Path)
    args = parser.parse_args()

    configure_logging()
    document_id = args.doc_id or args.input.stem
    run(args.input, document_id, args.output_dir)
