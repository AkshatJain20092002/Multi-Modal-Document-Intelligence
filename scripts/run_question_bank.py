"""
Full whole-document pipeline: detect -> answer -> validate -> escalate
uncertain -> CSV (the "Question Bank" deliverable).

Usage:
    python scripts/run_question_bank.py --markdown-json output/task_page-0016.markdown.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.normalization.markdown_render import MarkdownDocument
from app.observability.logging_config import configure_logging
from app.retrieval.answer_extractor import extract_answers_for_questions
from app.retrieval.escalation import escalate_all_uncertain
from app.retrieval.question_bank_export import write_question_bank_csv
from app.retrieval.question_detector import detect_questions_on_page_with_context
from app.retrieval.validator import validate_answered_questions


def run(markdown_json_path: Path, output_path: Path) -> None:
    markdown_document = MarkdownDocument.model_validate_json(markdown_json_path.read_text(encoding="utf-8"))

    all_questions = []
    for page in markdown_document.pages:
        detected = detect_questions_on_page_with_context(markdown_document, page.page_number)
        all_questions.extend(detected)
        print(f"page {page.page_number}: detected {len(detected)} question(s)")

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

    write_question_bank_csv(final, output_path)
    print(f"\nWrote question bank to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-json", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    output = args.output or args.markdown_json.with_suffix("").with_suffix(".question_bank.csv")
    configure_logging()
    run(args.markdown_json, output)
