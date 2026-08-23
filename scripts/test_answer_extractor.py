"""
Manual smoke-test harness: runs question_detector.py then answer_extractor.py
back to back over a Markdown manifest, so you can see the full detect ->
answer pipeline for real pages.

Usage:
    python scripts/test_answer_extractor.py --markdown-json output/task_page-0016.markdown.json
    python scripts/test_answer_extractor.py --markdown-json output/task_page-0016.markdown.json --page 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.normalization.markdown_render import MarkdownDocument
from app.observability.logging_config import configure_logging
from app.retrieval.answer_extractor import extract_answers_for_questions
from app.retrieval.question_detector import detect_questions_on_page_with_context


def run(markdown_json_path: Path, only_page: int | None) -> None:
    markdown_document = MarkdownDocument.model_validate_json(markdown_json_path.read_text(encoding="utf-8"))
    pages = markdown_document.pages if only_page is None else [
        p for p in markdown_document.pages if p.page_number == only_page
    ]

    all_questions = []
    for page in pages:
        all_questions.extend(detect_questions_on_page_with_context(markdown_document, page.page_number))

    print(f"Detected {len(all_questions)} question(s) across {len(pages)} page(s). Extracting answers...\n")
    answered = extract_answers_for_questions(markdown_document, all_questions)

    for a in answered:
        q = a.question
        print(f"[{q.question_number or '?'}] page={q.page_number} window={a.context_pages_used} "
              f"status={a.answer_status} conf={a.confidence:.2f}")
        print(f"  Q: {q.question_text!r}")
        if q.options:
            print(f"  options: {q.options}")
        print(f"  A: {a.answer_text!r} (source_pages={a.source_pages})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-json", required=True, type=Path)
    parser.add_argument("--page", default=None, type=int, help="Only test this page number (default: all pages)")
    args = parser.parse_args()

    configure_logging()
    run(args.markdown_json, args.page)
