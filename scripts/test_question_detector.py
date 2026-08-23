"""
Manual smoke-test harness for app/retrieval/question_detector.py — run it
against one page (or every page) of an already-rendered Markdown manifest
(scripts/export_markdown.py's output).

Usage:
    python scripts/test_question_detector.py --markdown-json output/task_page-0016.markdown.json
    python scripts/test_question_detector.py --markdown-json output/SAS_Maths-Class-9_p6-10.markdown.json --page 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.normalization.markdown_render import MarkdownDocument
from app.observability.logging_config import configure_logging
from app.retrieval.question_detector import detect_questions_on_page_with_context


def run(markdown_json_path: Path, only_page: int | None) -> None:
    markdown_document = MarkdownDocument.model_validate_json(markdown_json_path.read_text(encoding="utf-8"))
    pages = markdown_document.pages if only_page is None else [
        p for p in markdown_document.pages if p.page_number == only_page
    ]

    total = 0
    for page in pages:
        questions = detect_questions_on_page_with_context(markdown_document, page.page_number)
        total += len(questions)
        print(f"\n--- page {page.page_number}: {len(questions)} question(s) ---")
        for q in questions:
            print(f"  [{q.question_number or '?'}] type={q.question_type} marks={q.marks_raw} "
                  f"diag={q.has_diagram_reference} conf={q.confidence:.2f}")
            print(f"    {q.question_text!r}")
            if q.options:
                print(f"    options: {q.options}")

    print(f"\nTotal: {total} question(s) across {len(pages)} page(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown-json", required=True, type=Path)
    parser.add_argument("--page", default=None, type=int, help="Only test this page number (default: all pages)")
    args = parser.parse_args()

    configure_logging()
    run(args.markdown_json, args.page)
