"""
Answer extraction — Phase 2 of the whole-document extraction pipeline.

For each DetectedQuestion (question_detector.py's output, one page's worth of
questions, verbatim), builds that question's page's P-1/P/P+1 window
(page_window.py, deterministic, no vector search) and asks gpt-4o-mini
whether that window contains a direct answer — reusing the same "verbatim
answer only, no related-fact false positives" rigor already validated and
fixed in rag_query.py's prompt (the exact bug this catches: a document
saying 'height is 0.15 m more than width' is NOT an answer to 'what is the
height', even though it's on-topic).

Deliberately standalone — no reuse of app.normalization.schema's Question/
Answer models, per "forget about Milestone 2" for this whole pipeline.
answer_status still follows exact/multi_source/inferred/not_found, because
that's a project-wide extraction-not-generation invariant, not a
Milestone-2-specific coupling.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from app.llm.router import LLMRouter
from app.normalization.markdown_render import MarkdownDocument
from app.retrieval.context_builder import DocumentContext
from app.retrieval.page_window import page_windows
from app.retrieval.question_detector import DetectedQuestion

AnswerStatus = Literal["exact", "multi_source", "inferred", "not_found"]

_SYSTEM_PROMPT = (
    "You are a document answer-finding engine, not an assistant. You are given a "
    "specific question (already detected, verbatim, from this document) and Markdown "
    "text from consecutive pages of the same document, each page marked with an HTML "
    "comment like '<!-- page 3 -->'. Determine whether these pages contain a direct "
    "answer to THIS EXACT question, and if so, return it verbatim — never generate, "
    "guess, paraphrase, or fall back on outside knowledge.\n\n"
    "Decide answer_status by these rules, with no exceptions:\n"
    "- 'exact': the answer is explicitly and unambiguously stated on a single page "
    "(e.g. 'Ans. (c)', a worked solution, an explicit final value, a selected "
    "option).\n"
    "- 'multi_source': the answer is explicit, but assembling it required combining "
    "verbatim information that appears on more than one of the given pages.\n"
    "- 'inferred': no explicit answer is stated anywhere in the given pages, but it "
    "can be reasonably deduced from the question/options/context. You must still flag "
    "this as 'inferred' — never present a deduced answer as if it were explicitly "
    "extracted.\n"
    "- 'not_found': no answer can be determined from the given pages at all. Set "
    "answer_text=null.\n\n"
    "A direct answer means text that, on its own, actually answers this specific "
    "question — a final value, an explicit statement, a selected option. A RELATED "
    "fact that only mentions the same topic, or gives a relationship/constraint that "
    "would require further reasoning or computation to reach the answer, does NOT "
    "count as 'exact' or 'multi_source' — at best it supports 'inferred'. For "
    "example, if asked 'what is the height of the container' and the pages only say "
    "'the height is 0.15 m more than the width' with no width value given, that is "
    "NOT a direct answer.\n\n"
    "Copy the answer text exactly as printed — symbols, numerals, units, punctuation "
    "included. Do not invent or guess illegible content. List every page (from the "
    "'<!-- page N -->' markers) the answer actually came from in source_pages. "
    "confidence (0.0-1.0) should reflect how directly/unambiguously the pages support "
    "this answer_status."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_status": {
            "type": "string",
            "enum": ["exact", "multi_source", "inferred", "not_found"],
        },
        "answer_text": {"type": ["string", "null"]},
        "source_pages": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number"},
    },
    "required": ["answer_status", "answer_text", "source_pages", "confidence"],
    "additionalProperties": False,
}


class AnsweredQuestion(BaseModel):
    question: DetectedQuestion
    answer_status: AnswerStatus = "not_found"
    answer_text: str | None = None
    source_pages: list[int] = Field(default_factory=list)
    confidence: float = 0.0
    context_pages_used: list[int] = Field(default_factory=list)  # the full P-1/P/P+1 window sent to the LLM


def _question_block(question: DetectedQuestion) -> str:
    lines = [f"Question: {question.question_text}"]
    if question.options:
        lines.append("Options: " + " | ".join(question.options))
    return "\n".join(lines)


def extract_answer(question: DetectedQuestion, context: DocumentContext) -> AnsweredQuestion:
    raw = LLMRouter().call(
        tier="tier_1_cheap",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _question_block(question) + "\n\nDocument pages:\n\n" + context.combined_text(),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": _RESPONSE_SCHEMA},
        },
    )

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, AttributeError) as exc:
        raise RuntimeError(f"answer_extractor: gpt-4o-mini returned non-JSON output: {exc}") from exc

    return AnsweredQuestion(
        question=question,
        answer_status=parsed["answer_status"],
        answer_text=parsed.get("answer_text"),
        source_pages=parsed.get("source_pages", []),
        confidence=float(parsed.get("confidence", 0.0)),
        context_pages_used=[page.page_number for page in context.pages],
    )


def extract_answers_for_questions(
    markdown_document: MarkdownDocument, questions: list[DetectedQuestion]
) -> list[AnsweredQuestion]:
    """Batch entry point: builds every page's P-1/P/P+1 window once, then
    answers each question using its own page's window."""
    windows = {ctx.matched_page_numbers[0]: ctx for ctx in page_windows(markdown_document)}

    results: list[AnsweredQuestion] = []
    for question in questions:
        context = windows.get(question.page_number)
        if context is None:
            results.append(AnsweredQuestion(question=question))  # not_found default, no window available
            continue
        results.append(extract_answer(question, context))
    return results
