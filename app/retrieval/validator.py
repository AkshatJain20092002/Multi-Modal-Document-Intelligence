"""
Validation — Phase 3 of the whole-document extraction pipeline. Deterministic,
no LLM call: a set of plain rule-checks over each AnsweredQuestion, producing
a "good" or "uncertain" verdict with the specific reason(s) attached.

"uncertain" doesn't mean "wrong" — answer_status=not_found is flagged here
not because it's an error, but because it's worth a second look with a
different context-finding strategy (see escalation.py): the real answer may
live further away in the document than the fixed P-1/P/P+1 window could see.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.retrieval.answer_extractor import AnsweredQuestion

Verdict = Literal["good", "uncertain"]

MIN_QUESTION_CONFIDENCE = 0.7
MIN_ANSWER_CONFIDENCE = 0.7
MIN_QUESTION_TEXT_LENGTH = 5


class ValidationResult(BaseModel):
    answered_question: AnsweredQuestion
    verdict: Verdict = "good"
    reasons: list[str] = Field(default_factory=list)


def validate_answered_question(answered: AnsweredQuestion) -> ValidationResult:
    question = answered.question
    reasons: list[str] = []

    if not question.question_text or len(question.question_text.strip()) < MIN_QUESTION_TEXT_LENGTH:
        reasons.append("question_text is empty or too short")

    if question.confidence < MIN_QUESTION_CONFIDENCE:
        reasons.append(f"low detection confidence ({question.confidence:.2f})")

    if answered.answer_status in ("exact", "multi_source") and not answered.answer_text:
        reasons.append(f"answer_status={answered.answer_status} but answer_text is empty")

    if answered.answer_status == "inferred":
        reasons.append("answer_status=inferred (always flagged for review)")

    if answered.answer_status == "not_found":
        reasons.append("answer_status=not_found (worth a broader search)")

    if answered.confidence < MIN_ANSWER_CONFIDENCE:
        reasons.append(f"low answer confidence ({answered.confidence:.2f})")

    if answered.source_pages and not set(answered.source_pages).issubset(set(answered.context_pages_used)):
        reasons.append("answer cites a source page outside the context it was actually shown")

    verdict: Verdict = "uncertain" if reasons else "good"
    return ValidationResult(answered_question=answered, verdict=verdict, reasons=reasons)


def validate_answered_questions(answered: list[AnsweredQuestion]) -> list[ValidationResult]:
    return [validate_answered_question(a) for a in answered]
