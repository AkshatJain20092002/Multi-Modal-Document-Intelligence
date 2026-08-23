"""
Escalation — for AnsweredQuestions the deterministic validator marked
"uncertain" only. Retries with a DIFFERENT context-building strategy:
targeted RAG (a real vector search using the question's own text as the
query, against that document's Qdrant collection) instead of the fixed
P-1/P/P+1 window — since the real answer may live further away in the
document than immediate neighbor pages.

Model stays gpt-4o-mini throughout, not escalated to gpt-4o: the fix here is
a better context-finding strategy, not a smarter model, per explicit
instruction.

Never silently downgrades — a retry only replaces the original result if
it's actually better (fewer validation reasons, or equal reasons with higher
confidence).
"""

from __future__ import annotations

from app.retrieval.answer_extractor import extract_answer
from app.retrieval.context_builder import build_context
from app.retrieval.validator import ValidationResult, validate_answered_question


def escalate_uncertain(result: ValidationResult) -> ValidationResult:
    if result.verdict == "good":
        return result

    question = result.answered_question.question
    contexts = build_context(question.question_text, top_k=3, document_id=question.document_id)
    if not contexts:
        return result  # nothing indexed for this document in Qdrant -- can't retry

    retried = extract_answer(question, contexts[0])
    retried_result = validate_answered_question(retried)

    if len(retried_result.reasons) < len(result.reasons):
        return retried_result
    if len(retried_result.reasons) == len(result.reasons) and retried.confidence > result.answered_question.confidence:
        return retried_result
    return result


def escalate_all_uncertain(results: list[ValidationResult]) -> list[ValidationResult]:
    return [escalate_uncertain(r) if r.verdict == "uncertain" else r for r in results]
