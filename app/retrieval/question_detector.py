"""
Question detection — Phase 1 of the whole-document extraction pipeline.

detect_questions_on_page() treats page P as the ONLY source of NEW questions
— but optionally sees a small boundary peek (last few blocks of P-1, first
few of P+1) so a question anchored on P whose subpart/continuation spills
across a page break isn't truncated or silently dropped. This is a
deliberate middle ground between two extremes:
  - page-alone (no peek): duplicate-proof by construction, but a question
    split across a page break gets truncated or missed entirely.
  - full P-1/P/P+1 (the earlier, reverted design): captures spanning
    questions fully, but reopens the same-question-extracted-twice problem
    that design was reverted over, requiring a real fuzzy-dedup step to fix.

The peek sections are explicitly marked CONTEXT ONLY in the prompt: the
model is told to never extract a new question whose stem lives entirely in
a peek section — only to use a peek to complete something already anchored
on page P. Symmetry makes this safe: when P-1 is later processed as its own
center page, ITS next-page-head peek is P's own head, so a question anchored
on P-1 that spills onto P still gets captured in full — just attributed to
P-1, not duplicated on P.

Answer-finding (a separate concern, needing its own P-1/P/P+1 window for a
different reason — the answer, not the question text, may be on a
neighboring page) is a deliberately separate later phase (answer_extractor.py).

This is the "high end exact prompt" piece — gpt-4o-mini stays the model
(cheap tier, per LLMRouter), the rigor is entirely in the prompt.
"""

from __future__ import annotations

import json
import uuid

from pydantic import BaseModel, Field

from app.llm.router import LLMRouter
from app.normalization.markdown_render import MarkdownDocument

_QUESTION_TYPES = [
    "MCQ", "SA-I", "SA-II", "LA", "fill_in_blank", "true_false",
    "match_the_following", "worked_example", "case_study", "open_ended", "other",
]

_SYSTEM_PROMPT = (
    "You are a verbatim question-detection engine, not an assistant. You are given "
    "the Markdown rendering of ONE page of an educational textbook/workbook — this is "
    "'THIS PAGE' below. Find and transcribe every distinct question anchored on THIS "
    "PAGE — nothing else.\n\n"
    "You may also be shown a small excerpt from the end of the previous page and/or "
    "the start of the next page, each clearly marked '[CONTEXT ONLY]'. These excerpts "
    "exist for exactly one purpose: so you can complete a question whose stem is on "
    "THIS PAGE but whose subpart, option list, or final sentence visibly continues "
    "into that excerpt. Rules for these excerpts:\n"
    "- NEVER extract a new question whose stem/opening text lives entirely inside a "
    "[CONTEXT ONLY] excerpt — that question belongs to that other page, not this one, "
    "and will be (or was) detected when that page is processed on its own.\n"
    "- You MAY append text from a [CONTEXT ONLY] excerpt onto a THIS-PAGE question's "
    "question_text/options ONLY if it is clearly a direct continuation (e.g. THIS "
    "PAGE ends mid-sentence, or ends with '(a) ... (b) ...' and the excerpt starts "
    "with '(c) ...').\n"
    "- If you're not confident an excerpt fragment continues a THIS-PAGE question, "
    "leave it out rather than guessing.\n\n"
    "WHAT COUNTS AS A QUESTION — extract each of these as its own entry:\n"
    "- Numbered exercise questions (e.g. '5.', 'Q5', '5)').\n"
    "- Worked examples (e.g. 'Example 3', 'Try This') — set question_type='worked_example' "
    "and question_number to the exact printed label (e.g. 'Example 3').\n"
    "- Case-study / passage-based question sets.\n"
    "- Any prompt that expects the reader to produce an answer, even if it has no "
    "visible number (e.g. a bare 'Simplify:' or 'What is the value of x?').\n\n"
    "SUBPARTS — a question with lettered/numbered subparts that each need a separate "
    "answer (e.g. '5. (a) ... (b) ... (c) ...') must be split into one entry PER "
    "SUBPART, each self-contained: question_number is 'parent(subpart)' (e.g. '5(a)', "
    "'5(b)'), and question_text repeats whatever shared stem/preamble text the subpart "
    "needs to stand alone, followed by that subpart's own text. Do NOT split a question "
    "that only has one blank/answer just because it's phrased with multiple sentences.\n\n"
    "WHAT IS NOT A QUESTION — never extract these:\n"
    "- Answers, worked solutions, or 'Ans.' lines.\n"
    "- 'Related Theory' / explanation / callout boxes.\n"
    "- Learning objectives, section headings, chapter titles, captions.\n"
    "- Running headers/footers, page numbers, decorative text.\n\n"
    "VERBATIM RULES — violating these corrupts a Q&A dataset downstream:\n"
    "- Copy question text and every option exactly as printed: symbols, roman "
    "numerals (I/II/III/IV), numbers, units (λ, ν, °, ×, →), and punctuation. Do not "
    "'fix', complete, or paraphrase anything, even if it looks like a typo or OCR "
    "error — transcribe exactly what is there.\n"
    "- If a character or word is genuinely illegible/garbled in the source, use '?' "
    "at that position rather than guessing.\n"
    "- options must be the exact printed option text, in printed order, with no "
    "invented options and none dropped. Leave options empty for non-MCQ questions.\n"
    "- marks_raw must be the exact printed substring for marks (e.g. '[2 Marks]', "
    "'(1 mark)') if and only if actually printed next to this question — otherwise "
    "null. Never invent a marks value.\n"
    "- question_type must be the closest match from the allowed list.\n"
    "- has_diagram_reference is true only if the question's stem or options "
    "explicitly depend on an image, diagram, table, or equation that appears on this "
    "page (e.g. 'In the figure below...', 'Using the table above...') — check for "
    "Markdown image links (![...](...)) or a Markdown table near this question.\n"
    "- confidence (0.0-1.0): how unambiguous this is as a genuine, complete question "
    "with clear boundaries — lower it if the page's OCR/rendering looks garbled or "
    "the question's start/end is unclear.\n\n"
    "Output structured JSON only, matching the provided schema. If this page has no "
    "questions at all, return an empty questions array — do not invent one."
)

_USER_PROMPT_PREFIX = "Detect every question on this page (page {page_number}):\n\n"

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_number": {"type": ["string", "null"]},
                    "question_type": {"type": "string", "enum": _QUESTION_TYPES},
                    "marks_raw": {"type": ["string", "null"]},
                    "question_text": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "has_diagram_reference": {"type": "boolean"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "question_number", "question_type", "marks_raw", "question_text",
                    "options", "has_diagram_reference", "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["questions"],
    "additionalProperties": False,
}


class DetectedQuestion(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    document_id: str
    page_number: int
    question_number: str | None = None
    question_type: str | None = None
    marks_raw: str | None = None
    question_text: str
    options: list[str] = Field(default_factory=list)
    has_diagram_reference: bool = False
    confidence: float = 0.0


def _tail_blocks(markdown: str, n: int) -> str:
    blocks = [b for b in markdown.split("\n\n") if b.strip()]
    return "\n\n".join(blocks[-n:])


def _head_blocks(markdown: str, n: int) -> str:
    blocks = [b for b in markdown.split("\n\n") if b.strip()]
    return "\n\n".join(blocks[:n])


def detect_questions_on_page(
    page_markdown: str,
    *,
    document_id: str,
    page_number: int,
    prev_page_tail: str | None = None,
    next_page_head: str | None = None,
) -> list[DetectedQuestion]:
    if not page_markdown.strip():
        return []

    sections = []
    if prev_page_tail:
        sections.append(
            f"[CONTEXT ONLY — end of page {page_number - 1}]\n\n{prev_page_tail}"
        )
    sections.append(f"[THIS PAGE — page {page_number}]\n\n{page_markdown}")
    if next_page_head:
        sections.append(
            f"[CONTEXT ONLY — start of page {page_number + 1}]\n\n{next_page_head}"
        )
    user_content = _USER_PROMPT_PREFIX.format(page_number=page_number) + "\n\n---\n\n".join(sections)

    raw = LLMRouter().call(
        tier="tier_1_cheap",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "detected_questions", "strict": True, "schema": _RESPONSE_SCHEMA},
        },
    )

    try:
        parsed = json.loads(raw)
        raw_questions = parsed.get("questions", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        raise RuntimeError(f"question_detector: gpt-4o-mini returned non-JSON output: {exc}") from exc

    return [
        DetectedQuestion(
            document_id=document_id,
            page_number=page_number,
            question_number=item.get("question_number"),
            question_type=item.get("question_type"),
            marks_raw=item.get("marks_raw"),
            question_text=item["question_text"],
            options=item.get("options", []),
            has_diagram_reference=bool(item.get("has_diagram_reference", False)),
            confidence=float(item.get("confidence", 0.0)),
        )
        for item in raw_questions
    ]


def detect_questions_on_page_with_context(
    markdown_document: MarkdownDocument, page_number: int, *, boundary_blocks: int = 3
) -> list[DetectedQuestion]:
    """Convenience wrapper: computes the prev/next boundary peeks automatically
    from a MarkdownDocument's own page list, then calls detect_questions_on_page().
    boundary_blocks is how many \\n\\n-separated blocks to peek into each
    neighbor — small on purpose, this is a boundary excerpt, not a full window."""
    pages_by_number = {page.page_number: page for page in markdown_document.pages}
    page = pages_by_number.get(page_number)
    if page is None:
        raise ValueError(f"page {page_number} not found in document {markdown_document.document_id}")

    prev_page = pages_by_number.get(page_number - 1)
    next_page = pages_by_number.get(page_number + 1)
    prev_page_tail = _tail_blocks(prev_page.markdown, boundary_blocks) if prev_page else None
    next_page_head = _head_blocks(next_page.markdown, boundary_blocks) if next_page else None

    return detect_questions_on_page(
        page.markdown,
        document_id=markdown_document.document_id,
        page_number=page_number,
        prev_page_tail=prev_page_tail,
        next_page_head=next_page_head,
    )
