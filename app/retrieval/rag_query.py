"""
Standalone RAG query answering — deliberately NOT tied to Milestone 2. No reuse
of app.normalization.schema's Question/Answer models, no synthesized
DocumentElements, no element_ids. This answers one user question at a time
against whatever's already in the vector store:

    User query -> build_context() (Qdrant search -> P-1/P/P+1 window,
    already built/verified in context_builder.py) -> gpt-4o-mini -> RAGAnswer

Extraction, not generation: the model is instructed to search the given
context for a verbatim answer and return found=false rather than fabricate
one from outside knowledge if the document doesn't actually contain it.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from app.llm.router import LLMRouter
from app.retrieval.context_builder import build_context

_SYSTEM_PROMPT = (
    "You are a document search-and-answer engine, not a general assistant. You are "
    "given Markdown text from consecutive pages of a document (each page marked with "
    "an HTML comment like '<!-- page 3 -->') and a user's question. Determine whether "
    "this document actually contains an answer to the question, and if so, return it "
    "verbatim — never generate, guess, paraphrase, or fall back on outside knowledge.\n\n"
    "Rules:\n"
    "- If the document contains an explicit, verbatim answer to the question, set "
    "found=true and answer_text to the exact text from the document (copy character "
    "for character — do not paraphrase or summarize).\n"
    "- A direct answer means text that, on its own, actually answers the specific "
    "question asked — a final value, an explicit statement, a selected option. A "
    "RELATED fact that only mentions the same topic, gives a relationship/constraint "
    "that would require further reasoning or computation to reach the answer, or "
    "provides context without stating the answer itself does NOT count as found. For "
    "example, if asked 'what is the height of container X' and the document only "
    "says 'the height is 0.15 m more than the width' (no width given, no final "
    "number), that is NOT a direct answer — set found=false. Do not lower your "
    "standard just because something on-topic exists nearby.\n"
    "- If the document discusses the topic but never gives a direct answer, or the "
    "question isn't addressed at all, set found=false and answer_text=null. Do NOT "
    "invent an answer from general/outside knowledge, even if you happen to know the "
    "real answer — this document may not agree with it, or may not cover it at all.\n"
    "- If a matching question exists in the document (e.g. this is a Q&A workbook), "
    "also return matched_question_text with that question's exact printed text, so "
    "the caller can confirm this is really the same question the user asked about.\n"
    "- List every page (from the '<!-- page N -->' markers) the answer actually came "
    "from in source_pages.\n"
    "- confidence (0.0-1.0) should reflect how directly/unambiguously the document "
    "supports this answer — low if it required piecing together implied information, "
    "high if it's an explicit stated answer."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "matched_question_text": {"type": ["string", "null"]},
        "answer_text": {"type": ["string", "null"]},
        "source_pages": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number"},
    },
    "required": ["found", "matched_question_text", "answer_text", "source_pages", "confidence"],
    "additionalProperties": False,
}


class RAGAnswer(BaseModel):
    query: str
    document_id: str
    found: bool
    matched_question_text: str | None = None
    answer_text: str | None = None
    source_pages: list[int] = Field(default_factory=list)
    confidence: float = 0.0
    context_pages_used: list[int] = Field(default_factory=list)  # the full P-1/P/P+1 window sent to the LLM


def answer_query(query: str, *, document_id: str | None = None, top_k: int = 5) -> list[RAGAnswer]:
    """One RAGAnswer per document that vector search matched (usually one, if
    document_id is given; can be more than one for an unfiltered query that
    hits several documents — see context_builder.py's cross-document grouping)."""
    contexts = build_context(query, top_k=top_k, document_id=document_id)
    results: list[RAGAnswer] = []

    for context in contexts:
        raw = LLMRouter().call(
            tier="tier_1_cheap",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nDocument pages:\n\n{context.combined_text()}",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "rag_answer", "strict": True, "schema": _RESPONSE_SCHEMA},
            },
        )

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, AttributeError) as exc:
            raise RuntimeError(f"rag_query: gpt-4o-mini returned non-JSON output: {exc}") from exc

        results.append(
            RAGAnswer(
                query=query,
                document_id=context.document_id,
                found=parsed["found"],
                matched_question_text=parsed.get("matched_question_text"),
                answer_text=parsed.get("answer_text"),
                source_pages=parsed.get("source_pages", []),
                confidence=float(parsed.get("confidence", 0.0)),
                context_pages_used=[page.page_number for page in context.pages],
            )
        )

    return results
