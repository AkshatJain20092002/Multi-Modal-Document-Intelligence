"""
Vision-LLM OCR fallback — last resort in the chain, only reached if every local
OCR engine (Docling+RapidOCR, Docling+EasyOCR) fails or blows its time budget. Uses
OpenAI's gpt-4o-mini by default, escalating to gpt-4o via the same LLMRouter tiering
used elsewhere in the pipeline. No Anthropic dependency.

Unlike the earlier version of this engine (plain-text transcription -> one
whole-page TEXT element), this asks the model for structured output via
OpenAI's Structured Outputs (response_format=json_schema, strict mode) so the
result is comparable to what Docling produces: separate elements per
type (heading/question/option/answer/table/equation/...), tables as
headers+rows, and equations as LaTeX instead of losing them entirely.

Still no per-element bbox — a vision-LLM call cannot recover reliable
per-line geometry, so every element's provenance stays whole-page. That's
why this is the fallback of last resort, not a primary engine.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from app.config.settings import settings
from app.extraction.base import ExtractionEngine
from app.llm.router import LLMRouter
from app.normalization.schema import (
    Document,
    DocumentElement,
    ElementType,
    Page,
    Provenance,
    TableData,
)

_ELEMENT_TYPES = [
    "heading",
    "section_heading",
    "text",
    "question_stem",
    "question_subpart",
    "mcq_option",
    "answer",
    "answer_subpart",
    "explanation",
    "related_theory",
    "learning_objective",
    "table",
    "equation",
    "caption",
    "running_header_footer",
]

_SYSTEM_PROMPT = (
    "You are a verbatim document-extraction engine, not an assistant. You transcribe "
    "exactly what is printed on an educational textbook/workbook page image. You never "
    "paraphrase, correct, complete, or infer content that is not visibly present.\n\n"
    "Critical accuracy rules — violating these corrupts a Q&A dataset downstream:\n"
    "- Copy every character exactly as printed: option labels, roman numerals "
    "(I/II/III/IV), numbers, units, symbols (λ, ν, °, ×, →), and punctuation. Do NOT "
    "'fix' what looks like an inconsistency between an option list and its answer "
    "explanation — transcribe each occurrence independently, exactly as printed, even "
    "if they appear to disagree.\n"
    "- Never invent, complete, or guess text that is unclear, cut off, or ambiguous. "
    "If a character is genuinely illegible, use '?' at that position rather than "
    "guessing a plausible one.\n"
    "- Preserve reading order top-to-bottom, left-column-then-right-column for any "
    "multi-column layout — including when a paragraph/list item splits across the "
    "column break (continue it as one element only if it is genuinely one continuous "
    "sentence; otherwise keep them separate).\n"
    "- Every equation/formula (including inline ones like 'v = νλ' or stacked "
    "fractions) becomes its own 'equation' element with its content transcribed into "
    "the 'latex' field (plain LaTeX, no $ delimiters) — never folded into surrounding "
    "text, never skipped.\n"
    "- Every table becomes one 'table' element with 'table_headers' and 'table_rows' "
    "populated with the exact cell text — never flattened into prose.\n"
    "- Do not include page numbers, running headers/footers, or decorative elements "
    "unless they carry actual content (e.g. a citation tag like '[NCERT]' stays "
    "attached to the text it follows, not split out).\n"
    "- Output structured JSON only, matching the provided schema exactly."
)

_USER_PROMPT = (
    "Extract every content element from this page image into the structured schema, "
    "in correct reading order. Each element's 'type' must be the closest match from "
    "the allowed list. Set 'latex' only for 'equation' elements and 'table_headers'/"
    "'table_rows' only for 'table' elements; leave the other of those two fields null "
    "for every element."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": _ELEMENT_TYPES},
                    "content": {"type": ["string", "null"]},
                    "latex": {"type": ["string", "null"]},
                    "table_headers": {"type": ["array", "null"], "items": {"type": "string"}},
                    "table_rows": {
                        "type": ["array", "null"],
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "required": ["type", "content", "latex", "table_headers", "table_rows"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["elements"],
    "additionalProperties": False,
}


class VisionLLMEngine(ExtractionEngine):
    name = "vision_llm"

    def extract(self, path: Path, document_id: str) -> Document:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set — vision-LLM OCR fallback is unavailable."
            )

        image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

        raw = LLMRouter().call(
            tier="tier_1_cheap",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "page_elements",
                    "strict": True,
                    "schema": _RESPONSE_SCHEMA,
                },
            },
        )

        try:
            parsed = json.loads(raw)
            raw_elements = parsed.get("elements", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            raise RuntimeError(f"vision_llm returned non-JSON output: {exc}") from exc

        from PIL import Image

        with Image.open(path) as im:
            width, height = im.size

        document = Document(id=document_id, source_path=str(path), source_format="image")
        page = Page(page_number=1, width=width, height=height, has_native_text=False, image_path=str(path))

        for reading_order, raw_element in enumerate(raw_elements):
            type_str = raw_element.get("type", "text")
            try:
                element_type = ElementType(type_str)
            except ValueError:
                element_type = ElementType.TEXT

            content = raw_element.get("content")
            content = content.strip() if content else None
            latex = raw_element.get("latex")
            latex = latex.strip() if latex else None

            table_data = None
            headers, rows = raw_element.get("table_headers"), raw_element.get("table_rows")
            if element_type == ElementType.TABLE and (headers or rows):
                table_data = TableData(headers=headers or [], rows=rows or [])

            if not content and not latex and table_data is None:
                continue

            element = DocumentElement(
                type=element_type,
                content=content,
                latex=latex,
                table_data=table_data,
                reading_order=reading_order,
                provenance=[
                    Provenance(
                        document_id=document_id,
                        page_number=1,
                        bbox=None,
                        parser=self.name,
                        # Whole-page transcription, no per-element geometry — explicitly
                        # lower-confidence than a real OCR engine would produce.
                        parser_confidence=0.5,
                        element_id="",
                    )
                ],
            )
            element.provenance[0].element_id = element.id
            document.add_element(element)
            page.element_ids.append(element.id)

        document.pages.append(page)
        return document
