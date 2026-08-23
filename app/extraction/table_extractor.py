"""
LLM-based table extraction, ported from the metadata-chunking approach in the
futureInnovationChatbot reference repo (metadataChunking/table_extraction.py).

Why an LLM here and not a deterministic parser: this repo has no dedicated table-
structure engine (Docling/Marker aren't wired up for tables yet in milestone 1), and
tables in scanned educational content are frequently borderless/full-page grids that
a bbox heuristic can't reliably segment. Per the accepted plan, deterministic parsing
is preferred everywhere it's viable; table extraction is one of the genuinely
ambiguous cases that's allowed to reach OpenAI (Tier 1 — see app.config.model_config,
"table_or_diagram_complexity" and "continuation detection" are both listed as
LLM-appropriate triggers).

Cross-page split-table merging is itself decided with a second, small Tier 1 call
rather than a heuristic, mirroring the reference implementation, because "does this
table continue the previous page's table" genuinely needs semantic judgement (same
headers repeated vs. a new unrelated table starting at the top of a page).

No caption-number ("Table N") fallback trickery beyond what the reference did:
callers get raw page-anchored tables back and decide how to attach them.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.llm.router import LLMRouter

logger = logging.getLogger(__name__)

_TABLE_NUMBER_RE = re.compile(r"(?:Table|Tab\.?)\s*(\d+)", re.IGNORECASE)


def _strip_code_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json\n", "", 1)
    return content


class TableExtractor:
    def __init__(self, llm_router: LLMRouter | None = None) -> None:
        self.llm = llm_router or LLMRouter()

    def _extract_tables_from_page(
        self, page_text: str, page_number: int, context_headers: list[str] | None
    ) -> list[dict]:
        context_instruction = ""
        if context_headers:
            context_instruction = f"""
            IMPORTANT CONTEXT FROM PREVIOUS PAGE:
            The previous page ended with a table having these headers: {context_headers}.
            Check the top of this page: does it contain rows that fit these headers?
            If YES, extract those rows as a table (headers may be empty [] for the
            continuation segment) and mark "is_continuation": true.
            """

        prompt = f"""
        You are a specialist in PDF table extraction.

        PAGE TEXT:
        \"\"\"{page_text}\"\"\"

        {context_instruction}

        TASK:
        1. Extract ALL tabular data found on this page.
        2. Treat borderless/full-page grid-structured text as a table even with no
           visible borders or "Table" label.
        3. If a table starts here or continues from the previous page, extract it as seen.

        MANDATORY VERIFICATION:
        - Verify the data is actually a table (aligned columns, repeated pattern) before extracting.
        - Do not extract a list of items or misaligned text as a table.
        - Do not extract references/bibliography sections as tables.

        OUTPUT FORMAT (strict JSON, no markdown fence):
        [{{"position": 1, "caption": "exact caption or null", "headers": ["Col 1", ...] , "rows": [["r1c1", "r1c2"], ...]}}]

        RULES:
        - "position": 1 for the first table on the page, 2 for the second, etc.
        - "headers": [] if this looks like a raw continuation (no header row visible).
        - "caption": null if no caption exists — never invent one.
        - Do not summarize data; extract raw text verbatim.
        """

        try:
            raw = self.llm.call(
                tier="tier_1_cheap",
                messages=[
                    {"role": "system", "content": "You are a JSON-only API. Return valid JSON lists. No Markdown formatting."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            parsed = json.loads(_strip_code_fence(raw))
            if not isinstance(parsed, list):
                return []
            return [
                {
                    "position": int(t.get("position", 999)),
                    "caption": t.get("caption"),
                    "headers": t.get("headers") or [],
                    "rows": [[str(c) for c in r] for r in t.get("rows", []) if isinstance(r, list)],
                    "page": page_number,
                }
                for t in parsed
                if isinstance(t, dict)
            ]
        except Exception as exc:
            logger.warning("table_extraction.page_failed page=%d error=%s", page_number, exc)
            return []

    def _check_continuation(self, prev_table: dict, curr_table: dict) -> bool:
        prev_fragment = {"headers": prev_table.get("headers", []), "last_rows": prev_table["rows"][-3:]}
        curr_fragment = {"headers": curr_table.get("headers", []), "first_rows": curr_table["rows"][:3]}
        prompt = f"""
        Check if "Table B" (page {curr_table['page']}) is a continuation of "Table A" (page {prev_table['page']}).

        Table A (end): {json.dumps(prev_fragment)}
        Table B (start): {json.dumps(curr_fragment)}

        DECISION RULES:
        1. YES if Table B has NO headers but its columns align with Table A.
        2. YES if Table B repeats the SAME headers as Table A.
        3. NO if Table B has different headers/structure, or a new caption (e.g. "Table 2").

        OUTPUT JSON: {{"is_continuation": true or false}}
        """
        try:
            raw = self.llm.call(
                tier="tier_1_cheap",
                messages=[
                    {"role": "system", "content": "You are a logical engine. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return bool(json.loads(_strip_code_fence(raw)).get("is_continuation", False))
        except Exception as exc:
            logger.warning("table_extraction.continuation_check_failed error=%s", exc)
            return False

    def _merge_split_tables(self, tables: list[dict]) -> list[dict]:
        if not tables:
            return []
        merged = []
        current = tables[0]
        for nxt in tables[1:]:
            is_sequential = nxt["page"] == current.get("page_end", current["page"]) + 1
            is_top_of_page = nxt.get("position", 1) == 1
            should_merge = is_sequential and is_top_of_page and self._check_continuation(current, nxt)
            if should_merge:
                current["rows"].extend(nxt["rows"])
                if not current["caption"] and nxt["caption"]:
                    current["caption"] = nxt["caption"]
                current["page_end"] = nxt["page"]
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
        return merged

    def extract(self, pdf: Any) -> list[dict]:
        """Whole-document table extraction with cross-page merge. `pdf` is an
        already-open fitz.Document. Returns page-anchored table dicts:
        {id, caption, page_start, page_end, headers, rows}."""
        raw_tables: list[dict] = []
        last_headers: list[str] | None = None
        last_page_had_table = False

        for page_index in range(pdf.page_count):
            page = pdf[page_index]
            page_number = page_index + 1
            page_text = page.get_text("text")
            context = last_headers if last_page_had_table else None

            page_tables = self._extract_tables_from_page(page_text, page_number, context)
            raw_tables.extend(page_tables)

            if page_tables:
                last_table = page_tables[-1]
                if last_table.get("headers"):
                    last_headers = last_table["headers"]
                last_page_had_table = len(last_table["rows"]) > 0
            else:
                last_headers = None
                last_page_had_table = False

        raw_tables.sort(key=lambda t: (t["page"], t["position"]))
        merged_tables = self._merge_split_tables(raw_tables)

        final: list[dict] = []
        for idx, t in enumerate(merged_tables, start=1):
            caption = t.get("caption")
            match = _TABLE_NUMBER_RE.search(caption) if caption else None
            table_id = int(match.group(1)) if match else idx
            final.append({
                "id": table_id,
                "caption": caption or f"Table {table_id} (Extracted)",
                "page_start": t["page"],
                "page_end": t.get("page_end", t["page"]),
                "headers": t["headers"],
                "rows": t["rows"],
            })

        logger.info("table_extraction.summary tables=%d", len(final))
        return final
