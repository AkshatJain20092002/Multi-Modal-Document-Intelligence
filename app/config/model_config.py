"""LLM tier routing. Model *names* live in settings.py; this defines *when* each tier fires."""

from enum import Enum


class LLMTier(str, Enum):
    TIER_0_NO_LLM = "tier_0_no_llm"              # clear parser result, unambiguous numbering
    TIER_1_CHEAP = "tier_1_cheap"                # classification, ranking, continuation detection
    TIER_2_ESCALATION = "tier_2_escalation"      # competing candidates, cross-page/column ambiguity


ESCALATION_TRIGGERS = (
    "table_or_diagram_complexity",
    "multiple_competing_answer_candidates",
    "cross_page_ambiguity",
    "multi_column_uncertainty",
    "answer_key_association_ambiguity",
    "low_parser_agreement",
)
