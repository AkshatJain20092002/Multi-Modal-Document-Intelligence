"""
OpenAI-only tiered LLM router. Tier 0 = no call at all (handled by callers before
reaching here). Tier 1 = cheap default model. Tier 2 = escalation, only on the
triggers listed in app.config.model_config.ESCALATION_TRIGGERS.

No Anthropic dependency anywhere in this module or its callers.
"""

from __future__ import annotations

from app.config.settings import settings
from app.observability.operation_logger import log_operation


class LLMRouter:
    primary_model = settings.openai_extraction_model
    fallback_model = settings.openai_reasoning_fallback

    def __init__(self) -> None:
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Set it in .env before any Tier 1/2 call runs."
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def call(self, *, tier: str, messages: list[dict], **kwargs) -> str:
        model = self.fallback_model if tier == "tier_2_escalation" else self.primary_model
        with log_operation("llm_call", tier=tier, model=model, message_count=len(messages)):
            client = self._client_or_raise()
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
            return response.choices[0].message.content or ""
