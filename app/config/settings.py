"""Central configuration. No secrets or model names hardcoded elsewhere in the pipeline."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenAI (only external LLM API assumed — no Anthropic) ---
    openai_api_key: str | None = None
    openai_extraction_model: str = "gpt-4o-mini"       # Tier 1 — cheap, default
    openai_reasoning_fallback: str = "gpt-4o"          # Tier 2 — escalation only

    # --- OCR chain (app/extraction/ocr_chain.py) ---
    ocr_engine_timeout_seconds: float = 60.0   # per-engine budget before falling back

    # --- Page-window policy ---
    page_window_default: int = 1          # local window before expansion
    page_window_ceiling: int = 5          # hard cost-control ceiling, not an accuracy ceiling
    page_window_backward: int = 1         # look-back when a question opens at a page break

    # --- Retrieval / embeddings (app/retrieval/) ---
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"  # local, no API key — retrieval-tuned
    chunk_max_tokens: int = 480    # headroom under bge-small-en-v1.5's hard 512-token limit

    # --- Storage (V1: local, lightweight — see plan for V2 graduation path) ---
    data_dir: Path = Path("output")
    sqlite_path: Path = Path("output") / "metadata.sqlite3"
    qdrant_path: Path = Path("output") / "qdrant_local"

    # --- Observability ---
    log_level: str = "INFO"


settings = Settings()
