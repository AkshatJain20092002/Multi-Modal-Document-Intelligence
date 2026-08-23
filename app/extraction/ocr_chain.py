"""
OCR engine chain:

    Docling+RapidOCR (base, no timeout) -> Docling+EasyOCR (fallback, 60s budget)
    -> gpt-4o-mini/gpt-4o vision (fallback, no timeout — last resort, nothing left
    to fall back to)

RapidOCR is deliberately given no time budget: it's the chosen primary engine
precisely because it's fast and dependency-light, so a slow run on it is a
signal something's actually wrong (not just "this page is hard") — better to
let it finish/fail on its own than silently swap engines mid-page. EasyOCR
(pure PyTorch, heavier) gets the settings.ocr_engine_timeout_seconds budget
(default 60s/page) since it's more plausibly slow-but-working depending on
hardware. If an engine raises OR (for EasyOCR only) blows its time budget,
the chain moves to the next one.

Surya is deliberately not in this chain: its recognition backend needs a
llama-server binary and, even when available, is a 650M-param VLM running on
CPU — too slow to be a viable per-page fallback (see conversation/project
notes). Marker was considered too but turned out to share the exact same
llama.cpp/VLM backend as Surya under the hood, so it doesn't sidestep the
problem either.

Note on the timeout mechanism: EasyOCR runs in a worker thread so a slow call
can be abandoned via future.result(timeout=...) without blocking the chain.
Python can't forcibly kill a running thread, so a timed-out run keeps
running in the background until it finishes on its own; accepted tradeoff
for V1 (favors moving on quickly over strict resource cleanup).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

from app.config.settings import settings
from app.extraction.base import ExtractionEngine
from app.extraction.docling_engine import DoclingEngine
from app.extraction.vision_llm_engine import VisionLLMEngine
from app.normalization.schema import Document

logger = logging.getLogger(__name__)


class OCRChainEngine(ExtractionEngine):
    name = "ocr_chain"

    def __init__(self) -> None:
        # (engine, timeout_seconds | None) — None means run inline, no budget.
        self._engines: list[tuple[ExtractionEngine, float | None]] = [
            (DoclingEngine(ocr_backend="rapidocr"), None),
            (DoclingEngine(ocr_backend="easyocr"), settings.ocr_engine_timeout_seconds),
            (VisionLLMEngine(), None),
        ]

    def extract(self, path: Path, document_id: str) -> Document:
        errors: list[str] = []

        for engine, timeout in self._engines:
            engine_name = getattr(engine, "name", engine.__class__.__name__)
            try:
                if timeout is None:
                    document = engine.extract(path, document_id=document_id)
                else:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(engine.extract, path, document_id=document_id)
                        document = future.result(timeout=timeout)
                if errors:
                    logger.info(
                        "OCR chain: %s succeeded on %s after failures: %s",
                        engine_name, path.name, "; ".join(errors),
                    )
                return document
            except FutureTimeoutError:
                errors.append(f"{engine_name}: exceeded {timeout:.0f}s budget")
                logger.warning(
                    "OCR chain: %s exceeded %.0fs budget on %s, falling back",
                    engine_name, timeout, path.name,
                )
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure moves to next engine
                errors.append(f"{engine_name}: {exc}")
                logger.warning("OCR chain: %s failed on %s: %s", engine_name, path.name, exc)

        raise RuntimeError(
            f"All OCR engines failed for {path}: " + " | ".join(errors)
        )
