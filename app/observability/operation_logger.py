"""
Structured operation logging, used the way an agent framework logs "intent
dispatched -> handler invoked -> result/error" — except there is no agent here,
just a plain context manager/decorator that every pipeline stage (profiling,
parser routing, each extraction engine, each LLM tier call) wraps itself with,
so the logs read as a trace of *which operation ran, with what inputs, and
whether it succeeded* — e.g.:

    23:41:02 INFO  app.extraction.ocr_chain :: operation.start  op=extract engine=docling[rapidocr] path=task_page-0002.jpg
    23:41:04 INFO  app.extraction.ocr_chain :: operation.success op=extract engine=docling[rapidocr] duration_ms=1840

This is the single place that decides the log line shape, so every stage's logs
are greppable/consistent without each module reinventing its own format.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any


@contextlib.contextmanager
def log_operation(operation: str, *, logger: logging.Logger | None = None, **context: Any):
    """Usable both as a context manager and (since it's built on
    @contextmanager, which returns a ContextDecorator) as a plain decorator:

        with log_operation("profile_document", path=str(path)):
            ...

        @log_operation("extract", engine="pymupdf")
        def extract(self, path, document_id): ...
    """
    log = logger or logging.getLogger("app.pipeline")
    ctx_str = " ".join(f"{k}={v}" for k, v in context.items())
    log.info("operation.start op=%s %s", operation, ctx_str)
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        log.warning(
            "operation.failed op=%s %s duration_ms=%d error=%s: %s",
            operation, ctx_str, duration_ms, type(exc).__name__, exc,
        )
        raise
    else:
        duration_ms = int((time.perf_counter() - start) * 1000)
        log.info("operation.success op=%s %s duration_ms=%d", operation, ctx_str, duration_ms)
