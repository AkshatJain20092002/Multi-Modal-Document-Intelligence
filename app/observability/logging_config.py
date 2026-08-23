"""One place to configure logging for the whole pipeline. Call configure_logging()
once, at the top of any entrypoint (scripts/run_pipeline.py, tests, a future API
layer) — every module below just does `logging.getLogger(__name__)` and gets
the same format/level for free."""

from __future__ import annotations

import logging

from app.config.settings import settings

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True
