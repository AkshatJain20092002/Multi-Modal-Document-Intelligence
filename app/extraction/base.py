"""Common interface every extraction engine implements, so callers never care
whether PyMuPDF, Docling, or a vision-LLM fallback actually did the work.

Every subclass's extract() is automatically wrapped with structured operation
logging (see app.observability.operation_logger) — this is the "intent handler"
style logging the whole repo follows: whichever engine actually gets invoked
for a given file logs its own start/success/failure, without each engine
needing to remember to add that logging itself.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from pathlib import Path

from app.normalization.schema import Document
from app.observability.operation_logger import log_operation


class ExtractionEngine(ABC):
    name: str = "base"

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        original_extract = cls.extract

        @functools.wraps(original_extract)
        def logged_extract(self, path: Path, document_id: str) -> Document:
            engine_name = getattr(self, "name", cls.__name__)
            with log_operation("extract", engine=engine_name, path=Path(path).name, document_id=document_id):
                return original_extract(self, path, document_id)

        cls.extract = logged_extract

    @abstractmethod
    def extract(self, path: Path, document_id: str) -> Document:
        """Parse the source file and return a normalized Document with pages +
        elements populated, all provenance-tagged with self.name."""
        raise NotImplementedError
