"""
The common normalized representation every extraction engine (PyMuPDF, PyMuPDF4LLM,
Docling, Marker, PaddleOCR) must produce, regardless of which one actually ran on a
given page. This is the backbone of the whole pipeline and doubles as LangGraph state
once orchestration is added.

Design rules carried over from the accepted plan:
- Every element carries full provenance (never lose the path back to the source pixel).
- Answers are extraction, not generation: answer_status is explicit and
  EXACT is always preferred over INFERRED, and NOT_FOUND is a valid, kept, terminal state.
- This is extraction, not generation — content fields should always be verbatim
  source text unless answer_status says otherwise.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


def new_element_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Geometry / provenance
# --------------------------------------------------------------------------- #

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class Provenance(BaseModel):
    document_id: str
    page_number: int
    bbox: BoundingBox | None = None
    parser: str                       # which engine produced this element, e.g. "pymupdf", "docling"
    parser_confidence: float | None = None
    element_id: str
    source_crop_path: str | None = None  # path to a saved image crop of this region, if extracted


# --------------------------------------------------------------------------- #
# Document elements
# --------------------------------------------------------------------------- #

class ElementType(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    QUESTION_STEM = "question_stem"
    QUESTION_SUBPART = "question_subpart"
    MCQ_OPTION = "mcq_option"
    ANSWER = "answer"
    ANSWER_SUBPART = "answer_subpart"
    EXPLANATION = "explanation"
    RELATED_THEORY = "related_theory"
    TABLE = "table"
    IMAGE = "image"
    DIAGRAM = "diagram"
    EQUATION = "equation"
    CAPTION = "caption"
    LEARNING_OBJECTIVE = "learning_objective"
    SECTION_HEADING = "section_heading"
    RUNNING_HEADER_FOOTER = "running_header_footer"
    PAGE_NUMBER = "page_number"
    DECORATIVE = "decorative"
    UNKNOWN = "unknown"


class TableData(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class DocumentElement(BaseModel):
    """One structural unit of a document: a paragraph, a table, a figure, a question
    stem, an OCR'd text block — anything the extraction engines detect on a page."""

    id: str = Field(default_factory=lambda: new_element_id("el"))
    type: ElementType
    content: str | None = None                 # verbatim text content, if any
    table_data: TableData | None = None
    latex: str | None = None                    # for EQUATION elements
    asset_path: str | None = None               # for IMAGE/DIAGRAM elements (saved crop)
    caption: str | None = None
    reading_order: int | None = None            # position within the page's reconciled reading order
    provenance: list[Provenance] = Field(default_factory=list)
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)


class Page(BaseModel):
    page_number: int
    width: float | None = None
    height: float | None = None
    has_native_text: bool = False
    column_count: int | None = None             # 1 or 2 typically; None if undetermined
    image_path: str | None = None                # rendered page image, always kept as ground truth
    element_ids: list[str] = Field(default_factory=list)  # in reconciled reading order


class Document(BaseModel):
    id: str = Field(default_factory=lambda: new_element_id("doc"))
    source_path: str
    source_format: Literal["pdf", "docx", "pptx", "image", "unknown"]
    title: str | None = None
    subject: str | None = None
    grade: str | None = None
    publisher: str | None = None
    pages: list[Page] = Field(default_factory=list)
    elements: dict[str, DocumentElement] = Field(default_factory=dict)  # id -> element

    def add_element(self, element: DocumentElement) -> DocumentElement:
        self.elements[element.id] = element
        return element


# --------------------------------------------------------------------------- #
# Question / Answer / QA pair
# --------------------------------------------------------------------------- #

class Question(BaseModel):
    id: str = Field(default_factory=lambda: new_element_id("q"))
    element_ids: list[str]                       # stem + subparts + options + inline diagrams
    question_number: str | None = None
    question_type: str | None = None             # MCQ | SA-I | SA-II | LA | open_ended | ...
    marks: float | None = None                    # parsed numeric value, e.g. 2.0
    marks_raw: str | None = None                  # verbatim source text, e.g. "[2 Marks]"
    confidence: float = 0.0
    source_pages: list[int] = Field(default_factory=list)


AnswerStatus = Literal["exact", "multi_source", "inferred", "not_found"]


class Answer(BaseModel):
    id: str = Field(default_factory=lambda: new_element_id("a"))
    source_element_ids: list[str] = Field(default_factory=list)
    answer_status: AnswerStatus = "not_found"
    text: str | None = None                       # verbatim (exact/multi_source) or flagged (inferred)
    confidence: float = 0.0
    exact_source_available: bool = False


class QuestionAnswerPair(BaseModel):
    id: str = Field(default_factory=lambda: new_element_id("qa"))
    document_id: str
    question: Question
    answer: Answer
    supporting_elements: list[str] = Field(default_factory=list)
    review_flag: bool = False
    cross_check_flags: list[str] = Field(default_factory=list)
