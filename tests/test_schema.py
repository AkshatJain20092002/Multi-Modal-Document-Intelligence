from app.normalization.schema import (
    Answer,
    BoundingBox,
    Document,
    DocumentElement,
    ElementType,
    Page,
    Provenance,
    Question,
    QuestionAnswerPair,
)


def test_document_add_element_roundtrip():
    document = Document(id="doc1", source_path="x.pdf", source_format="pdf")
    element = DocumentElement(
        type=ElementType.TEXT,
        content="hello",
        provenance=[
            Provenance(
                document_id="doc1",
                page_number=1,
                bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
                parser="pymupdf",
                element_id="placeholder",
            )
        ],
    )
    document.add_element(element)
    assert document.elements[element.id].content == "hello"


def test_answer_status_defaults_to_not_found_and_is_kept():
    answer = Answer()
    assert answer.answer_status == "not_found"
    assert answer.text is None


def test_qa_pair_requires_document_id():
    question = Question(element_ids=["el1"])
    answer = Answer(answer_status="exact", text="42", source_element_ids=["el1"])
    pair = QuestionAnswerPair(document_id="doc1", question=question, answer=answer)
    assert pair.answer.answer_status == "exact"
    assert pair.review_flag is False
