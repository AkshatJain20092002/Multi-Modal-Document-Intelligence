"""
The relationship graph — NetworkX for V1 (per the accepted plan: do not make the
graph DB the primary store of everything; it represents relationships, not content).
Content/metadata live in the Document/DocumentElement Pydantic objects and, later,
SQLite. This module only tracks edges between element/question/answer ids.
"""

from __future__ import annotations

import networkx as nx

from app.normalization.schema import Document, QuestionAnswerPair


class DocumentGraph:
    def __init__(self) -> None:
        self.g = nx.DiGraph()

    def add_document(self, document: Document) -> None:
        self.g.add_node(document.id, kind="document", source_path=document.source_path)
        for page in document.pages:
            page_node = f"{document.id}:page:{page.page_number}"
            self.g.add_node(page_node, kind="page", page_number=page.page_number)
            self.g.add_edge(document.id, page_node, relation="has_page")

            prev_element_id: str | None = None
            for element_id in page.element_ids:
                element = document.elements[element_id]
                self.g.add_node(element_id, kind="element", type=element.type.value)
                self.g.add_edge(page_node, element_id, relation="has_element")
                if prev_element_id is not None:
                    self.g.add_edge(prev_element_id, element_id, relation="reading_order_next")
                prev_element_id = element_id

    def add_qa_pair(self, pair: QuestionAnswerPair) -> None:
        self.g.add_node(pair.question.id, kind="question")
        self.g.add_node(pair.answer.id, kind="answer", status=pair.answer.answer_status)
        self.g.add_edge(pair.question.id, pair.answer.id, relation="answered_by")
        for element_id in pair.question.element_ids:
            self.g.add_edge(pair.question.id, element_id, relation="composed_of")
        for element_id in pair.answer.source_element_ids:
            self.g.add_edge(pair.answer.id, element_id, relation="sourced_from")

    def neighbors(self, node_id: str, relation: str | None = None) -> list[str]:
        out = []
        for _, target, data in self.g.out_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation:
                out.append(target)
        return out
