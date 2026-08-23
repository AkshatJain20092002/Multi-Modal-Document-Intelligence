"""
Milestone 1 runner — per the accepted plan, this is deliberately NOT LangGraph and
NOT question detection yet. It proves the normalized-document layer works:

    profiler -> deterministic router -> extraction engine(s) -> common Pydantic
    Document (page + bbox + reading order + element-type separation) -> graph

Usage:
    python scripts/run_pipeline.py --input sample_data --doc-id educart-science-ix-sound
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings
from app.extraction.docling_engine import DoclingEngine
from app.extraction.ocr_chain import OCRChainEngine
from app.extraction.pymupdf_engine import PyMuPDFEngine
from app.extraction.router import select_extraction_path
from app.ingestion.file_detector import detect_file_format
from app.ingestion.profiler import profile_document
from app.normalization.schema import Document
from app.observability.logging_config import configure_logging
from app.observability.operation_logger import log_operation
from app.structure.graph import DocumentGraph

logger = logging.getLogger(__name__)


def _relocate_asset(current_path: str, assets_dir: Path, real_page_number: int) -> str:
    """Moves a crop from its scratch location (see build_document_from_image_folder)
    into the real assets_dir/page_<N>/ folder."""
    src = Path(current_path)
    if not src.exists():
        return current_path
    dest_dir = assets_dir / f"page_{real_page_number}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))
    return str(dest)


def build_document_from_image_folder(folder: Path, document_id: str) -> Document:
    """Treat a folder of page images (like sample_data/) as one multi-page scanned
    document — each image is one page, OCR'd independently, merged in filename order."""
    image_paths = sorted(
        p for p in folder.iterdir() if detect_file_format(p) == "image"
    )
    if not image_paths:
        raise ValueError(f"No image pages found in {folder}")

    engine = OCRChainEngine()
    document = Document(id=document_id, source_path=str(folder), source_format="image")
    assets_dir = settings.data_dir / "assets" / document_id

    with log_operation("build_document_from_image_folder", folder=folder.name, page_count=len(image_paths)):
        for page_number, image_path in enumerate(image_paths, start=1):
            # Each per-image extract() call has no idea it's actually page N of a
            # larger virtual document -- DoclingEngine always numbers a standalone
            # image as "page 1" internally, and its assets_dir is derived straight
            # from document_id. Passing the SAME document_id for every page would
            # mean every page's crops land in the identical assets_dir/page_1/
            # folder, so a later page's Docling call would silently overwrite an
            # earlier page's same-named crop (every page's first picture is
            # 'picture_1.png') before this loop even finishes -- confirmed as a
            # real data-loss bug, not just a labeling issue. A page-unique scratch
            # document_id forces each call into its own folder, so no two pages
            # can ever collide regardless of processing order; the crops are then
            # relocated into the real assets_dir/page_<N>/ folder and the scratch
            # folder is discarded.
            scratch_id = f"{document_id}__scratch_page{page_number:04d}"
            page_doc = engine.extract(image_path, document_id=scratch_id)

            # page_doc has exactly one Page (page_number=1); re-number and merge.
            page = page_doc.pages[0]
            page.page_number = page_number
            for element_id, element in page_doc.elements.items():
                for prov in element.provenance:
                    prov.page_number = page_number
                    prov.document_id = document_id  # was set to scratch_id inside extract()
                if element.asset_path:
                    element.asset_path = _relocate_asset(element.asset_path, assets_dir, page_number)
                document.elements[element_id] = element
            document.pages.append(page)

            scratch_dir = settings.data_dir / "assets" / scratch_id
            if scratch_dir.exists():
                shutil.rmtree(scratch_dir, ignore_errors=True)

            logger.info(
                "page.processed page=%d/%d file=%s elements=%d",
                page_number, len(image_paths), image_path.name, len(page.element_ids),
            )

    return document


def run(input_path: Path, document_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with log_operation("run_pipeline", document_id=document_id, input=str(input_path)):
        if input_path.is_dir():
            # Folder-of-page-images case (sample_data/).
            logger.info("input.detected kind=image_folder path=%s", input_path)
            document = build_document_from_image_folder(input_path, document_id)
        else:
            profile = profile_document(input_path)
            path_choice = select_extraction_path(profile)

            # docx has no native page concept, so profiler.py already rendered it
            # to a real, paginated PDF (profile.converted_pdf_path) — extract from
            # that instead of the original .docx so page boundaries are genuine.
            extract_path = Path(profile.converted_pdf_path) if profile.converted_pdf_path else input_path

            if path_choice == "pymupdf_first":
                engine = PyMuPDFEngine()
            elif path_choice in ("ocr_layout_first", "ocr_chain"):
                # Scanned PDF or a standalone image: needs OCR, run the full
                # RapidOCR -> EasyOCR -> vision-LLM fallback chain.
                engine = OCRChainEngine()
            elif path_choice == "docling":
                # PPTX: Docling parses it natively from its XML content, no
                # OCR/fallback chain involved — ocr_backend is irrelevant here
                # but required by the constructor.
                engine = DoclingEngine(ocr_backend="rapidocr")
            else:
                raise ValueError(f"Unsupported extraction path '{path_choice}' for {input_path}")
            document = engine.extract(extract_path, document_id=document_id)

            if profile.converted_pdf_path:
                # Restore the document's identity to the original .docx now that
                # extraction (from the converted PDF) is done.
                document.source_path = str(input_path)
                document.source_format = "docx"

        with log_operation("build_graph", document_id=document_id):
            graph = DocumentGraph()
            graph.add_document(document)

        total_elements = len(document.elements)
        logger.info(
            "pipeline.summary pages=%d elements=%d graph_nodes=%d graph_edges=%d",
            len(document.pages), total_elements,
            graph.g.number_of_nodes(), graph.g.number_of_edges(),
        )

        out_path = output_dir / f"{document_id}.json"
        out_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        logger.info("output.written path=%s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--doc-id", required=None)
    parser.add_argument("--output-dir", default=Path("output"), type=Path)
    args = parser.parse_args()

    configure_logging()
    document_id = args.doc_id or args.input.stem
    run(args.input, document_id, args.output_dir)
