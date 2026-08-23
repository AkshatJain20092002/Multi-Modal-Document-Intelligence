"""
Manual smoke-test harness for individual extraction/OCR engines, run one at a
time against a single sample page image (or PDF, for pymupdf/docling).

Usage:
    python scripts/test_extraction_engine.py --engine docling_rapidocr --image sample_data/task_page-0016.jpg
    python scripts/test_extraction_engine.py --engine docling_easyocr  --image sample_data/task_page-0016.jpg
    python scripts/test_extraction_engine.py --engine vision_llm       --image sample_data/task_page-0016.jpg
    python scripts/test_extraction_engine.py --engine ocr_chain        --image sample_data/task_page-0016.jpg
    python scripts/test_extraction_engine.py --engine pymupdf          --image some_native_text.pdf

Prints element count, reading order, and a text preview; writes the full
normalized Document JSON to output/engine_test_<engine>_<image_stem>.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.extraction.docling_engine import DoclingEngine
from app.extraction.ocr_chain import OCRChainEngine
from app.extraction.pymupdf_engine import PyMuPDFEngine
from app.extraction.vision_llm_engine import VisionLLMEngine
from app.observability.logging_config import configure_logging

ENGINES = {
    "pymupdf": lambda: PyMuPDFEngine(),
    "docling_rapidocr": lambda: DoclingEngine(ocr_backend="rapidocr"),
    "docling_easyocr": lambda: DoclingEngine(ocr_backend="easyocr"),
    "vision_llm": lambda: VisionLLMEngine(),
    "ocr_chain": lambda: OCRChainEngine(),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=sorted(ENGINES))
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("output"), type=Path)
    args = parser.parse_args()

    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    engine = ENGINES[args.engine]()
    document_id = f"engine_test_{args.engine}_{args.image.stem}"

    start = time.perf_counter()
    document = engine.extract(args.image, document_id=document_id)
    elapsed = time.perf_counter() - start

    total_elements = len(document.elements)
    print(f"\nengine={args.engine} image={args.image.name} elapsed={elapsed:.2f}s")
    print(f"pages={len(document.pages)} elements={total_elements}")

    for page in document.pages:
        print(f"\n--- page {page.page_number} ({len(page.element_ids)} elements) ---")
        for element_id in page.element_ids[:10]:
            el = document.elements[element_id]
            preview = (el.content or el.asset_path or el.latex or "")[:80].replace("\n", " ")
            print(f"  [{el.reading_order}] {el.type.value:12s} {preview}")
        if len(page.element_ids) > 10:
            print(f"  ... ({len(page.element_ids) - 10} more)")

    out_path = args.output_dir / f"{document_id}.json"
    out_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nfull output written to {out_path}")


if __name__ == "__main__":
    main()
