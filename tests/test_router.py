from app.extraction.router import select_extraction_path
from app.ingestion.profiler import DocumentProfile


def test_pdf_with_native_text_routes_to_pymupdf():
    profile = DocumentProfile(source_path="x.pdf", file_type="pdf", has_native_text=True)
    assert select_extraction_path(profile) == "pymupdf_first"


def test_pdf_without_native_text_routes_to_ocr():
    profile = DocumentProfile(source_path="x.pdf", file_type="pdf", has_native_text=False)
    assert select_extraction_path(profile) == "ocr_layout_first"


def test_docx_with_native_text_routes_to_pymupdf():
    # docx has no native page concept -- profiler.py renders it to a real PDF
    # first (docx_converter.py), then profiles that PDF for has_native_text,
    # so docx follows the exact same pdf-shaped routing as a native PDF.
    profile = DocumentProfile(source_path="x.docx", file_type="docx", has_native_text=True)
    assert select_extraction_path(profile) == "pymupdf_first"


def test_docx_without_native_text_routes_to_ocr():
    profile = DocumentProfile(source_path="x.docx", file_type="docx", has_native_text=False)
    assert select_extraction_path(profile) == "ocr_layout_first"


def test_image_routes_to_ocr_chain():
    profile = DocumentProfile(source_path="x.jpg", file_type="image")
    assert select_extraction_path(profile) == "ocr_chain"
