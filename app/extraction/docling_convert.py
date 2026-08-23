"""Shared helper: DoclingDocument -> our normalized Document/DocumentElement schema.
Used by both DoclingEngine variants (RapidOCR/EasyOCR backends) so the mapping
logic only lives in one place."""

from __future__ import annotations

from pathlib import Path

from app.normalization.schema import (
    BoundingBox,
    Document,
    DocumentElement,
    ElementType,
    Page,
    Provenance,
    TableData,
    new_element_id,
)

_LABEL_TO_TYPE = {
    "section_header": ElementType.HEADING,
    "title": ElementType.HEADING,
    "text": ElementType.TEXT,
    "paragraph": ElementType.TEXT,
    "list_item": ElementType.TEXT,
    "caption": ElementType.CAPTION,
    "picture": ElementType.IMAGE,
    "table": ElementType.TABLE,
    "formula": ElementType.EQUATION,
    "page_header": ElementType.RUNNING_HEADER_FOOTER,
    "page_footer": ElementType.RUNNING_HEADER_FOOTER,
    "footnote": ElementType.RUNNING_HEADER_FOOTER,
}


def docling_document_to_normalized(
    docling_doc,
    *,
    document_id: str,
    source_path: str,
    source_format: str,
    parser_name: str,
    assets_dir: Path | None = None,
) -> Document:
    document = Document(id=document_id, source_path=source_path, source_format=source_format)
    crop_indices: dict[tuple[int, str], int] = {}

    def _save_crop(item, page_no: int, kind: str) -> str | None:
        """Pure geometric crop (page image -> bbox), no model inference.
        Used for both pictures and formulas — deliberately not attempting
        local LaTeX recognition (too slow on CPU); the crop alone is kept
        so nothing is silently lost, and LaTeX can be added later via a
        cheap on-demand OpenAI vision call per formula if ever needed.
        Nested under a per-page subfolder so a long document doesn't dump
        hundreds of crops into one flat directory."""
        if assets_dir is None:
            return None
        try:
            pil_image = item.get_image(docling_doc)
        except Exception:
            pil_image = None
        if pil_image is None:
            return None
        key = (page_no, kind)
        crop_indices[key] = crop_indices.get(key, 0) + 1
        page_dir = assets_dir / f"page_{page_no}"
        page_dir.mkdir(parents=True, exist_ok=True)
        out_path = page_dir / f"{kind}_{crop_indices[key]}.png"
        pil_image.save(out_path)
        return str(out_path)

    page_numbers = sorted(docling_doc.pages.keys()) if docling_doc.pages else [1]
    pages_by_number: dict[int, Page] = {}
    for page_no in page_numbers:
        page_info = docling_doc.pages.get(page_no)
        size = getattr(page_info, "size", None)
        pages_by_number[page_no] = Page(
            page_number=page_no,
            width=getattr(size, "width", None) if size else None,
            height=getattr(size, "height", None) if size else None,
            has_native_text=False,
        )

    order_counters: dict[int, int] = {}

    for item, _level in docling_doc.iterate_items():
        label = str(getattr(item, "label", "text")).lower()
        element_type = _LABEL_TO_TYPE.get(label, ElementType.TEXT)
        element_id = new_element_id("el")

        prov_list = getattr(item, "prov", None) or []
        page_no = prov_list[0].page_no if prov_list else 1
        bbox = None
        if prov_list and getattr(prov_list[0], "bbox", None) is not None:
            b = prov_list[0].bbox
            bbox = BoundingBox(x1=b.l, y1=b.t, x2=b.r, y2=b.b)

        content = None
        table_data = None
        latex = None
        asset_path = None

        if element_type == ElementType.TABLE and hasattr(item, "export_to_dataframe"):
            try:
                df = item.export_to_dataframe()
                table_data = TableData(headers=list(df.columns.astype(str)), rows=df.astype(str).values.tolist())
            except Exception:
                table_data = None
        elif element_type == ElementType.IMAGE:
            asset_path = _save_crop(item, page_no, "picture")
        elif element_type == ElementType.EQUATION:
            asset_path = _save_crop(item, page_no, "formula")
            raw_text = getattr(item, "text", None)
            latex = raw_text.strip() if raw_text else None
        else:
            content = getattr(item, "text", None)
            if content is not None:
                content = content.strip()

        if element_type not in (ElementType.TABLE, ElementType.IMAGE, ElementType.EQUATION) and not content:
            continue

        order_counters.setdefault(page_no, 0)
        reading_order = order_counters[page_no]
        order_counters[page_no] += 1

        element = DocumentElement(
            id=element_id,
            type=element_type,
            content=content,
            table_data=table_data,
            latex=latex,
            asset_path=asset_path,
            reading_order=reading_order,
            provenance=[
                Provenance(
                    document_id=document_id,
                    page_number=page_no,
                    bbox=bbox,
                    parser=parser_name,
                    element_id=element_id,
                )
            ],
        )
        document.add_element(element)

        page = pages_by_number.setdefault(page_no, Page(page_number=page_no))
        page.element_ids.append(element.id)

    document.pages = [pages_by_number[n] for n in sorted(pages_by_number)]
    return document
