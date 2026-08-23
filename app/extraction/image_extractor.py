"""
Image extraction with caption matching, ported from the metadata-chunking approach in
the futureInnovationChatbot reference repo (metadataChunking/image_extraction.py) and
adapted to this repo's conventions: crops are saved to local disk (asset_path) instead
of S3, and results are consumed directly as DocumentElement objects rather than a
standalone JSON/Markdown export.

Two things the previous naive per-page image loop in PyMuPDFEngine didn't do:
- Filter tiny bbox/byte-count noise (icons, decorative artifacts).
- Match each image to its "Figure N" / "Fig. N" caption via geometric proximity,
  so downstream stages don't have to re-derive that association from raw bboxes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.normalization.schema import new_element_id


class ImageExtractor:
    _CAPTION_RE = re.compile(r"^(Figure|Fig\.?)\s*\d+", re.IGNORECASE)
    _MIN_BBOX_AREA = 2000
    _MIN_BYTES = 1000
    _ROW_BIN = 20  # tolerance (px) for treating two images as being on the same row

    def extract_captions_on_page(self, page: Any) -> list[dict]:
        captions = []
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            text = text.strip().replace("\n", " ")
            if self._CAPTION_RE.match(text):
                captions.append({"text": text, "bbox": (x0, y0, x1, y1), "y": y0})
        captions.sort(key=lambda c: c["y"])
        return captions

    @staticmethod
    def _x_overlap(a_bbox: tuple, b_bbox: tuple) -> float:
        ax0, _, ax1, _ = a_bbox
        bx0, _, bx1, _ = b_bbox
        return max(0.0, min(ax1, bx1) - max(ax0, bx0))

    def match_images_with_captions(self, images: list[dict], captions: list[dict]) -> None:
        """Mutates `images` in place, setting "caption" on the best-matched entry.
        Prefers a caption's nearest image above it (the common "figure, then caption
        below it" layout); falls back to the nearest image below otherwise."""
        if not images or not captions:
            return

        used: set[int] = set()
        for cap in captions:
            cap_bbox = cap["bbox"]
            cy_top = cap_bbox[1]
            above, below = [], []

            for idx, img in enumerate(images):
                if idx in used:
                    continue
                ix0, iy0, ix1, iy1 = img["bbox"]
                overlap = self._x_overlap(img["bbox"], cap_bbox)
                if iy1 <= cy_top:
                    above.append((overlap, -(cy_top - iy1), idx))
                else:
                    below.append((overlap, -abs(iy0 - cy_top), idx))

            chosen = None
            if above:
                above.sort(reverse=True)
                chosen = above[0][2]
            elif below:
                below.sort(reverse=True)
                chosen = below[0][2]

            if chosen is not None:
                images[chosen]["caption"] = cap["text"]
                used.add(chosen)

    def extract_page_images(self, pdf: Any, page: Any, page_number: int, assets_dir: Path) -> list[dict]:
        """Extract images on one page: bbox, saved crop path, and (if found) caption.
        Returns plain dicts; PyMuPDFEngine converts these to DocumentElements."""
        raw_images = []
        for image_info in page.get_images(full=True):
            xref = image_info[0]
            base = pdf.extract_image(xref)
            if not base or len(base.get("image", b"")) < self._MIN_BYTES:
                continue
            try:
                bbox = tuple(page.get_image_bbox(image_info))
            except Exception:
                bbox = (0.0, 0.0, 0.0, 0.0)
            w, h = max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1])
            if w * h < self._MIN_BBOX_AREA:
                continue
            raw_images.append({"xref": xref, "base": base, "bbox": bbox})

        # Row-wise reading order: left-to-right within a row, then top-to-bottom.
        raw_images.sort(key=lambda item: (round(item["bbox"][1] / self._ROW_BIN), item["bbox"][0]))

        # Nested under a per-page subfolder so a long document doesn't dump
        # hundreds of crops into one flat directory. A pre-generated element
        # id is still returned (as "element_id") so the caller can reuse the
        # same id on the DocumentElement it builds, even though the filename
        # itself stays positional (img_1.ext, img_2.ext, ...).
        page_dir = assets_dir / f"page_{page_number}"
        page_dir.mkdir(parents=True, exist_ok=True)
        page_images = []
        for idx, item in enumerate(raw_images, start=1):
            element_id = new_element_id("el")
            ext = item["base"]["ext"]
            image_bytes = item["base"]["image"]
            asset_path = page_dir / f"img_{idx}.{ext}"
            asset_path.write_bytes(image_bytes)
            page_images.append({
                "bbox": item["bbox"],
                "asset_path": str(asset_path),
                "caption": None,
                "element_id": element_id,
            })

        captions = self.extract_captions_on_page(page)
        self.match_images_with_captions(page_images, captions)
        return page_images
