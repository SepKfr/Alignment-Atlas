"""Extract figures/images from PDF pages using PyMuPDF."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.alignment_atlas.figures.schemas import ExtractedFigure, ImageBbox
from src.alignment_atlas.figures.utils import ensure_dir, safe_image_extension

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore


def extract_figures(
    pdf_path: str | Path,
    out_dir: str | Path,
    min_side_px: int = 200,
    min_figure_pt: float = 80.0,
) -> list[ExtractedFigure]:
    """
    Extract figures: embedded raster images first; on pages with none,
    vector graphic clusters (cluster_drawings) are rendered. No full-page screenshots.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for extract_figures. Install with: pip install PyMuPDF")

    pdf_path = Path(pdf_path)
    out_dir = ensure_dir(out_dir)
    doc_path_str = str(pdf_path.resolve())
    results: list[ExtractedFigure] = []

    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_1based = page_num + 1
            images = _get_embedded_images(page, doc)
            if images:
                for idx, (xref, bbox, img_bytes, ext) in enumerate(images):
                    w = bbox.x1 - bbox.x0
                    h = bbox.y1 - bbox.y0
                    if w < min_side_px or h < min_side_px:
                        logger.debug(
                            "Skip small image page %s idx %s size %.0fx%.0f",
                            page_1based,
                            idx,
                            w,
                            h,
                        )
                        continue
                    figure_id = f"p{page_1based}_fig{idx}"
                    suffix = safe_image_extension(ext)
                    image_name = f"{figure_id}{suffix}"
                    image_path = out_dir / image_name
                    image_path.write_bytes(img_bytes)
                    results.append(
                        ExtractedFigure(
                            doc_path=doc_path_str,
                            page=page_1based,
                            figure_id=figure_id,
                            image_path=str(image_path),
                            image_bbox=ImageBbox(
                                bbox.x0, bbox.y0, bbox.x1, bbox.y1
                            ),
                            source="embedded",
                        )
                    )
            else:
                vector_figures = _get_vector_figures(
                    page, page_1based, doc_path_str, out_dir, min_figure_pt
                )
                results.extend(vector_figures)
    finally:
        doc.close()

    logger.info("Extracted %s figures from %s", len(results), pdf_path)
    return results


def _get_vector_figures(
    page: "fitz.Page",
    page_1based: int,
    doc_path_str: str,
    out_dir: Path,
    min_figure_pt: float,
) -> list[ExtractedFigure]:
    """Extract vector-drawn figures via cluster_drawings(); render each cluster as PNG."""
    results: list[ExtractedFigure] = []
    if not hasattr(page, "cluster_drawings"):
        return results
    try:
        bboxes = page.cluster_drawings()
    except Exception:
        return results
    if not bboxes:
        return results
    for idx, rect in enumerate(bboxes):
        try:
            x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
        except AttributeError:
            continue
        if (x1 - x0) < min_figure_pt or (y1 - y0) < min_figure_pt:
            continue
        figure_id = f"p{page_1based}_fig{idx}"
        image_path = out_dir / f"{figure_id}.png"
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=rect, alpha=False)
            pix.save(str(image_path))
        except Exception:
            continue
        results.append(
            ExtractedFigure(
                doc_path=doc_path_str,
                page=page_1based,
                figure_id=figure_id,
                image_path=str(image_path),
                image_bbox=ImageBbox(x0, y0, x1, y1),
                source="rendered_region",
            )
        )
    return results


def _get_embedded_images(
    page: "fitz.Page",
    doc: "fitz.Document",
) -> list[tuple[int, ImageBbox, bytes, str]]:
    """Return list of (xref, bbox, image_bytes, ext) for embedded images on page."""
    out: list[tuple[int, ImageBbox, bytes, str]] = []
    image_list = page.get_images(full=True)
    for item in image_list:
        xref = item[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        if not rects:
            continue
        rect = rects[0]
        try:
            base = doc.extract_image(xref)
        except Exception:
            continue
        img_bytes = base.get("image")
        ext = base.get("ext", "png")
        if not img_bytes:
            continue
        bbox = ImageBbox(rect.x0, rect.y0, rect.x1, rect.y1)
        out.append((xref, bbox, img_bytes, ext))
    return out
