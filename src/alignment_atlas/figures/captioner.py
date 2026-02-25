"""Match extracted figures to caption text on the same PDF page."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.alignment_atlas.figures.schemas import ExtractedFigure, ImageBbox

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore

# Horizontal overlap threshold to consider text "below" or "above" a figure
OVERLAP_THRESHOLD = 0.2
# Max vertical gap (px) to merge consecutive caption lines
CAPTION_LINE_GAP_PX = 15.0
# Boost for blocks that look like caption starters
CAPTION_START_BOOST = 1000.0

# Pattern for "Figure N" / "Fig. N" / "Table N" at start of text
CAPTION_START_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|Table|TABLE)\s*(?:\d+[\.:]?\s*)?",
    re.IGNORECASE,
)


def get_caption_for_figure(
    page: "fitz.Page",
    figure_bbox: ImageBbox,
    *,
    prefer_below: bool = True,
) -> Optional[str]:
    """
    Find the best caption for a figure on the given page.
    Prefer text below the figure; fall back to above if nothing below.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for caption matching.")

    text_dict = page.get_text("dict")
    blocks = _collect_blocks(text_dict)
    if not blocks:
        return None

    if prefer_below:
        caption = _best_caption_below(figure_bbox, blocks)
        if caption is None:
            caption = _best_caption_above(figure_bbox, blocks)
    else:
        caption = _best_caption_above(figure_bbox, blocks)
        if caption is None:
            caption = _best_caption_below(figure_bbox, blocks)

    return caption


def _collect_blocks(text_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten page dict into list of blocks with bbox and text."""
    out: list[dict[str, Any]] = []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            bbox = line.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            text_parts = []
            for span in line.get("spans", []):
                text_parts.append(span.get("text", ""))
            text = " ".join(text_parts).strip()
            if not text:
                continue
            out.append({
                "bbox": (bbox[0], bbox[1], bbox[2], bbox[3]),
                "text": text,
            })
    return out


def _horizontal_overlap(bbox_fig: ImageBbox, bbox_text: tuple[float, float, float, float]) -> float:
    """Fraction of figure width overlapped by text block (0..1)."""
    x0_f, x1_f = bbox_fig.x0, bbox_fig.x1
    x0_t, x1_t = bbox_text[0], bbox_text[2]
    overlap_start = max(x0_f, x0_t)
    overlap_end = min(x1_f, x1_t)
    if overlap_end <= overlap_start:
        return 0.0
    return (overlap_end - overlap_start) / (x1_f - x0_f) if (x1_f - x0_f) > 0 else 0.0


def _vertical_distance_below(fig_bbox: ImageBbox, text_bbox: tuple[float, float, float, float]) -> float:
    """Vertical distance from figure bottom to text top. Positive = text is below."""
    return text_bbox[1] - fig_bbox.y1


def _vertical_distance_above(fig_bbox: ImageBbox, text_bbox: tuple[float, float, float, float]) -> float:
    """Vertical distance from text bottom to figure top. Positive = text is above."""
    return fig_bbox.y0 - text_bbox[3]


def _caption_score(block: dict[str, Any], distance: float, is_below: bool) -> float:
    """Lower is better: we want small positive distance and caption-like start."""
    score = distance if distance >= 0 else 1e6
    if CAPTION_START_RE.search(block["text"]):
        score -= CAPTION_START_BOOST
    return score


def _best_caption_below(
    figure_bbox: ImageBbox,
    blocks: list[dict[str, Any]],
) -> Optional[str]:
    """Best caption among blocks below the figure (horizontal overlap > threshold)."""
    candidates: list[tuple[float, dict[str, Any]]] = []
    for b in blocks:
        ov = _horizontal_overlap(figure_bbox, b["bbox"])
        if ov < OVERLAP_THRESHOLD:
            continue
        dist = _vertical_distance_below(figure_bbox, b["bbox"])
        if dist < 0:
            continue
        score = _caption_score(b, dist, is_below=True)
        candidates.append((score, b))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    best_block = candidates[0][1]
    return _merge_consecutive_caption_blocks(best_block, blocks, figure_bbox, below=True)


def _best_caption_above(
    figure_bbox: ImageBbox,
    blocks: list[dict[str, Any]],
) -> Optional[str]:
    """Best caption among blocks above the figure."""
    candidates: list[tuple[float, dict[str, Any]]] = []
    for b in blocks:
        ov = _horizontal_overlap(figure_bbox, b["bbox"])
        if ov < OVERLAP_THRESHOLD:
            continue
        dist = _vertical_distance_above(figure_bbox, b["bbox"])
        if dist < 0:
            continue
        score = _caption_score(b, dist, is_below=False)
        candidates.append((score, b))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    best_block = candidates[0][1]
    return _merge_consecutive_caption_blocks(best_block, blocks, figure_bbox, below=False)


def _merge_consecutive_caption_blocks(
    start_block: dict[str, Any],
    all_blocks: list[dict[str, Any]],
    figure_bbox: ImageBbox,
    *,
    below: bool,
) -> str:
    """Merge start_block with following blocks that are close vertically and aligned."""
    used = {id(start_block)}
    parts = [start_block["text"].strip()]
    bbox = start_block["bbox"]
    gap_key = 3 if below else 1  # index for bottom (below) or top (above)
    ref_y = bbox[gap_key]

    def vertical_dist(b: dict[str, Any]) -> float:
        by = b["bbox"][1] if below else b["bbox"][3]
        return abs(by - ref_y)

    sorted_blocks = sorted(all_blocks, key=vertical_dist)
    for b in sorted_blocks:
        if id(b) in used:
            continue
        ov = _horizontal_overlap(figure_bbox, b["bbox"])
        if ov < OVERLAP_THRESHOLD:
            continue
        if below:
            dist = _vertical_distance_below(figure_bbox, b["bbox"])
            if dist < 0:
                continue
        else:
            dist = _vertical_distance_above(figure_bbox, b["bbox"])
            if dist < 0:
                continue
        # Check gap from last added block
        other_ref = b["bbox"][gap_key]
        if abs(other_ref - ref_y) > CAPTION_LINE_GAP_PX:
            continue
        used.add(id(b))
        parts.append(b["text"].strip())
        ref_y = other_ref

    return " ".join(parts).strip() or parts[0]


def select_caption_for_bbox(
    figure_bbox: ImageBbox,
    blocks: list[dict[str, Any]],
    *,
    prefer_below: bool = True,
) -> Optional[str]:
    """
    Select best caption from a list of text blocks (same format as _collect_blocks).
    Used for testing without a real PDF page.
    """
    if not blocks:
        return None
    if prefer_below:
        caption = _best_caption_below(figure_bbox, blocks)
        if caption is None:
            caption = _best_caption_above(figure_bbox, blocks)
    else:
        caption = _best_caption_above(figure_bbox, blocks)
        if caption is None:
            caption = _best_caption_below(figure_bbox, blocks)
    return caption


def attach_captions(
    pdf_path: str | Path,
    figures: list[ExtractedFigure],
) -> list[tuple[ExtractedFigure, Optional[str]]]:
    """
    For each figure, open the PDF page and compute the caption.
    Returns list of (figure, caption_or_none).
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required.")

    from pathlib import Path
    doc = fitz.open(Path(pdf_path))
    result: list[tuple[ExtractedFigure, Optional[str]]] = []
    try:
        for fig in figures:
            page = doc[fig.page - 1]
            caption = get_caption_for_figure(page, fig.image_bbox, prefer_below=True)
            result.append((fig, caption))
    finally:
        doc.close()
    return result
