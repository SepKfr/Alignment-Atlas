"""Tests for caption matching heuristic using synthetic bbox + text blocks."""

from __future__ import annotations

import pytest

from src.alignment_atlas.figures.captioner import (
    CAPTION_LINE_GAP_PX,
    OVERLAP_THRESHOLD,
    select_caption_for_bbox,
)
from src.alignment_atlas.figures.schemas import ImageBbox


def _block(x0: float, y0: float, x1: float, y1: float, text: str) -> dict:
    return {"bbox": (x0, y0, x1, y1), "text": text}


def test_caption_below_figure_preferred():
    """Block below figure with horizontal overlap and 'Figure 1' is selected."""
    # Figure in middle of page
    fig_bbox = ImageBbox(100, 200, 400, 450)
    blocks = [
        _block(50, 50, 450, 80, "Some title"),
        _block(100, 460, 400, 480, "Figure 1: Training loss over time."),
        _block(100, 490, 400, 510, "The curve decreases monotonically."),
    ]
    caption = select_caption_for_bbox(fig_bbox, blocks, prefer_below=True)
    assert caption is not None
    assert "Figure 1" in caption
    assert "Training loss" in caption


def test_caption_above_fallback_when_nothing_below():
    """When no block below has overlap, caption above is used."""
    fig_bbox = ImageBbox(100, 300, 400, 500)
    blocks = [
        _block(100, 100, 400, 130, "Table 2: Hyperparameters."),
        _block(50, 520, 100, 540, "Unrelated note"),  # no horizontal overlap
    ]
    caption = select_caption_for_bbox(fig_bbox, blocks, prefer_below=True)
    assert caption is not None
    assert "Table 2" in caption


def test_no_caption_when_no_overlap():
    """Text with no horizontal overlap with figure is ignored."""
    fig_bbox = ImageBbox(100, 200, 400, 400)
    blocks = [
        _block(10, 420, 90, 440, "Figure 1"),  # entirely left of figure, zero overlap
    ]
    caption = select_caption_for_bbox(fig_bbox, blocks, prefer_below=True)
    assert caption is None


def test_caption_like_start_boost():
    """Block starting with 'Fig. 2' is preferred over generic text below."""
    fig_bbox = ImageBbox(100, 100, 400, 350)
    blocks = [
        _block(100, 360, 400, 380, "Some random text below."),
        _block(100, 355, 400, 375, "Fig. 2: Architecture diagram."),
    ]
    caption = select_caption_for_bbox(fig_bbox, blocks, prefer_below=True)
    assert caption is not None
    assert "Fig." in caption or "Architecture" in caption


def test_empty_blocks_returns_none():
    """No blocks returns None."""
    fig_bbox = ImageBbox(0, 0, 100, 100)
    assert select_caption_for_bbox(fig_bbox, []) is None


def test_constants():
    """Sanity check constants used in heuristic."""
    assert OVERLAP_THRESHOLD > 0 and OVERLAP_THRESHOLD <= 1
    assert CAPTION_LINE_GAP_PX > 0
