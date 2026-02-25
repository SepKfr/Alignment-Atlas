"""Figures pipeline: extract images from PDFs, attach captions, explain via vision API."""

from src.alignment_atlas.figures.captioner import attach_captions, get_caption_for_figure
from src.alignment_atlas.figures.cli import main as cli_main
from src.alignment_atlas.figures.extractor import extract_figures
from src.alignment_atlas.figures.pipeline import process_pdf_figures
from src.alignment_atlas.figures.schemas import (
    ExtractedFigure,
    ImageBbox,
    VisionResult,
    validate_figure_record,
)
from src.alignment_atlas.figures.vision_api import VisionAPIClient, make_cache_key

__all__ = [
    "attach_captions",
    "get_caption_for_figure",
    "cli_main",
    "extract_figures",
    "process_pdf_figures",
    "ExtractedFigure",
    "ImageBbox",
    "VisionResult",
    "validate_figure_record",
    "VisionAPIClient",
    "make_cache_key",
]
