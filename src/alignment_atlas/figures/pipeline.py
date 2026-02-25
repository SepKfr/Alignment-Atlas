"""Orchestrate figure extraction, captioning, and vision API explanation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.alignment_atlas.figures.captioner import attach_captions
from src.alignment_atlas.figures.extractor import extract_figures
from src.alignment_atlas.figures.schemas import ExtractedFigure, ImageBbox
from src.alignment_atlas.figures.utils import ensure_dir
from src.alignment_atlas.figures.vision_api import VisionAPIClient

logger = logging.getLogger(__name__)

FIGURE_RECORDS_FILENAME = "figure_records.jsonl"
VISION_LOGS_DIRNAME = "vision_logs"


def _write_vision_log(
    log_dir: Path,
    figure_id: str,
    caption: Optional[str],
    result: Any,
) -> None:
    """Write one JSON file per figure with caption, raw response, and structured output."""
    from src.alignment_atlas.figures.schemas import VisionResult

    entry = {
        "figure_id": figure_id,
        "caption": caption,
        "prompt_preview": f"Caption: {caption or 'None'}\nTask: Explain figure and return JSON (explanation_md, facts_json).",
        "raw_response": result.raw_text if isinstance(result, VisionResult) else None,
        "structured_json": result.structured_json if isinstance(result, VisionResult) else None,
        "model": result.model if isinstance(result, VisionResult) else None,
        "usage": result.usage if isinstance(result, VisionResult) else None,
        "error": None if isinstance(result, VisionResult) else (str(result) if result is not None else "No result (exception or skipped)"),
    }
    path = log_dir / f"{figure_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    logger.debug("Wrote vision log %s", path)


def process_pdf_figures(
    pdf_path: str | Path,
    out_dir: str | Path,
    paper_id: str,
    *,
    min_side_px: int = 200,
    max_figures: Optional[int] = None,
    force_refresh: bool = False,
    cache_dir: Optional[str] | Path = None,
    save_vision_logs: bool = False,
) -> list[dict[str, Any]]:
    """
    Extract figures from PDF, attach captions, call vision API, write JSONL.
    Returns list of figure records (same as written to JSONL).
    """
    pdf_path = Path(pdf_path)
    out_dir = ensure_dir(out_dir)
    if cache_dir is None:
        cache_dir = out_dir / "vision_cache"
    else:
        cache_dir = Path(cache_dir)
    ensure_dir(cache_dir)
    vision_log_dir = (out_dir / VISION_LOGS_DIRNAME) if save_vision_logs else None
    if vision_log_dir:
        ensure_dir(vision_log_dir)

    # 1) Extract
    figures = extract_figures(
        pdf_path, out_dir, min_side_px=min_side_px, min_figure_pt=80.0
    )
    if max_figures is not None and max_figures >= 0:
        figures = figures[:max_figures]

    # 2) Captions
    captioned = attach_captions(pdf_path, figures)
    pdf_path_str = str(pdf_path.resolve())

    # 3) Vision API
    client = VisionAPIClient(cache_dir=cache_dir)
    records: list[dict[str, Any]] = []
    for fig, caption in captioned:
        try:
            result = client.explain(
                fig.image_path,
                caption,
                use_cache=True,
                force_refresh=force_refresh,
            )
        except Exception as e:
            logger.exception("Vision API failed for %s: %s", fig.figure_id, e)
            result = None

        explanation = ""
        structured_json: Optional[dict[str, Any]] = None
        api_model = ""
        if result:
            explanation = result.raw_text
            structured_json = result.structured_json
            api_model = result.model
            if structured_json:
                expl_md = structured_json.get("explanation_md")
                if isinstance(expl_md, str):
                    explanation = expl_md

        if save_vision_logs and vision_log_dir is not None:
            _write_vision_log(vision_log_dir, fig.figure_id, caption, result)

        record = {
            "paper_id": paper_id,
            "pdf_path": pdf_path_str,
            "page": fig.page,
            "figure_id": fig.figure_id,
            "image_path": fig.image_path,
            "image_bbox": fig.image_bbox.to_list(),
            "caption": caption,
            "api_model": api_model,
            "explanation": explanation,
            "structured_json": structured_json,
        }
        records.append(record)

    # 4) Write JSONL
    jsonl_path = out_dir / FIGURE_RECORDS_FILENAME
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %s records to %s", len(records), jsonl_path)
    return records
