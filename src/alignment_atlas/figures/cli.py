"""CLI for the figures pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.alignment_atlas.figures.pipeline import (
    FIGURE_RECORDS_FILENAME,
    process_pdf_figures,
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="figures_pipeline",
        description="Extract figures from PDFs, attach captions, and generate explanations via vision API.",
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument("--out", required=True, help="Output directory for images and JSONL.")
    parser.add_argument("--paper_id", required=True, help="Paper identifier for records.")
    parser.add_argument(
        "--min_side_px",
        type=int,
        default=200,
        help="Minimum side length (px) for embedded images (default: 200).",
    )
    parser.add_argument(
        "--max_figures",
        type=int,
        default=None,
        help="Maximum number of figures to process (default: no limit).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-call vision API for all figures.",
    )
    parser.add_argument(
        "--save-vision-logs",
        action="store_true",
        help="Save per-figure vision request/response logs to out_dir/vision_logs/.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        records = process_pdf_figures(
            pdf_path,
            out_dir,
            args.paper_id,
            min_side_px=args.min_side_px,
            max_figures=args.max_figures,
            force_refresh=args.force,
            save_vision_logs=args.save_vision_logs,
        )
    except Exception as e:
        logger.exception("Pipeline failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1

    num_extracted = len(records)
    num_captioned = sum(1 for r in records if r.get("caption"))
    num_api_called = sum(1 for r in records if r.get("api_model"))

    jsonl_path = out_dir / FIGURE_RECORDS_FILENAME
    print("Summary:")
    print(f"  Figures extracted: {num_extracted}")
    print(f"  With caption:      {num_captioned}")
    print(f"  API explanations:  {num_api_called}")
    print(f"  Output:            {jsonl_path}")
    if args.save_vision_logs:
        print(f"  Vision logs:       {out_dir / 'vision_logs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
