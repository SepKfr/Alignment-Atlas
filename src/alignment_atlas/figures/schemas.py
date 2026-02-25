"""Data schemas for the figures pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ImageBbox:
    """Bounding box (x0, y0, x1, y1) in PDF page coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class ExtractedFigure:
    """A figure extracted from a PDF page."""

    doc_path: str
    page: int  # 1-based page number
    figure_id: str  # e.g. "p3_fig0"
    image_path: str
    image_bbox: ImageBbox
    source: str = "embedded"  # "embedded" | "rendered_page"


@dataclass
class VisionResult:
    """Result from the vision API for a single figure."""

    raw_text: str
    structured_json: dict[str, Any] | None
    model: str
    usage: dict[str, Any] | None = None


def figure_record_schema() -> dict[str, type]:
    """Expected types for a figure record written to JSONL."""
    return {
        "paper_id": str,
        "pdf_path": str,
        "page": int,
        "figure_id": str,
        "image_path": str,
        "image_bbox": list,  # [x0, y0, x1, y1]
        "caption": (str, type(None)),
        "api_model": str,
        "explanation": str,
        "structured_json": (dict, type(None)),
    }


def validate_figure_record(record: dict[str, Any]) -> list[str]:
    """
    Validate a figure record against the expected schema.
    Returns a list of error messages; empty list means valid.
    """
    errors: list[str] = []
    schema = figure_record_schema()
    for key, expected_type in schema.items():
        if key not in record:
            errors.append(f"Missing key: {key}")
            continue
        val = record[key]
        if isinstance(expected_type, tuple):
            if type(val) not in expected_type:
                errors.append(
                    f"Key '{key}': expected one of {expected_type}, got {type(val).__name__}"
                )
        elif key == "image_bbox":
            if not isinstance(val, list) or len(val) != 4:
                errors.append(
                    f"Key 'image_bbox': expected list of 4 numbers, got {type(val).__name__}"
                )
            elif not all(isinstance(x, (int, float)) for x in val):
                errors.append("Key 'image_bbox': all elements must be numbers")
        elif not isinstance(val, expected_type):
            errors.append(
                f"Key '{key}': expected {expected_type.__name__}, got {type(val).__name__}"
            )
    return errors
