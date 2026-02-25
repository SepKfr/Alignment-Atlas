"""Tests for vision API caching and JSONL schema validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.alignment_atlas.figures.schemas import (
    figure_record_schema,
    validate_figure_record,
)
from src.alignment_atlas.figures.utils import sha256_bytes
from src.alignment_atlas.figures.vision_api import (
    VisionAPIClient,
    make_cache_key,
    parse_structured_output,
)


def test_make_cache_key_deterministic():
    """Same inputs produce same cache key."""
    img = b"fake image bytes"
    cap = "Figure 1"
    model = "gpt-4o"
    ver = "v1"
    k1 = make_cache_key(img, cap, model, ver)
    k2 = make_cache_key(img, cap, model, ver)
    assert k1 == k2


def test_make_cache_key_different_inputs_different_keys():
    """Different caption or image gives different key."""
    img = b"fake image"
    k1 = make_cache_key(img, "Cap A", "gpt-4o", "v1")
    k2 = make_cache_key(img, "Cap B", "gpt-4o", "v1")
    k3 = make_cache_key(b"other image", "Cap A", "gpt-4o", "v1")
    assert k1 != k2
    assert k1 != k3


def test_sha256_bytes():
    """Consistent with utils.sha256_bytes."""
    from src.alignment_atlas.figures.utils import sha256_bytes as util_sha
    data = b"hello"
    assert len(util_sha(data)) == 64
    assert util_sha(data) == util_sha(data)


def test_vision_cache_save_and_load(tmp_path: Path):
    """Cache write and read round-trip."""
    cache_dir = tmp_path / "vision_cache"
    cache_dir.mkdir()
    client = VisionAPIClient(api_key="test-key", cache_dir=cache_dir)
    key = "a" * 64
    from src.alignment_atlas.figures.schemas import VisionResult
    result = VisionResult(
        raw_text="Some explanation",
        structured_json={"explanation_md": "md", "facts_json": {}},
        model="gpt-4o",
        usage={"total_tokens": 100},
    )
    client._save_cached(key, result)
    loaded = client._load_cached(key)
    assert loaded is not None
    assert loaded.raw_text == result.raw_text
    assert loaded.model == result.model
    assert loaded.structured_json == result.structured_json


def test_vision_cache_miss_returns_none(tmp_path: Path):
    """Missing cache file returns None."""
    client = VisionAPIClient(cache_dir=tmp_path)
    assert client._load_cached("nonexistent_key") is None


# --- JSONL schema validation ---


def test_validate_figure_record_valid():
    """Valid record returns no errors."""
    record = {
        "paper_id": "paper_1",
        "pdf_path": "/path/to/file.pdf",
        "page": 1,
        "figure_id": "p1_fig0",
        "image_path": "/out/p1_fig0.png",
        "image_bbox": [10.0, 20.0, 100.0, 200.0],
        "caption": "Figure 1: Results.",
        "api_model": "gpt-4o-mini",
        "explanation": "The figure shows...",
        "structured_json": {"explanation_md": "...", "facts_json": {}},
    }
    errors = validate_figure_record(record)
    assert errors == []


def test_validate_figure_record_missing_key():
    """Missing required key returns error."""
    record = {
        "paper_id": "paper_1",
        "pdf_path": "/path/to/file.pdf",
        "page": 1,
        "figure_id": "p1_fig0",
        "image_path": "/out/p1_fig0.png",
        "image_bbox": [10, 20, 100, 200],
        "caption": None,
        "api_model": "gpt-4o",
        "explanation": "",
        # missing structured_json
    }
    errors = validate_figure_record(record)
    assert any("structured_json" in e or "Missing key" in e for e in errors)


def test_validate_figure_record_invalid_bbox():
    """Invalid image_bbox (wrong length or type) returns error."""
    record = {
        "paper_id": "p",
        "pdf_path": "/p.pdf",
        "page": 1,
        "figure_id": "p1_fig0",
        "image_path": "/p.png",
        "image_bbox": [10, 20],  # should be 4 numbers
        "caption": None,
        "api_model": "m",
        "explanation": "",
        "structured_json": None,
    }
    errors = validate_figure_record(record)
    assert any("image_bbox" in e for e in errors)


def test_figure_record_schema_keys():
    """Schema contains all expected keys."""
    schema = figure_record_schema()
    expected = {
        "paper_id", "pdf_path", "page", "figure_id", "image_path",
        "image_bbox", "caption", "api_model", "explanation", "structured_json",
    }
    assert set(schema.keys()) == expected


def test_parse_structured_output_raw_json():
    """parse_structured_output parses raw JSON string."""
    text = '{"explanation_md": "Hi", "facts_json": {"figure_type": "chart"}}'
    out = parse_structured_output(text)
    assert out is not None
    assert out.get("explanation_md") == "Hi"
    assert out.get("facts_json", {}).get("figure_type") == "chart"


def test_parse_structured_output_markdown_code_block():
    """parse_structured_output extracts JSON from ```json ... ```."""
    text = 'Some text\n```json\n{"explanation_md": "x", "facts_json": {}}\n```'
    out = parse_structured_output(text)
    assert out is not None
    assert out.get("explanation_md") == "x"


def test_parse_structured_output_invalid_returns_none():
    """Invalid JSON returns None."""
    assert parse_structured_output("not json at all") is None
    assert parse_structured_output("") is None
