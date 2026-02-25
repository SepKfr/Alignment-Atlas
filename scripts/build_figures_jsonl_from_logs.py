#!/usr/bin/env python3
"""
One-off: build data/processed/figures.jsonl from data/figures_out/perez_2022/vision_logs
so we can test KG ingestion without re-running the figures pipeline.
Uses paper_id = perez_2022_model_written_evals to match docs/claims.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES_OUT = ROOT / "data" / "figures_out" / "perez_2022"
VISION_LOGS = FIGURES_OUT / "vision_logs"
PROCESSED = ROOT / "data" / "processed"
PDF_PATH = ROOT / "data" / "raw_pdfs" / "perez_2022_model_written_evals.pdf"
PAPER_ID = "perez_2022_model_written_evals"


def page_from_figure_id(figure_id: str) -> int:
    m = re.match(r"p(\d+)_fig\d+", figure_id)
    return int(m.group(1)) if m else 0


def main() -> None:
    out_path = PROCESSED / "figures.jsonl"
    records = []
    for log_path in sorted(VISION_LOGS.glob("*.json")):
        with open(log_path, encoding="utf-8") as f:
            data = json.load(f)
        figure_id = data.get("figure_id") or log_path.stem
        caption = data.get("caption")
        model = data.get("model", "")
        raw = data.get("raw_response") or ""
        structured = data.get("structured_json")
        if isinstance(structured, dict) and "explanation_md" in structured:
            explanation = structured.get("explanation_md") or raw
        else:
            explanation = raw
        page = page_from_figure_id(figure_id)
        image_path = str(FIGURES_OUT / f"{figure_id}.png")
        record = {
            "paper_id": PAPER_ID,
            "pdf_path": str(PDF_PATH.resolve()),
            "page": page,
            "figure_id": figure_id,
            "image_path": image_path,
            "image_bbox": [0.0, 0.0, 100.0, 100.0],
            "caption": caption,
            "api_model": model,
            "explanation": explanation,
            "structured_json": structured,
        }
        records.append(record)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} figure records -> {out_path}")


if __name__ == "__main__":
    main()
