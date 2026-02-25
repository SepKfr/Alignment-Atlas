"""
Integration test: build_kg ingests figures.jsonl and links figure nodes to papers.
Runs build_kg against real data/processed (figures.jsonl built from vision_logs)
and checks that the graph contains figure nodes and paper->figure edges.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
KG_DIR = PROCESSED / "kg"
FIGURES_JSONL = PROCESSED / "figures.jsonl"
OUT_JSON = KG_DIR / "graph.json"
OUT_STATS = KG_DIR / "stats.json"


@pytest.mark.skipif(
    not FIGURES_JSONL.exists(),
    reason="data/processed/figures.jsonl missing; run scripts/build_figures_jsonl_from_logs.py first",
)
def test_build_kg_ingests_figures_and_links_to_paper():
    """Run build_kg and assert graph has figure nodes and has_figure edges."""
    from src.ingest import stages

    stage = stages.BuildKGStage()
    result = stage.run(
        progress_callback=None,
        stage_index=5,
        stage_total=10,
        stage_results=[],
        env_overrides=None,
        timeout_seconds=120,
    )
    assert result.ok, result.output_tail or "build_kg failed"

    with open(OUT_STATS, encoding="utf-8") as f:
        stats = json.load(f)
    assert "figures" in stats, "stats should include figures count"
    assert stats["figures"] >= 1, "at least one figure should be ingested"

    with open(OUT_JSON, encoding="utf-8") as f:
        graph_data = json.load(f)
    nodes = {n["id"]: n for n in graph_data["nodes"]}
    edges = graph_data["edges"]

    figure_nodes = [n for n in graph_data["nodes"] if n.get("type") == "figure"]
    assert len(figure_nodes) >= 1, "graph should contain figure nodes"
    has_figure_edges = [e for e in edges if e.get("rel") == "has_figure"]
    assert len(has_figure_edges) >= 1, "graph should contain paper->figure edges"

    # One figure should link to paper perez_2022_model_written_evals
    paper_id = "perez_2022_model_written_evals"
    paper_node = f"paper:{paper_id}"
    assert any(
        e["source"] == paper_node and e["target"].startswith("figure:")
        for e in has_figure_edges
    ), f"expected at least one has_figure edge from {paper_node}"
