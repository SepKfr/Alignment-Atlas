# src/ingest/05_build_kg.py
"""
Stage 5 — Build a Knowledge Graph from extracted claims

Reads:
- data/processed/docs.jsonl       (paper metadata, optional but recommended)
- data/processed/claims.jsonl     (one line per claim)
- data/processed/figures.jsonl    (optional; figure records from figures pipeline)

Writes:
- data/processed/kg/graph.graphml   (Gephi-friendly)
- data/processed/kg/graph.json      (app-friendly)
- data/processed/kg/stats.json      (debug)

Graph schema (NetworkX):
Nodes:
- paper:{doc_id}   type="paper", title, year
- claim:{doc_id}:{chunk_id}:{i}  type="claim", claim, section, confidence, evidence_span, chunk_id
- figure:{paper_id}:{figure_id}  type="figure", paper_id, figure_id, page, caption, explanation, image_path, ...
- tag:{tag}        type="tag", tag

Edges:
- paper -> claim   rel="has_claim"
- paper -> figure  rel="has_figure"   (when figures.jsonl is present)
- claim -> tag     rel="has_tag"
- claim <-> claim  rel="related_by_tag" (optional, lightweight; capped)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DOCS_JSONL = PROCESSED_DIR / "docs.jsonl"
CLAIMS_JSONL = PROCESSED_DIR / "claims.jsonl"
FIGURES_JSONL = PROCESSED_DIR / "figures.jsonl"

KG_DIR = PROCESSED_DIR / "kg"
OUT_GRAPHML = KG_DIR / "graph.graphml"
OUT_JSON = KG_DIR / "graph.json"
OUT_STATS = KG_DIR / "stats.json"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_docs() -> Dict[str, Dict[str, Any]]:
    """
    docs.jsonl is optional, but helpful for paper title/year.
    Returns doc_id -> metadata dict.
    """
    if not DOCS_JSONL.exists():
        return {}
    docs = {}
    for d in iter_jsonl(DOCS_JSONL):
        doc_id = d.get("doc_id")
        if not doc_id:
            continue
        docs[str(doc_id)] = d
    return docs


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--add-related-by-tag",
        action="store_true",
        default=True,
        help="Add claim-claim links if they share a tag (capped per tag).",
    )
    parser.add_argument(
        "--no-related-by-tag",
        dest="add_related_by_tag",
        action="store_false",
    )
    parser.add_argument(
        "--max-claims-per-tag",
        type=int,
        default=80,
        help="Cap: for each tag, only connect up to this many claims (avoids dense blowup).",
    )
    args = parser.parse_args()

    KG_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_docs()

    G = nx.MultiDiGraph()

    # Track for optional related_by_tag edges
    tag_to_claims: Dict[str, List[str]] = defaultdict(list)

    n_claims = 0
    n_papers = 0
    n_tags = 0

    # 1) Add paper + claim + tag nodes, plus edges paper->claim and claim->tag
    for c in iter_jsonl(CLAIMS_JSONL):
        claim_id = safe_str(c.get("claim_id"))
        doc_id = safe_str(c.get("doc_id"))
        chunk_id = c.get("chunk_id")
        section = safe_str(c.get("section"))
        claim_text = safe_str(c.get("claim"))
        claim_type = safe_str(c.get("type"))
        evidence_span = safe_str(c.get("evidence_span"))
        confidence = float(c.get("confidence", 0.0))
        tags = c.get("tags", []) or []

        if not claim_id or not doc_id or chunk_id is None or not claim_text:
            # skip malformed claim lines
            continue

        paper_node = f"paper:{doc_id}"
        claim_node = claim_id  # already unique, keep it as node id

        # Paper node
        if not G.has_node(paper_node):
            md = docs.get(doc_id, {})
            G.add_node(
                paper_node,
                type="paper",
                doc_id=doc_id,
                title=safe_str(md.get("title")) or doc_id,
                year=md.get("year"),
                source_type=md.get("source_type"),
                source_url=md.get("source_url"),
            )
            n_papers += 1

        # Claim node
        if not G.has_node(claim_node):
            G.add_node(
                claim_node,
                type="claim",
                claim_id=claim_id,
                doc_id=doc_id,
                chunk_id=int(chunk_id),
                section=section,
                claim_type=claim_type,
                claim=claim_text,
                evidence_span=evidence_span,
                confidence=confidence,
            )
            n_claims += 1

        # Edge: paper -> claim
        G.add_edge(paper_node, claim_node, rel="has_claim")

        # Tag nodes + edges: claim -> tag
        for t in tags:
            tag = safe_str(t).strip()
            if not tag:
                continue
            tag_node = f"tag:{tag}"
            if not G.has_node(tag_node):
                G.add_node(tag_node, type="tag", tag=tag)
                n_tags += 1
            G.add_edge(claim_node, tag_node, rel="has_tag")
            tag_to_claims[tag].append(claim_node)

    # 2) Optional: add lightweight "related_by_tag" claim-claim edges
    related_edges = 0
    if args.add_related_by_tag:
        for tag, claim_nodes in tag_to_claims.items():
            # cap to avoid quadratic explosion for very common tags
            claim_nodes = list(dict.fromkeys(claim_nodes))[: args.max_claims_per_tag]
            if len(claim_nodes) < 2:
                continue

            # connect all pairs (undirected-ish via two directed edges) — still can be big, but capped
            for i in range(len(claim_nodes)):
                a = claim_nodes[i]
                for j in range(i + 1, len(claim_nodes)):
                    b = claim_nodes[j]
                    G.add_edge(a, b, rel="related_by_tag", tag=tag)
                    G.add_edge(b, a, rel="related_by_tag", tag=tag)
                    related_edges += 2

    # 2b) Optional: add figure nodes and paper->figure edges from figures.jsonl
    n_figures = 0
    if FIGURES_JSONL.exists():
        for rec in iter_jsonl(FIGURES_JSONL):
            paper_id = safe_str(rec.get("paper_id"))
            figure_id = safe_str(rec.get("figure_id"))
            if not paper_id or not figure_id:
                continue
            node_id = f"figure:{paper_id}:{figure_id}"
            if G.has_node(node_id):
                continue
            paper_node = f"paper:{paper_id}"
            if not G.has_node(paper_node):
                md = docs.get(paper_id, {})
                G.add_node(
                    paper_node,
                    type="paper",
                    doc_id=paper_id,
                    title=safe_str(md.get("title")) or paper_id,
                    year=md.get("year"),
                    source_type=md.get("source_type"),
                    source_url=md.get("source_url"),
                )
                n_papers += 1
            explanation = safe_str(rec.get("explanation"))
            structured = rec.get("structured_json")
            structured_str = json.dumps(structured, ensure_ascii=False) if isinstance(structured, dict) else safe_str(structured)
            G.add_node(
                node_id,
                type="figure",
                paper_id=paper_id,
                figure_id=figure_id,
                page=int(rec.get("page", 0)),
                caption=safe_str(rec.get("caption")),
                explanation=explanation,
                image_path=safe_str(rec.get("image_path")),
                api_model=safe_str(rec.get("api_model")),
                structured_json=structured_str,
            )
            G.add_edge(paper_node, node_id, rel="has_figure")
            n_figures += 1

    # 3) Save
    # GraphML is great for visualization tools, but can be picky with attribute types.
    # We'll ensure values are simple scalars/strings.
    nx.write_graphml(G, OUT_GRAPHML)

    # JSON export (nodes + edges)
    nodes_json = []
    for nid, attrs in G.nodes(data=True):
        nodes_json.append({"id": nid, **{k: (v if isinstance(v, (int, float)) else safe_str(v)) for k, v in attrs.items()}})

    edges_json = []
    for u, v, key, attrs in G.edges(keys=True, data=True):
        edges_json.append(
            {
                "source": u,
                "target": v,
                "key": safe_str(key),
                **{k: (a if isinstance(a, (int, float)) else safe_str(a)) for k, a in attrs.items()},
            }
        )

    OUT_JSON.write_text(json.dumps({"nodes": nodes_json, "edges": edges_json}, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "num_nodes": int(G.number_of_nodes()),
        "num_edges": int(G.number_of_edges()),
        "papers": int(n_papers),
        "claims": int(n_claims),
        "tags": int(n_tags),
        "figures": int(n_figures),
        "related_by_tag_edges": int(related_edges),
        "graphml_path": str(OUT_GRAPHML.as_posix()),
        "json_path": str(OUT_JSON.as_posix()),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote KG GraphML -> {OUT_GRAPHML}")
    print(f"Wrote KG JSON    -> {OUT_JSON}")
    print(f"Wrote KG stats   -> {OUT_STATS}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
