# src/retrieval/retriever.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

EMBS_NPY = INDEX_DIR / "chunk_embs.npy"
ROWIDS_JSON = INDEX_DIR / "chunk_row_ids.json"

META_PATH = INDEX_DIR / "chunk_meta.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks_with_neighbors.jsonl"
KG_GRAPHML = PROCESSED_DIR / "kg" / "graph_with_relations.graphml"


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_id: str
    section: str
    score: float
    text: str
    prev_chunk_id: Optional[int]
    next_chunk_id: Optional[int]


@dataclass
class ClaimNode:
    claim_id: str
    doc_id: str
    chunk_id: int
    section: str
    claim: str
    confidence: float


class TorchEmbedder:
    """
    Mean-pooling transformer embedder. Outputs L2-normalized vectors.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None, max_length: int = 512):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_one(self, text: str) -> np.ndarray:
        inputs = self.tok([text], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        h = self.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().numpy().astype("float32")[0]


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_meta_by_row(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for rec in _iter_jsonl(path):
        out[int(rec["row_id"])] = rec
    return out


def _load_chunks_by_id(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for rec in _iter_jsonl(path):
        out[int(rec["chunk_id"])] = rec
    return out


class AlignmentAtlasRetriever:
    """
    Stage 8: retrieval core
    - vector search over chunk embeddings (FAISS)
    - expand neighbor chunks
    - attach claim nodes and contradiction/entailment edges from KG
    """

    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        if not META_PATH.exists():
            raise FileNotFoundError(f"Missing {META_PATH}. Run Stage 3 (03_embed_chunks.py) or regenerate meta.")
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(f"Missing {CHUNKS_PATH}. Run Stage 2 (+neighbors).")
        if not KG_GRAPHML.exists():
            raise FileNotFoundError(f"Missing {KG_GRAPHML}. Run Stage 7 (merge relations into KG).")
        if not EMBS_NPY.exists():
            raise FileNotFoundError(f"Missing {EMBS_NPY}. Run Stage 3b (03b_export_chunk_embs.py).")
        if not ROWIDS_JSON.exists():
            raise FileNotFoundError(f"Missing {ROWIDS_JSON}. Run Stage 3b (03b_export_chunk_embs.py).")

        self.embedder = TorchEmbedder(embed_model, device=device)
        self.meta_by_row = _load_meta_by_row(META_PATH)
        self.chunks_by_id = _load_chunks_by_id(CHUNKS_PATH)
        self.chunk_embs = np.load(str(EMBS_NPY)).astype("float32")
        self.row_ids = json.loads(ROWIDS_JSON.read_text(encoding="utf-8"))

        G = nx.read_graphml(KG_GRAPHML)
        self.kg = nx.MultiDiGraph(G) if not isinstance(G, nx.MultiDiGraph) else G

        # Precompute chunk_id -> claim node ids
        self.chunk_to_claims: Dict[int, List[str]] = {}
        for nid, attrs in self.kg.nodes(data=True):
            if attrs.get("type") == "claim":
                try:
                    cid = int(attrs.get("chunk_id"))
                except Exception:
                    continue
                self.chunk_to_claims.setdefault(cid, []).append(nid)

    def runtime_info(self) -> Dict[str, Any]:
        return {
            "ready": True,
            "embed_model": self.embedder.model_name,
            "embed_device": self.embedder.device,
            "retrieval_backend": "numpy_dot",
            "num_chunks": int(self.chunk_embs.shape[0]),
            "embedding_dim": int(self.chunk_embs.shape[1]) if self.chunk_embs.ndim == 2 else None,
            "num_claim_nodes": int(len(self.chunk_to_claims)),
            "kg_nodes": int(self.kg.number_of_nodes()),
            "kg_edges": int(self.kg.number_of_edges()),
        }

    def vector_retrieve(self, question: str, top_k: int = 8) -> List[Tuple[int, float]]:
        q = self.embedder.encode_one(question)  # shape [D], L2-normalized

        # NumPy retrieval (no FAISS)
        scores = self.chunk_embs @ q  # [N]
        k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        out: List[Tuple[int, float]] = []
        for pos in idx:
            row_id = int(self.row_ids[pos])
            meta = self.meta_by_row[row_id]
            out.append((int(meta["chunk_id"]), float(scores[pos])))
        return out


    def expand_neighbors(self, chunk_ids: List[int], radius: int = 1) -> List[int]:
        seen = set()
        out: List[int] = []

        def add(cid: Optional[int]):
            if cid is None:
                return
            if cid in seen:
                return
            if cid not in self.chunks_by_id:
                return
            seen.add(cid)
            out.append(cid)

        for cid in chunk_ids:
            add(cid)
            cur = cid
            for _ in range(radius):
                prev_id = self.chunks_by_id[cur].get("prev_chunk_id")
                next_id = self.chunks_by_id[cur].get("next_chunk_id")
                add(int(prev_id) if prev_id is not None else None)
                add(int(next_id) if next_id is not None else None)

        return out

    def get_chunks(self, chunk_ids_with_scores: List[Tuple[int, float]]) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        for cid, score in chunk_ids_with_scores:
            rec = self.chunks_by_id[cid]
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=str(rec["doc_id"]),
                    section=str(rec.get("section", "unknown")),
                    score=score,
                    text=str(rec["text"]),
                    prev_chunk_id=rec.get("prev_chunk_id"),
                    next_chunk_id=rec.get("next_chunk_id"),
                )
            )
        return out

    def claims_for_chunks(self, chunk_ids: List[int]) -> List[ClaimNode]:
        claims: List[ClaimNode] = []
        for cid in chunk_ids:
            for claim_id in self.chunk_to_claims.get(cid, []):
                attrs = self.kg.nodes[claim_id]
                claims.append(
                    ClaimNode(
                        claim_id=claim_id,
                        doc_id=str(attrs.get("doc_id", "")),
                        chunk_id=int(attrs.get("chunk_id", cid)),
                        section=str(attrs.get("section", "unknown")),
                        claim=str(attrs.get("claim", "")),
                        confidence=float(attrs.get("confidence", 0.0)),
                    )
                )
        # sort by confidence desc
        claims.sort(key=lambda c: c.confidence, reverse=True)
        return claims

    def expand_claim_relations(
        self,
        claim_ids: List[str],
        rel_types: Tuple[str, ...] = ("contradiction", "entails"),
        max_per_claim: int = 5,
        min_confidence: float = 0.70,
    ) -> Dict[str, List[Tuple[str, str, float, str]]]:
        """
        Returns:
          {"contradiction": [(src,dst,conf,just), ...], "entails": [...]}
        """
        out = {r: [] for r in rel_types}
        for src in claim_ids:
            if not self.kg.has_node(src):
                continue
            count_by_rel = {r: 0 for r in rel_types}
            for _, dst, attrs in self.kg.out_edges(src, data=True):
                rel = attrs.get("rel")
                if rel not in rel_types:
                    continue
                conf = float(attrs.get("confidence", 0.0))
                if conf < min_confidence:
                    continue
                if count_by_rel[rel] >= max_per_claim:
                    continue
                out[rel].append(
                    (src, dst, conf, str(attrs.get("short_justification", "")))
                )
                count_by_rel[rel] += 1
        return out

    def build_evidence_pack(
        self,
        question: str,
        top_k_chunks: int = 12,
        neighbor_radius: int = 2,
        max_claim_rel_per_claim: int = 4,
    ) -> Dict[str, Any]:
        base = self.vector_retrieve(question, top_k=top_k_chunks)
        base_chunk_ids = [cid for cid, _ in base]

        expanded_ids = self.expand_neighbors(base_chunk_ids, radius=neighbor_radius)

        score_map = {cid: sc for cid, sc in base}
        scored: List[Tuple[int, float]] = []
        for cid in expanded_ids:
            sc = score_map.get(cid, (min(score_map.values()) - 0.05) if score_map else 0.0)
            scored.append((cid, sc))
        scored.sort(key=lambda x: x[1], reverse=True)

        chunks = self.get_chunks(scored)
        claims = self.claims_for_chunks([c.chunk_id for c in chunks])
        rels = self.expand_claim_relations([c.claim_id for c in claims], max_per_claim=max_claim_rel_per_claim)

        return {
            "question": question,
            "chunks": [c.__dict__ for c in chunks],
            "claims": [c.__dict__ for c in claims],
            "relations": {
                k: [
                    {"src": s, "dst": d, "confidence": conf, "justification": just}
                    for (s, d, conf, just) in v
                ]
                for k, v in rels.items()
            },
        }
