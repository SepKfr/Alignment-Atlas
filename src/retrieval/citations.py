# src/retrieval/citations.py
"""
Citation resolver: turns {"kind":"claim|chunk","id":...} into human-friendly evidence.

Loads (once per process):
- data/processed/claims.jsonl
- data/processed/chunks_with_neighbors.jsonl
- data/processed/docs.jsonl (from Stage 0; optional but recommended)

Primary use:
- Stage 10/11 UI: show paper title + section + snippet for each citation
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

CLAIMS_JSONL = PROCESSED_DIR / "claims.jsonl"
CHUNKS_JSONL = PROCESSED_DIR / "chunks_with_neighbors.jsonl"
DOCS_JSONL = PROCESSED_DIR / "docs.jsonl"

DEFAULT_SNIPPET_CHARS = 420


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _compact(s: str, max_chars: int = DEFAULT_SNIPPET_CHARS) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    # Some PDF extractions include leading page/line numbers like "84 We ...".
    s = re.sub(r"^\d{1,3}\s+(?=[A-Z])", "", s).strip()
    # Chunks can begin mid-sentence because of overlap windows.
    # When that happens, trim leading fragment so snippets start cleanly.
    if s and s[0].islower():
        m = re.search(r"(?<=[\.\?\!])\s+[\"“]?[A-Z0-9#]", s[:320])
        if m:
            s = s[m.end() - 1 :].lstrip()
    if len(s) <= max_chars:
        return s

    # Prefer ending at a sentence boundary so evidence does not look cut.
    boundaries = list(re.finditer(r"[\.\!\?](?:[\"”'\)\]]+)?(?=\s|$)", s))
    if boundaries:
        # Best case: last full sentence end within max_chars.
        within = [b.end() for b in boundaries if b.end() <= max_chars]
        if within:
            # Avoid overly tiny snippets; if too short, allow one sentence past max_chars.
            cut = within[-1]
            if cut >= int(max_chars * 0.45):
                return s[:cut].strip()

        # If none/too short, allow slight overshoot to finish a sentence.
        ahead = [b.end() for b in boundaries if max_chars < b.end() <= max_chars + 180]
        if ahead:
            return s[: ahead[0]].strip()

    # Fallback: hard trim but make it look like a finished statement.
    t = s[:max_chars].rstrip()
    if " " in t:
        t = t.rsplit(" ", 1)[0]
    if t and t[-1] not in ".!?":
        t += "."
    return t


def _sentence_bounds_around(text: str, idx: int) -> Tuple[int, int]:
    left = 0
    right = len(text)
    # Find prior sentence boundary.
    for m in re.finditer(r"[\.\!\?](?:[\"”'\)\]]+)?\s+", text):
        if m.end() <= idx:
            left = m.end()
        else:
            break
    # Find next sentence boundary.
    for m in re.finditer(r"[\.\!\?](?:[\"”'\)\]]+)?\s+", text[idx:]):
        right = idx + m.end()
        break
    return left, right


def _count_sentences(text: str) -> int:
    return len(re.findall(r"[\.\!\?](?:[\"”'\)\]]+)?(?=\s|$)", text or ""))


def _context_snippet_from_chunk(
    chunk_text: str,
    *,
    evidence_span: str,
    claim_text: str,
    max_chars: int,
    min_chars: int = 140,
) -> str:
    text = re.sub(r"\s+", " ", (chunk_text or "")).strip()
    if not text:
        return _compact(evidence_span or claim_text, max_chars=max_chars)

    anchor = (evidence_span or claim_text or "").strip()
    if not anchor:
        return _compact(text, max_chars=max_chars)

    # Locate anchor loosely in chunk (case-insensitive).
    pos = text.lower().find(anchor.lower())
    if pos < 0:
        return _compact(text, max_chars=max_chars)

    left, right = _sentence_bounds_around(text, pos)
    snippet = text[left:right].strip()

    # Expand by neighboring sentences while snippet is too short.
    while len(snippet) < min_chars:
        expanded = False
        if left > 0:
            # Expand one sentence to the left.
            new_left, _ = _sentence_bounds_around(text, max(0, left - 1))
            if new_left < left:
                left = new_left
                expanded = True
        if len(text[left:right].strip()) < min_chars and right < len(text):
            # Expand one sentence to the right.
            _tmp_left, new_right = _sentence_bounds_around(text, min(len(text) - 1, right))
            if new_right > right:
                right = new_right
                expanded = True
        snippet = text[left:right].strip()
        if not expanded:
            break

    # Even when long enough, one sentence can still feel context-poor.
    # Prefer at least two sentences when the chunk has enough context.
    chunk_sentence_count = _count_sentences(text)
    while chunk_sentence_count >= 2 and _count_sentences(snippet) < 2:
        expanded = False
        if right < len(text):
            _tmp_left, new_right = _sentence_bounds_around(text, min(len(text) - 1, right))
            if new_right > right:
                right = new_right
                expanded = True
        if not expanded and left > 0:
            new_left, _tmp_right = _sentence_bounds_around(text, max(0, left - 1))
            if new_left < left:
                left = new_left
                expanded = True
        snippet = text[left:right].strip()
        if not expanded:
            break

    if len(snippet) < min_chars:
        # Final fallback: centered window around anchor.
        center = pos + max(1, len(anchor) // 2)
        half = max_chars // 2
        ws = max(0, center - half)
        we = min(len(text), ws + max_chars)
        snippet = text[ws:we].strip()
    return _compact(snippet, max_chars=max_chars)


@dataclass
class CitationResolved:
    kind: str                 # "claim" | "chunk"
    id: str                   # claim_id or chunk_id-as-string
    doc_id: str
    title: str
    section: str
    snippet: str
    chunk_id: Optional[int] = None
    url: Optional[str] = None
    local_path: Optional[str] = None
    source_type: str = "atlas"


class CitationResolver:
    def __init__(
        self,
        claims_path: Path = CLAIMS_JSONL,
        chunks_path: Path = CHUNKS_JSONL,
        docs_path: Path = DOCS_JSONL,
    ):
        self.claims_path = claims_path
        self.chunks_path = chunks_path
        self.docs_path = docs_path

        self._claims_by_id: Dict[str, Dict[str, Any]] = {}
        self._chunks_by_id: Dict[int, Dict[str, Any]] = {}
        self._docs_by_id: Dict[str, Dict[str, Any]] = {}

        self._load()

    def _load(self) -> None:
        # claims
        if self.claims_path.exists():
            for r in _iter_jsonl(self.claims_path):
                cid = str(r.get("claim_id", ""))
                if cid:
                    self._claims_by_id[cid] = r

        # chunks
        if self.chunks_path.exists():
            for r in _iter_jsonl(self.chunks_path):
                try:
                    k = int(r["chunk_id"])
                except Exception:
                    continue
                self._chunks_by_id[k] = r

        # docs
        if self.docs_path.exists():
            for r in _iter_jsonl(self.docs_path):
                did = str(r.get("doc_id", ""))
                if did:
                    self._docs_by_id[did] = r

    def doc_title(self, doc_id: str) -> str:
        rec = self._docs_by_id.get(doc_id)
        if rec and rec.get("title"):
            return str(rec["title"])
        # fallback: prettify doc_id
        return doc_id.replace("_", " ").strip()

    def doc_url(self, doc_id: str) -> Optional[str]:
        rec = self._docs_by_id.get(doc_id)
        if not rec:
            return None
        for key in ("url", "source_url", "pdf_url", "link"):
            if rec.get(key):
                return str(rec[key])
        return None

    def doc_path(self, doc_id: str) -> Optional[str]:
        rec = self._docs_by_id.get(doc_id)
        if not rec:
            return None
        for key in ("path", "local_path", "pdf_path", "filepath"):
            if rec.get(key):
                return str(rec[key])
        return None

    def resolve_one(self, cit: Dict[str, str], *, snippet_chars: int = DEFAULT_SNIPPET_CHARS) -> Optional[CitationResolved]:
        kind = str(cit.get("kind", "")).strip()
        cid = str(cit.get("id", "")).strip()
        if not kind or not cid:
            return None

        if kind == "chunk":
            try:
                chunk_id = int(cid)
            except Exception:
                return None
            ch = self._chunks_by_id.get(chunk_id)
            if not ch:
                return None
            doc_id = str(ch.get("doc_id", ""))
            section = str(ch.get("section", "unknown"))
            snippet = _compact(str(ch.get("text", "")), max_chars=snippet_chars)
            return CitationResolved(
                kind="chunk",
                id=cid,
                doc_id=doc_id,
                title=self.doc_title(doc_id),
                section=section,
                snippet=snippet,
                chunk_id=chunk_id,
                url=self.doc_url(doc_id),
                local_path=self.doc_path(doc_id),
                source_type="atlas",
            )

        if kind == "claim":
            cl = self._claims_by_id.get(cid)
            if not cl:
                return None
            doc_id = str(cl.get("doc_id", ""))
            section = str(cl.get("section", "unknown"))
            claim_text = str(cl.get("claim", ""))
            evidence_span = str(cl.get("evidence_span", "")).strip()

            # Build context-aware snippet anchored on evidence span so snippets are readable,
            # not just a single short sentence.
            chunk_id = None
            snippet = evidence_span or claim_text
            try:
                chunk_id = int(cl.get("chunk_id"))
            except Exception:
                chunk_id = None

            if chunk_id is not None and chunk_id in self._chunks_by_id:
                chunk_text = str(self._chunks_by_id[chunk_id].get("text", ""))
                if chunk_text:
                    snippet = _context_snippet_from_chunk(
                        chunk_text,
                        evidence_span=evidence_span,
                        claim_text=claim_text,
                        max_chars=snippet_chars,
                    )
            elif not snippet:
                snippet = claim_text

            return CitationResolved(
                kind="claim",
                id=cid,
                doc_id=doc_id,
                title=self.doc_title(doc_id),
                section=section,
                snippet=_compact(snippet, max_chars=snippet_chars),
                chunk_id=chunk_id,
                url=self.doc_url(doc_id),
                local_path=self.doc_path(doc_id),
                source_type="atlas",
            )

        if kind == "external":
            title = str(cit.get("title", "External source")).strip() or "External source"
            section = str(cit.get("section", "external")).strip() or "external"
            snippet = _compact(str(cit.get("snippet", "")), max_chars=snippet_chars)
            return CitationResolved(
                kind="external",
                id=cid,
                doc_id=str(cit.get("doc_id", "")),
                title=title,
                section=section,
                snippet=snippet,
                chunk_id=None,
                url=str(cit.get("url") or "") or None,
                local_path=None,
                source_type=str(cit.get("source_type", "external_scholarly")),
            )

        return None

    def resolve_many(self, citations: List[Dict[str, str]], *, snippet_chars: int = DEFAULT_SNIPPET_CHARS) -> List[CitationResolved]:
        out: List[CitationResolved] = []
        seen = set()
        for c in citations or []:
            key = (c.get("kind"), c.get("id"))
            if key in seen:
                continue
            seen.add(key)
            r = self.resolve_one(c, snippet_chars=snippet_chars)
            if r:
                out.append(r)
        return out
