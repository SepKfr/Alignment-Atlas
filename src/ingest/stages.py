from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import requests
import networkx as nx
import numpy as np
import torch
import faiss
from pypdf import PdfReader
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from transformers import AutoModel, AutoTokenizer
from openai import APIStatusError, AsyncOpenAI, RateLimitError

ROOT = Path(__file__).resolve().parents[2]

ProgressCallback = Callable[[Dict[str, object]], None]


@dataclass
class StageResult:
    module: str
    ok: bool
    return_code: int
    elapsed_seconds: float
    output_tail: str


class ModuleStage:
    """
    OOP wrapper for existing stage entrypoints.
    Keeps legacy module implementations intact while providing a compact
    class-based orchestration interface.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        cmd = [sys.executable, "-m", self.module_name]
        started = time.time()
        timeout_s = timeout_seconds if timeout_seconds is not None else float(
            os.environ.get("INGEST_STAGE_TIMEOUT_SECONDS", "600")
        )
        env = os.environ.copy()
        if env_overrides:
            env.update({k: str(v) for k, v in env_overrides.items() if v is not None})
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        output_lines: List[str] = []
        timeout_hit = False
        assert proc.stdout is not None

        while True:
            line = proc.stdout.readline()
            if line:
                line = line.rstrip("\n")
                print(line, flush=True)
                output_lines.append(line)
                if len(output_lines) > 1200:
                    output_lines = output_lines[-1200:]
                if progress_callback is not None:
                    progress_callback(
                        {
                            "current_stage": self.module_name,
                            "stage_index": stage_index,
                            "stage_total": stage_total,
                            "stage_results": stage_results or [],
                            "detail_message": line,
                        }
                    )

            if proc.poll() is not None:
                break

            elapsed = time.time() - started
            if elapsed >= timeout_s:
                timeout_hit = True
                msg = (
                    f"[stage][timeout] {self.module_name} exceeded {timeout_s:.0f}s; "
                    "terminating subprocess"
                )
                print(msg, flush=True)
                output_lines.append(msg)
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                break

        proc.wait()
        elapsed = round(time.time() - started, 2)
        if timeout_hit:
            output_lines.append(
                f"[stage][timeout] {self.module_name} exceeded {timeout_s:.0f}s and was terminated."
            )
        return StageResult(
            module=self.module_name,
            ok=proc.returncode == 0,
            return_code=int(proc.returncode or 0),
            elapsed_seconds=elapsed,
            output_tail="\n".join(output_lines).strip()[-6000:],
        )


@dataclass
class IngestPaths:
    docs_jsonl: Path
    text_dir: Path
    chunks_jsonl: Path
    chunks_with_neighbors: Path
    neighbors_dir: Path
    raw_pdfs_dir: Path
    raw_html_dir: Path


class IncrementalStageOps:
    """
    OOP helper for incremental ingest operations.
    This centralizes stage internals outside AtlasService and is the
    migration step toward fully self-contained OOP stage logic.
    """

    def __init__(self, paths: IngestPaths):
        self.paths = paths

    @staticmethod
    def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not path.exists():
            return out
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def _load_docs_map(self) -> Dict[str, Dict[str, Any]]:
        rows = self._iter_jsonl(self.paths.docs_jsonl)
        return {str(r.get("doc_id")): r for r in rows if r.get("doc_id")}

    def download_single_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        ua = "alignment-britannica/0.1 (+research demo)"
        timeout = 60
        retries = 3
        source_type = str(doc.get("source_type", "html"))
        doc_id = str(doc["doc_id"])
        if source_type == "pdf":
            out = self.paths.raw_pdfs_dir / f"{doc_id}.pdf"
        else:
            out = self.paths.raw_html_dir / f"{doc_id}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            return {"doc_id": doc_id, "source_path": str(out.as_posix()), "skipped": True}
        last_err = None
        content = b""
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(
                    str(doc["source_url"]),
                    headers={"User-Agent": ua},
                    timeout=timeout,
                    allow_redirects=True,
                )
                r.raise_for_status()
                content = r.content
                break
            except Exception as e:
                last_err = e
                time.sleep(0.6 * attempt)
        if not content:
            raise RuntimeError(f"Failed to download {doc['source_url']}: {last_err}")
        out.write_bytes(content)
        return {"doc_id": doc_id, "source_path": str(out.as_posix()), "bytes": len(content), "skipped": False}

    def materialize_single_text(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        docs_map = self._load_docs_map()
        doc_id = str(doc["doc_id"])
        d = docs_map.get(doc_id)
        if not d:
            raise FileNotFoundError(f"Doc {doc_id} missing from {self.paths.docs_jsonl}")
        self.paths.text_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.paths.text_dir / f"{doc_id}.txt"
        if out_path.exists():
            return {
                "doc_id": doc_id,
                "text_path": str(out_path.as_posix()),
                "chars": len(out_path.read_text(encoding="utf-8", errors="ignore")),
                "skipped": True,
            }
        if str(d.get("source_type")) == "pdf":
            text = PdfToTextStage.extract_pdf_text_for_doc(doc_id, str(d["source_pdf"]))
        else:
            html = Path(str(d["source_html"])).read_text(encoding="utf-8", errors="ignore")
            text = HtmlToTextStage.html_bytes_to_text(html)
        out_path.write_text(text, encoding="utf-8")
        return {"doc_id": doc_id, "text_path": str(out_path.as_posix()), "chars": len(text)}

    def append_single_doc_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        if self.paths.chunks_jsonl.exists():
            for rec in self._iter_jsonl(self.paths.chunks_jsonl):
                if str(rec.get("doc_id", "")) == doc_id:
                    return []
        text_path = self.paths.text_dir / f"{doc_id}.txt"
        if not text_path.exists():
            raise FileNotFoundError(f"Missing text file: {text_path}")
        raw = text_path.read_text(encoding="utf-8")
        text = SectionChunkStage.normalize_text(raw)
        sections = SectionChunkStage.split_into_sections(text)

        max_chunk_id = -1
        for rec in self._iter_jsonl(self.paths.chunks_jsonl):
            max_chunk_id = max(max_chunk_id, int(rec.get("chunk_id", -1)))
        next_chunk_id = max_chunk_id + 1

        recs: List[Dict[str, Any]] = []
        chunker = SectionChunkStage.chunk_semantic
        for sec_name, sec_body, sec_start, _ in sections:
            for start_char, end_char, ctext in chunker(sec_body, sec_start):
                recs.append(
                    {
                        "chunk_id": next_chunk_id,
                        "doc_id": doc_id,
                        "section": sec_name,
                        "text": ctext,
                        "start_char": int(start_char),
                        "end_char": int(end_char),
                        "prev_chunk_id": None,
                        "next_chunk_id": None,
                    }
                )
                next_chunk_id += 1

        if not recs:
            return []
        for i in range(len(recs)):
            recs[i]["prev_chunk_id"] = recs[i - 1]["chunk_id"] if i > 0 else None
            recs[i]["next_chunk_id"] = recs[i + 1]["chunk_id"] if i + 1 < len(recs) else None

        self.paths.chunks_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.chunks_jsonl.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.paths.chunks_with_neighbors.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.chunks_with_neighbors.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        self.paths.neighbors_dir.mkdir(parents=True, exist_ok=True)
        nbr_map = {
            str(r["chunk_id"]): {"prev_chunk_id": r["prev_chunk_id"], "next_chunk_id": r["next_chunk_id"]} for r in recs
        }
        (self.paths.neighbors_dir / f"{doc_id}.json").write_text(
            json.dumps(nbr_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return recs

    def extract_claims_for_chunks(
        self,
        chunk_recs: List[Dict[str, Any]],
        *,
        batch_size: int = 10,
        max_concurrency: int = 8,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, Any]:
        if not chunk_recs:
            return {"num_claims_written": 0, "num_batches": 0}
        extractor = ExtractClaimsStage()
        chunk_objs = [
            ExtractClaimsStage.ChunkForExtraction(
                chunk_id=int(r["chunk_id"]),
                doc_id=str(r["doc_id"]),
                section=str(r.get("section", "unknown")),
                text=ExtractClaimsStage.normalize_ws(str(r.get("text", ""))),
            )
            for r in chunk_recs
        ]
        model = os.environ.get("CLAIMS_MODEL", ExtractClaimsStage.default_model())
        written = 0
        batches: List[List[Any]] = [
            chunk_objs[i : i + batch_size] for i in range(0, len(chunk_objs), batch_size)
        ]
        calls_total = len(batches)
        calls_done = 0
        request_timeout = float(os.environ.get("CLAIMS_REQUEST_TIMEOUT_SECONDS", "90"))
        max_retries = int(os.environ.get("CLAIMS_REQUEST_RETRIES", "2"))

        async def _run_batches() -> None:
            nonlocal written, calls_done
            semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
            client = AsyncOpenAI()

            async def _one(batch: List[Any], batch_idx: int) -> Dict[str, Any]:
                last_err: Optional[Exception] = None
                for attempt in range(max_retries + 1):
                    try:
                        async with semaphore:
                            return await asyncio.wait_for(
                                extractor.call_openai_extract_async(
                                    client=client,
                                    model=model,
                                    chunks=batch,
                                    max_claims_per_chunk=ExtractClaimsStage.default_max_claims_per_chunk(),
                                    temperature=0.0,
                                ),
                                timeout=request_timeout,
                            )
                    except Exception as e:
                        last_err = e
                        if attempt >= max_retries:
                            break
                        await asyncio.sleep(min(4.0, 0.6 * (2**attempt)))
                raise RuntimeError(f"batch {batch_idx} failed after retries: {last_err}")

            tasks = {asyncio.create_task(_one(batch, idx)): idx for idx, batch in enumerate(batches, 1)}
            if progress_callback is not None:
                started_msg = f"[claims] started {calls_total} API calls (concurrency {max_concurrency})"
                print(started_msg, flush=True)
                progress_callback(
                    {
                        "current_stage": "incremental.extract_claims",
                        "stage_index": stage_index,
                        "stage_total": stage_total,
                        "stage_results": stage_results or [],
                        "detail_message": started_msg,
                        "claim_calls_done": calls_done,
                        "claim_calls_total": calls_total,
                        "claims_written": written,
                    }
                )

            pending = set(tasks.keys())
            while pending:
                done, pending = await asyncio.wait(pending, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    if progress_callback is not None:
                        heartbeat_msg = f"[claims] running... {calls_done}/{calls_total} complete"
                        print(heartbeat_msg, flush=True)
                        progress_callback(
                            {
                                "current_stage": "incremental.extract_claims",
                                "stage_index": stage_index,
                                "stage_total": stage_total,
                                "stage_results": stage_results or [],
                                "detail_message": heartbeat_msg,
                                "claim_calls_done": calls_done,
                                "claim_calls_total": calls_total,
                                "claims_written": written,
                            }
                        )
                    continue

                for t in done:
                    batch_idx = tasks[t]
                    try:
                        data = t.result()
                        items = data.get("items", []) or []
                        for item in items:
                            extractor.save_cache_for_chunk(item)
                        written += extractor.write_claims_jsonl(items)
                        calls_done += 1
                        msg = f"[claims] call {calls_done}/{calls_total}, claims written so far: {written}"
                    except Exception as e:
                        calls_done += 1
                        msg = f"[claims][error] batch {batch_idx}/{calls_total}: {e}"
                    print(msg, flush=True)
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "current_stage": "incremental.extract_claims",
                                "stage_index": stage_index,
                                "stage_total": stage_total,
                                "stage_results": stage_results or [],
                                "detail_message": msg,
                                "claim_calls_done": calls_done,
                                "claim_calls_total": calls_total,
                                "claims_written": written,
                            }
                        )

        asyncio.run(_run_batches())
        return {
            "num_claims_written": written,
            "num_batches": calls_total,
            "model": model,
            "batch_size": batch_size,
            "max_concurrency": max_concurrency,
        }


class CollectManifestStage:
    name = "collect_manifest"

    def __init__(self) -> None:
        self.data_dir = ROOT / "data"
        self.processed_dir = self.data_dir / "processed"
        self.papers_jsonl = self.data_dir / "papers.jsonl"
        self.docs_jsonl = self.processed_dir / "docs.jsonl"
        self.ss_timeout = float(os.environ.get("SEMANTIC_SCHOLAR_TIMEOUT_SECONDS", "8"))
        self.ss_api = "https://api.semanticscholar.org/graph/v1/paper"
        self.ss_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()

    @staticmethod
    def _canonicalize_source_url(url: str) -> str:
        u = (url or "").strip()
        if not u:
            return ""
        p = urlparse(u)
        scheme = (p.scheme or "https").lower()
        netloc = (p.netloc or "").lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = re.sub(r"/+", "/", p.path or "/")
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        keep = []
        for k, v in parse_qsl(p.query or "", keep_blank_values=False):
            kl = (k or "").lower()
            if kl.startswith("utm_") or kl in {"ref", "source", "fbclid", "gclid"}:
                continue
            keep.append((k, v))
        query = urlencode(keep, doseq=True)
        return urlunparse((scheme, netloc, path, "", query, ""))

    @staticmethod
    def _detect_source_type(url: str) -> str:
        u = (url or "").lower().strip()
        if u.endswith(".pdf") or "/pdf/" in u or "arxiv.org/pdf/" in u:
            return "pdf"
        return "html"

    @staticmethod
    def _infer_year_from_url(url: str) -> int | None:
        m = re.search(r"(?:arxiv\.org/(?:abs|pdf)/)(\d{2})(\d{2})\.\d{4,5}", (url or "").lower())
        if m:
            return 2000 + int(m.group(1))
        return None

    @staticmethod
    def _title_from_url(url: str) -> str:
        p = urlparse(url)
        host = (p.netloc or "").lower().replace("www.", "")
        path = (p.path or "").strip("/")
        if "arxiv.org" in host:
            m = re.search(r"/(?:abs|pdf)/([^/]+)", p.path or "")
            if m:
                return f"arxiv {m.group(1).replace('.pdf', '')}"
        tail = (path.split("/")[-1] if path else host) or "source"
        tail = re.sub(r"\.pdf$", "", tail, flags=re.I)
        tail = re.sub(r"[-_]+", " ", tail).strip()
        return tail or host or "source"

    @staticmethod
    def _arxiv_id_from_url(url: str) -> str:
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", (url or "").lower())
        if not m:
            return ""
        return m.group(1).replace(".pdf", "")

    @staticmethod
    def _is_placeholder_title(title: str) -> bool:
        t = re.sub(r"\s+", " ", (title or "").strip()).lower()
        if not t:
            return True
        return bool(re.match(r"^arxiv\s+\d{4}\.\d{4,5}(?:v\d+)?$", t))

    def _semantic_scholar_lookup_title_and_year(self, url: str) -> tuple[str, int | None]:
        headers = {"User-Agent": "alignment-atlas/0.1"}
        if self.ss_api_key:
            headers["x-api-key"] = self.ss_api_key
        fields = "title,year"
        try:
            u = f"{self.ss_api}/URL:{quote(url, safe='')}"
            r = requests.get(u, headers=headers, params={"fields": fields}, timeout=self.ss_timeout)
            if r.status_code == 200:
                obj = r.json()
                return str(obj.get("title") or "").strip(), obj.get("year")
        except Exception:
            pass
        arxiv_id = self._arxiv_id_from_url(url)
        if not arxiv_id:
            return "", None
        try:
            u = f"{self.ss_api}/ARXIV:{quote(arxiv_id, safe='')}"
            r = requests.get(u, headers=headers, params={"fields": fields}, timeout=self.ss_timeout)
            if r.status_code == 200:
                obj = r.json()
                return str(obj.get("title") or "").strip(), obj.get("year")
        except Exception:
            pass
        return "", None

    def _iter_papers(self) -> Iterable[Dict[str, Any]]:
        if not self.papers_jsonl.exists():
            raise FileNotFoundError(f"Missing {self.papers_jsonl}")
        with self.papers_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        seen_urls = set()
        n = 0
        with self.docs_jsonl.open("w", encoding="utf-8") as out:
            for p in self._iter_papers():
                source_url = str(p.get("source_url", "")).strip()
                if not source_url:
                    continue
                canonical_url = self._canonicalize_source_url(source_url)
                if canonical_url in seen_urls:
                    continue
                seen_urls.add(canonical_url)
                source_type = str(p.get("source_type") or self._detect_source_type(source_url)).strip().lower()
                if source_type not in {"pdf", "html"}:
                    source_type = self._detect_source_type(source_url)
                title = str(p.get("title", "")).strip() or self._title_from_url(canonical_url or source_url)
                year = p.get("year")
                if year is None:
                    year = self._infer_year_from_url(canonical_url or source_url)
                if self._is_placeholder_title(title):
                    resolved_title, resolved_year = self._semantic_scholar_lookup_title_and_year(canonical_url or source_url)
                    if resolved_title:
                        title = resolved_title
                    if year is None and resolved_year is not None:
                        year = resolved_year
                basis = canonical_url or source_url
                doc_id = str(p.get("doc_id", "")).strip() or f"doc_{hashlib.sha1(basis.encode('utf-8')).hexdigest()[:12]}"
                doc = {
                    "doc_id": doc_id,
                    "title": title,
                    "year": year,
                    "source_type": source_type,
                    "source_url": source_url,
                    "canonical_source_url": canonical_url or source_url,
                    "source_pdf": str((self.data_dir / "raw_pdfs" / f"{doc_id}.pdf").as_posix()) if source_type == "pdf" else None,
                    "source_html": str((self.data_dir / "raw_html" / f"{doc_id}.html").as_posix()) if source_type == "html" else None,
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                n += 1
        msg = f"Wrote {n} docs -> {self.docs_jsonl}"
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class DedupeManifestStage:
    name = "dedupe_manifest"

    def __init__(self) -> None:
        self.root = ROOT
        self.default_path = self.root / "data" / "papers.jsonl"

    @staticmethod
    def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    @staticmethod
    def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    @staticmethod
    def _canonicalize_source_url(url: str) -> str:
        u = (url or "").strip()
        if not u:
            return ""
        p = urlparse(u)
        scheme = (p.scheme or "https").lower()
        netloc = (p.netloc or "").lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = re.sub(r"/+", "/", p.path or "/")
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        keep = []
        for k, v in parse_qsl(p.query or "", keep_blank_values=False):
            kl = (k or "").lower()
            if kl.startswith("utm_") or kl in {"ref", "source", "fbclid", "gclid"}:
                continue
            keep.append((k, v))
        query = urlencode(keep, doseq=True)
        return urlunparse((scheme, netloc, path, "", query, ""))

    @staticmethod
    def _title_from_url(url: str) -> str:
        p = urlparse(url)
        host = (p.netloc or "").lower().replace("www.", "")
        path = (p.path or "").strip("/")
        if "arxiv.org" in host:
            m = re.search(r"/(?:abs|pdf)/([^/]+)", p.path or "")
            if m:
                return f"arxiv {m.group(1).replace('.pdf', '')}"
        tail = (path.split("/")[-1] if path else host) or "source"
        tail = re.sub(r"\.pdf$", "", tail, flags=re.I)
        tail = re.sub(r"[-_]+", " ", tail).strip()
        return tail or host or "source"

    @staticmethod
    def _normalize_title(title: str, canonical_url: str) -> tuple[str, str]:
        display = re.sub(r"\s+", " ", (title or "").strip()) or DedupeManifestStage._title_from_url(canonical_url)
        normalized = display.lower().strip()
        return display, normalized

    @staticmethod
    def _quality_score(row: Dict[str, Any]) -> float:
        score = 0.0
        title = str(row.get("title", "")).strip()
        did = str(row.get("doc_id", "")).strip()
        year = row.get("year")
        if year is not None:
            score += 3.0
        if title:
            score += 1.0
        if title and not title.isupper():
            score += 1.5
        if did and not did.startswith("doc_"):
            score += 0.7
        score += min(len(title), 120) / 300.0
        return score

    @staticmethod
    def _dedupe_rows(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        by_url: Dict[str, List[Dict[str, Any]]] = {}
        missing_url: List[Dict[str, Any]] = []
        for r in rows:
            url = DedupeManifestStage._canonicalize_source_url(str(r.get("source_url", "")))
            if not url:
                missing_url.append(r)
                continue
            by_url.setdefault(url, []).append(r)

        out: List[Dict[str, Any]] = []
        duplicates_removed = 0
        duplicate_groups = 0
        examples: List[str] = []
        for canon_url, group in by_url.items():
            sorted_group = sorted(group, key=DedupeManifestStage._quality_score, reverse=True)
            keep = dict(sorted_group[0])
            if len(sorted_group) > 1:
                duplicate_groups += 1
                duplicates_removed += len(sorted_group) - 1
                examples.append(canon_url)
            keep_title, keep_title_norm = DedupeManifestStage._normalize_title(str(keep.get("title", "")), canon_url)
            keep["title"] = keep_title
            keep["title_normalized"] = keep_title_norm
            keep["canonical_source_url"] = canon_url
            keep["source_url"] = str(keep.get("source_url", "")).strip() or canon_url
            if not keep.get("source_type"):
                keep["source_type"] = "pdf" if canon_url.endswith(".pdf") or "/pdf/" in canon_url else "html"
            out.append(keep)

        for r in missing_url:
            k = dict(r)
            display, normalized = DedupeManifestStage._normalize_title(str(k.get("title", "")), "")
            k["title"] = display
            k["title_normalized"] = normalized
            k["canonical_source_url"] = ""
            out.append(k)

        out.sort(key=lambda x: str(x.get("doc_id", "")))
        stats = {
            "rows_before": len(rows),
            "rows_after": len(out),
            "duplicates_removed": duplicates_removed,
            "duplicate_groups": duplicate_groups,
            "rows_missing_url": len(missing_url),
            "example_duplicate_urls": examples[:10],
        }
        return out, stats

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        apply_changes = os.environ.get("INGEST_DEDUPE_APPLY", "0").strip().lower() in {"1", "true", "yes"}
        rows = self._iter_jsonl(self.default_path)
        deduped, stats = self._dedupe_rows(rows)
        print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
        if apply_changes:
            backup = self.default_path.with_suffix(self.default_path.suffix + f".bak.{int(time.time())}")
            backup.write_text(self.default_path.read_text(encoding="utf-8"), encoding="utf-8")
            self._write_jsonl(self.default_path, deduped)
            msg = f"Wrote deduped manifest -> {self.default_path} | backup={backup}"
        else:
            msg = "Dry-run only (set INGEST_DEDUPE_APPLY=1 to apply)."
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class DownloadSourcesStage:
    name = "download_sources"

    def __init__(self) -> None:
        self.root = ROOT
        self.data_dir = ROOT / "data"
        self.docs_jsonl = self.data_dir / "processed" / "docs.jsonl"
        self.raw_pdfs_dir = self.data_dir / "raw_pdfs"
        self.raw_html_dir = self.data_dir / "raw_html"
        self.ua = "alignment-britannica/0.1 (+research demo)"
        self.timeout = 60
        self.retries = 3
        self.sleep = 0.6

    def _iter_docs(self) -> Iterable[Dict[str, Any]]:
        if not self.docs_jsonl.exists():
            raise FileNotFoundError(f"Missing {self.docs_jsonl}. Run collect manifest first.")
        with self.docs_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _download(self, url: str) -> bytes:
        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.get(url, headers={"User-Agent": self.ua}, timeout=self.timeout, allow_redirects=True)
                r.raise_for_status()
                return r.content
            except Exception as e:
                last_err = e
                time.sleep(0.6 * attempt)
        raise RuntimeError(f"Failed to download {url}: {last_err}")

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        self.raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.raw_html_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in self._iter_docs():
            n += 1
            doc_id = p["doc_id"]
            stype = p["source_type"]
            url = p["source_url"]
            if stype == "pdf":
                out = self.raw_pdfs_dir / f"{doc_id}.pdf"
            elif stype == "html":
                out = self.raw_html_dir / f"{doc_id}.html"
            else:
                raise ValueError(f"Unknown source_type={stype} for {doc_id}")
            if out.exists():
                print(f"[{n}] {doc_id}: exists -> skip", flush=True)
                continue
            content = self._download(url)
            out.write_bytes(content)
            print(f"[{n}] {doc_id}: saved -> {out}", flush=True)
            time.sleep(self.sleep)
        msg = "Done."
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class ApplyNeighborsStage:
    name = "apply_neighbors"

    def __init__(self) -> None:
        processed = ROOT / "data" / "processed"
        self.chunks_jsonl = processed / "chunks.jsonl"
        self.neighbors_dir = processed / "neighbors"
        self.out_jsonl = processed / "chunks_with_neighbors.jsonl"

    def _load_neighbor_map(self, doc_id: str) -> Dict[str, Dict[str, Optional[int]]]:
        path = self.neighbors_dir / f"{doc_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing neighbor map for doc_id={doc_id}: {path}")
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Neighbor map {path} must be a JSON object.")
        return obj

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        if not self.chunks_jsonl.exists():
            raise FileNotFoundError(f"Missing {self.chunks_jsonl}. Run chunk stage first.")
        if not self.neighbors_dir.exists():
            raise FileNotFoundError(f"Missing {self.neighbors_dir}. Run chunk stage first.")
        out = self.out_jsonl.open("w", encoding="utf-8")
        total = 0
        updated = 0
        current_doc_id: Optional[str] = None
        neighbor_map: Optional[Dict[str, Dict[str, Optional[int]]]] = None
        try:
            with self.chunks_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    total += 1
                    doc_id = rec.get("doc_id")
                    if doc_id != current_doc_id:
                        current_doc_id = doc_id
                        neighbor_map = self._load_neighbor_map(str(doc_id))
                    cid_key = str(rec.get("chunk_id"))
                    if neighbor_map and cid_key in neighbor_map:
                        nbr = neighbor_map[cid_key]
                        rec["prev_chunk_id"] = nbr.get("prev_chunk_id")
                        rec["next_chunk_id"] = nbr.get("next_chunk_id")
                        updated += 1
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        finally:
            out.close()
        msg = f"Wrote: {self.out_jsonl} | Total chunks: {total} | Updated with neighbors: {updated}"
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class PdfToTextStage:
    name = "pdf_to_text"
    _grobid_health_cache: Dict[str, bool] = {}

    def __init__(self) -> None:
        self.processed_dir = ROOT / "data" / "processed"
        self.docs_jsonl = self.processed_dir / "docs.jsonl"
        self.text_dir = self.processed_dir / "text"
        self.sections_dir = self.processed_dir / "sections"

    @staticmethod
    def _grobid_url() -> str:
        return os.environ.get("GROBID_URL", "http://localhost:8070/api/processFulltextDocument")

    @staticmethod
    def _grobid_timeout() -> int:
        return int(os.environ.get("GROBID_TIMEOUT_SECONDS", "45"))

    @staticmethod
    def _grobid_health_timeout() -> float:
        return float(os.environ.get("GROBID_HEALTH_TIMEOUT_SECONDS", "1.5"))

    @staticmethod
    def _grobid_enabled() -> bool:
        return os.environ.get("GROBID_ENABLED", "1").strip().lower() not in {"0", "false", "no"}

    @staticmethod
    def _grobid_required() -> bool:
        return os.environ.get("GROBID_REQUIRED", "1").strip().lower() not in {"0", "false", "no"}

    @staticmethod
    def _overwrite_text() -> bool:
        return os.environ.get("PDF_TEXT_OVERWRITE", "0").strip().lower() in {"1", "true", "yes"}

    def _iter_docs(self) -> Iterable[Dict[str, Any]]:
        if not self.docs_jsonl.exists():
            raise FileNotFoundError(f"Missing {self.docs_jsonl}. Run collect manifest first.")
        with self.docs_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def extract_pdf_text(pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        pages: List[str] = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            pages.append(f"\n\n=== PAGE {i + 1} ===\n\n{t}")
        return "\n".join(pages).strip()

    @staticmethod
    def _norm_ws(text: str) -> str:
        return " ".join((text or "").split()).strip()

    @classmethod
    def _extract_sections_from_tei(cls, tei_xml: str) -> List[Dict[str, str]]:
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        root = ET.fromstring(tei_xml)
        sections: List[Dict[str, str]] = []
        abstract_parts: List[str] = []
        for p in root.findall(".//tei:profileDesc/tei:abstract//tei:p", ns):
            txt = cls._norm_ws("".join(p.itertext()))
            if txt:
                abstract_parts.append(txt)
        if abstract_parts:
            sections.append({"section": "Abstract", "text": "\n\n".join(abstract_parts)})
        for div in root.findall(".//tei:text/tei:body//tei:div", ns):
            head_el = div.find("./tei:head", ns)
            if head_el is None:
                continue
            head_txt = cls._norm_ws("".join(head_el.itertext()))
            if not head_txt:
                continue
            n = (head_el.attrib.get("n") or "").strip()
            section_title = head_txt if (not n or head_txt.startswith(n)) else f"{n} {head_txt}"
            paras: List[str] = []
            for p in div.findall("./tei:p", ns):
                txt = cls._norm_ws("".join(p.itertext()))
                if txt:
                    paras.append(txt)
            if paras:
                sections.append({"section": section_title, "text": "\n\n".join(paras)})
        return sections

    @classmethod
    def _grobid_sections_from_pdf(cls, pdf_path: str) -> List[Dict[str, str]]:
        with open(pdf_path, "rb") as f:
            files = {"input": (Path(pdf_path).name, f, "application/pdf")}
            data = {
                "includeRawAffiliations": "0",
                "includeRawCitations": "0",
                "consolidateCitations": "0",
                "consolidateHeader": "0",
            }
            r = requests.post(cls._grobid_url(), files=files, data=data, timeout=cls._grobid_timeout())
            r.raise_for_status()
            tei_xml = r.text
        return cls._extract_sections_from_tei(tei_xml)

    @classmethod
    def _grobid_health_url(cls) -> str:
        grobid_url = cls._grobid_url()
        if "/api/" in grobid_url:
            base = grobid_url.split("/api/", 1)[0].rstrip("/")
            return f"{base}/api/isalive"
        return grobid_url.rstrip("/") + "/api/isalive"

    @classmethod
    def _grobid_is_alive(cls) -> bool:
        if not cls._grobid_enabled():
            return False
        cache_key = f"{cls._grobid_url()}|{cls._grobid_health_timeout()}"
        if cache_key in cls._grobid_health_cache:
            return cls._grobid_health_cache[cache_key]
        try:
            r = requests.get(cls._grobid_health_url(), timeout=cls._grobid_health_timeout())
            ok = r.status_code == 200
        except Exception:
            ok = False
        cls._grobid_health_cache[cache_key] = ok
        return ok

    @staticmethod
    def _sections_to_text(sections: List[Dict[str, str]]) -> str:
        blocks: List[str] = []
        for s in sections:
            name = (s.get("section") or "").strip()
            body = (s.get("text") or "").strip()
            if not body:
                continue
            blocks.append(f"{name}\n{body}" if name else body)
        return "\n\n".join(blocks).strip()

    @classmethod
    def extract_pdf_text_for_doc(cls, doc_id: str, pdf_path: str) -> str:
        sections_dir = ROOT / "data" / "processed" / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = sections_dir / f"{doc_id}.sections.json"
        if cls._grobid_is_alive():
            try:
                sections = cls._grobid_sections_from_pdf(pdf_path)
                if sections:
                    sidecar = {
                        "doc_id": doc_id,
                        "source_pdf": pdf_path,
                        "parser": "grobid",
                        "sections": sections,
                    }
                    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
                    return cls._sections_to_text(sections)
            except Exception as e:
                if cls._grobid_required():
                    raise RuntimeError(
                        f"{doc_id}: GROBID extraction failed and GROBID_REQUIRED=1 ({type(e).__name__}: {e})"
                    ) from e
                print(
                    f"[warn] {doc_id}: GROBID extraction failed, falling back to pypdf ({type(e).__name__}: {e})",
                    flush=True,
                )
        else:
            if cls._grobid_required():
                raise RuntimeError(f"{doc_id}: GROBID unavailable at {cls._grobid_health_url()} with GROBID_REQUIRED=1.")
            print(f"[warn] {doc_id}: GROBID unavailable, skipping straight to pypdf fallback.", flush=True)
        if sidecar_path.exists():
            sidecar_path.unlink()
        return cls.extract_pdf_text(pdf_path)

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.sections_dir.mkdir(parents=True, exist_ok=True)
        if self._grobid_required() and not self._grobid_is_alive():
            raise RuntimeError(
                "GROBID_REQUIRED=1 but GROBID is not reachable. "
                f"Expected healthy endpoint: {self._grobid_health_url()}"
            )
        n = 0
        skipped = 0
        for doc in self._iter_docs():
            if not doc.get("source_pdf"):
                continue
            doc_id = str(doc["doc_id"])
            pdf_path = str(doc["source_pdf"])
            out_path = self.text_dir / f"{doc_id}.txt"
            if out_path.exists() and not self._overwrite_text():
                skipped += 1
                print(f"[skip] {doc_id} -> {out_path} (already exists)", flush=True)
                continue
            text = self.extract_pdf_text_for_doc(doc_id, pdf_path)
            out_path.write_text(text, encoding="utf-8")
            n += 1
            print(f"[{n}] {doc_id} -> {out_path} ({len(text):,} chars)", flush=True)
        msg = f"Done. Wrote {n} text files to {self.text_dir} (skipped {skipped} existing)"
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class HtmlToTextStage:
    name = "html_to_text"

    def __init__(self) -> None:
        data_dir = ROOT / "data"
        self.processed_dir = data_dir / "processed"
        self.docs_jsonl = self.processed_dir / "docs.jsonl"
        self.text_dir = self.processed_dir / "text"

    def _iter_docs(self) -> Iterable[Dict[str, Any]]:
        if not self.docs_jsonl.exists():
            raise FileNotFoundError(f"Missing {self.docs_jsonl}. Run collect manifest first.")
        with self.docs_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def _clean(t: str) -> str:
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = re.sub(r"\n{3,}", "\n\n", t)
        t = re.sub(r"[ \t]+", " ", t)
        return t.strip()

    @staticmethod
    def html_bytes_to_text(html: str) -> str:
        from bs4 import BeautifulSoup  # lazy import

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "svg"]):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup.body or soup
        lines: List[str] = []
        for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote"]):
            txt = el.get_text(" ", strip=True)
            if not txt:
                continue
            if el.name in ("h1", "h2", "h3", "h4"):
                lvl = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}[el.name]
                lines.append("\n" + ("#" * lvl) + " " + txt + "\n")
            elif el.name == "li":
                lines.append(f"- {txt}")
            else:
                lines.append(txt)
        return HtmlToTextStage._clean("\n".join(lines))

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        self.text_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for doc in self._iter_docs():
            if doc.get("source_type") != "html":
                continue
            doc_id = doc["doc_id"]
            html_path = Path(doc["source_html"])
            if not html_path.exists():
                raise FileNotFoundError(f"Missing cached HTML: {html_path}")
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            text = self.html_bytes_to_text(html)
            out_path = self.text_dir / f"{doc_id}.txt"
            out_path.write_text(text, encoding="utf-8")
            n += 1
            print(f"[{n}] {doc_id} -> {out_path} ({len(text):,} chars)", flush=True)
        msg = f"Done. Wrote {n} HTML text files -> {self.text_dir}"
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class SectionChunkStage:
    name = "section_chunk"

    SECTION_PATTERNS = [
        r"abstract",
        r"introduction",
        r"background",
        r"related work",
        r"preliminaries",
        r"method",
        r"methods",
        r"approach",
        r"model",
        r"experimental setup",
        r"experiments",
        r"evaluation",
        r"results",
        r"discussion",
        r"limitations",
        r"conclusion",
        r"conclusions",
        r"appendix",
        r"references",
    ]

    KNOWN_HEADING_REGEX = re.compile(
        r"(?im)^\s*(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[\.\)]?\s*)?(" + "|".join(SECTION_PATTERNS) + r")\s*$"
    )
    NUMBERED_HEADING_REGEX = re.compile(
        r"(?im)^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[\.\)]?\s+([A-Za-z][A-Za-z0-9 ,:;()'\"/&\\-]{2,120})\s*$"
    )
    PAGE_MARK_REGEX = re.compile(r"(?m)^=== PAGE \d+ ===$")

    def __init__(self) -> None:
        processed = ROOT / "data" / "processed"
        self.docs_jsonl = processed / "docs.jsonl"
        self.text_dir = processed / "text"
        self.chunks_jsonl = processed / "chunks.jsonl"
        self.neighbors_dir = processed / "neighbors"
        self.sections_dir = processed / "sections"

    def _iter_docs(self) -> Iterable[Dict[str, Any]]:
        if not self.docs_jsonl.exists():
            raise FileNotFoundError(f"Missing {self.docs_jsonl}. Run collect manifest first.")
        with self.docs_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @classmethod
    def normalize_text(cls, text: str) -> str:
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        t = cls.PAGE_MARK_REGEX.sub("\n", t)
        t = "\n".join(line.rstrip() for line in t.splitlines())
        return t.strip()

    @staticmethod
    def _is_probable_title_heading(line: str) -> bool:
        txt = re.sub(r"\s+", " ", (line or "").strip())
        if not txt or txt.endswith((".", "?", "!")):
            return False
        words = [w for w in re.split(r"\s+", txt) if any(ch.isalpha() for ch in w)]
        if len(words) < 2 or len(words) > 12:
            return False
        upperish = 0
        for w in words:
            w_alpha = "".join(ch for ch in w if ch.isalpha())
            if not w_alpha:
                continue
            if w_alpha.isupper() or w_alpha[0].isupper():
                upperish += 1
        return upperish >= max(2, int(0.6 * len(words)))

    @classmethod
    def split_into_sections(cls, text: str) -> List[tuple[str, str, int, int]]:
        known_matches = list(cls.KNOWN_HEADING_REGEX.finditer(text))
        numbered_matches = list(cls.NUMBERED_HEADING_REGEX.finditer(text))
        heading_points: List[tuple[int, int, str]] = []
        for m in known_matches:
            heading_points.append((m.start(), m.end(), m.group(1).lower().strip()))
        for m in numbered_matches:
            heading_text = m.group(1).strip()
            if cls._is_probable_title_heading(heading_text):
                heading_points.append((m.start(), m.end(), heading_text.lower()))
        if heading_points:
            dedup = {(s, e, n): None for (s, e, n) in heading_points}
            heading_points = sorted(dedup.keys(), key=lambda x: (x[0], x[1]))
        if not heading_points:
            return [("unknown", text, 0, len(text))]
        sections: List[tuple[str, str, int, int]] = []
        for i, m in enumerate(heading_points):
            sec_name = m[2]
            start = m[1]
            end = heading_points[i + 1][0] if i + 1 < len(heading_points) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((sec_name, body, start, end))
        return sections if sections else [("unknown", text, 0, len(text))]

    def load_structured_sections(self, doc_id: str) -> List[tuple[str, str, int, int]]:
        p = self.sections_dir / f"{doc_id}.sections.json"
        if not p.exists():
            return []
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows = obj.get("sections") or []
        out: List[tuple[str, str, int, int]] = []
        cursor = 0
        for r in rows:
            sec_name = str(r.get("section") or "unknown").strip()
            sec_body = str(r.get("text") or "").strip()
            if not sec_body:
                continue
            start = cursor
            end = cursor + len(sec_body)
            out.append((sec_name, sec_body, start, end))
            cursor = end + 2
        return out

    @staticmethod
    def _sentence_start_positions(text: str) -> List[int]:
        starts = {0}
        for m in re.finditer(r"(?<=[\.\!\?])\s+(?=(?:[\"“”'(\[])?[A-Z0-9])", text):
            starts.add(m.end())
        for m in re.finditer(r"\n{2,}", text):
            starts.add(m.end())
        return sorted(s for s in starts if 0 <= s < len(text))

    @staticmethod
    def _chunk_end_boundaries(text: str) -> List[int]:
        ends = {len(text)}
        for m in re.finditer(r"(?<=[\.\!\?])\s+(?=(?:[\"“”'(\[])?[A-Z0-9])", text):
            ends.add(m.start())
        for m in re.finditer(r"\n{2,}", text):
            ends.add(m.start())
        return sorted(e for e in ends if 0 < e <= len(text))

    @staticmethod
    def _count_sentences(text: str) -> int:
        return len(re.findall(r"[\.\!\?](?:[\"”'\)\]]+)?(?=\s|$)", text or ""))

    @classmethod
    def chunk_semantic(cls, section_text: str, base_start: int, chunk_size: int = 1200):
        t = re.sub(r"[ \t]+", " ", section_text).strip()
        if not t:
            return
        starts = cls._sentence_start_positions(t)
        ends = cls._chunk_end_boundaries(t)
        min_chunk = max(280, int(chunk_size * 0.35))
        n = len(t)
        i = 0
        while i < n:
            target_end = min(n, i + chunk_size)
            min_end = min(n, i + min_chunk)
            left = bisect_left(ends, min_end)
            right = bisect_right(ends, target_end)
            if left < right:
                j = ends[right - 1]
            else:
                after = bisect_left(ends, target_end)
                if after < len(ends) and ends[after] <= min(n, target_end + 140):
                    j = ends[after]
                else:
                    j = target_end
            if j <= i:
                j = min(n, i + chunk_size)
                if j <= i:
                    break
            chunk_text = t[i:j].strip()
            if chunk_text:
                while cls._count_sentences(chunk_text) < 2 and j < n:
                    after = bisect_left(ends, j + 1)
                    if after < len(ends):
                        j2 = ends[after]
                        if j2 <= j:
                            break
                        j = j2
                        chunk_text = t[i:j].strip()
                    else:
                        break
                yield base_start + i, base_start + j, chunk_text
            if j >= n:
                break
            target_next = max(i + 1, j)
            nxt_idx = bisect_left(starts, target_next)
            next_i = starts[nxt_idx] if nxt_idx < len(starts) else target_next
            if next_i <= i:
                next_i = max(i + 1, j)
            i = next_i

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        if not self.text_dir.exists():
            raise FileNotFoundError(f"Missing {self.text_dir}. Run text extraction stage first.")
        self.chunks_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.neighbors_dir.mkdir(parents=True, exist_ok=True)
        chunk_id = 0
        with self.chunks_jsonl.open("w", encoding="utf-8") as out:
            for doc in self._iter_docs():
                doc_id = str(doc["doc_id"])
                path = self.text_dir / f"{doc_id}.txt"
                if not path.exists():
                    raise FileNotFoundError(f"Missing text for {doc_id}: {path}.")
                text = self.normalize_text(path.read_text(encoding="utf-8"))
                sections = self.load_structured_sections(doc_id) if str(doc.get("source_type")) == "pdf" else []
                if not sections:
                    sections = self.split_into_sections(text)
                doc_chunk_ids: List[int] = []
                for sec_name, sec_body, sec_start, _sec_end in sections:
                    for start_char, end_char, ctext in self.chunk_semantic(sec_body, sec_start):
                        rec = {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "section": sec_name,
                            "text": ctext,
                            "start_char": int(start_char),
                            "end_char": int(end_char),
                            "prev_chunk_id": None,
                            "next_chunk_id": None,
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        doc_chunk_ids.append(chunk_id)
                        chunk_id += 1
                nm = {}
                for i, cid in enumerate(doc_chunk_ids):
                    nm[str(cid)] = {
                        "prev_chunk_id": doc_chunk_ids[i - 1] if i > 0 else None,
                        "next_chunk_id": doc_chunk_ids[i + 1] if i + 1 < len(doc_chunk_ids) else None,
                    }
                (self.neighbors_dir / f"{doc_id}.json").write_text(
                    json.dumps(nm, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"{doc_id}: wrote {len(doc_chunk_ids)} chunks (+ neighbor map)", flush=True)
        msg = f"All chunks appended -> {self.chunks_jsonl}"
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class ExtractClaimsStage:
    name = "extract_claims"

    CLAIMS_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "integer"},
                        "doc_id": {"type": "string"},
                        "section": {"type": "string"},
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claim": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "definition",
                                            "result",
                                            "method",
                                            "assumption",
                                            "warning",
                                            "recommendation",
                                            "other",
                                        ],
                                    },
                                    "evidence_span": {"type": "string"},
                                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                    "tags": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["claim", "type", "evidence_span", "confidence", "tags"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["chunk_id", "doc_id", "section", "claims"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    @dataclass
    class ChunkForExtraction:
        chunk_id: int
        doc_id: str
        section: str
        text: str

    def __init__(self) -> None:
        processed = ROOT / "data" / "processed"
        self.chunks_with_neighbors = processed / "chunks_with_neighbors.jsonl"
        self.chunks_fallback = processed / "chunks.jsonl"
        self.out_claims_jsonl = processed / "claims.jsonl"
        self.cache_dir = processed / "cache" / "claims_by_chunk"

    @staticmethod
    def default_model() -> str:
        return os.environ.get("CLAIMS_MODEL", "gpt-4o-mini")

    @staticmethod
    def default_max_claims_per_chunk() -> int:
        return 3

    @staticmethod
    def default_batch_size() -> int:
        return int(os.environ.get("CLAIMS_BATCH_SIZE", "10"))

    @staticmethod
    def default_max_concurrency() -> int:
        return int(os.environ.get("CLAIMS_MAX_CONCURRENCY", "8"))

    @staticmethod
    def normalize_ws(s: str) -> str:
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def _iter_jsonl(self, path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def _safe_doc_folder(doc_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", (doc_id or "").strip()) or "unknown_doc"

    def cache_path_for_chunk(self, chunk_id: int, doc_id: Optional[str] = None) -> Path:
        if doc_id:
            return self.cache_dir / self._safe_doc_folder(doc_id) / f"{chunk_id}.json"
        return self.cache_dir / f"{chunk_id}.json"

    def load_done_chunk_ids_from_cache(self) -> Set[int]:
        if not self.cache_dir.exists():
            return set()
        done = set()
        for p in self.cache_dir.rglob("*.json"):
            try:
                done.add(int(p.stem))
            except ValueError:
                continue
        return done

    def choose_chunks_path(self) -> Path:
        if self.chunks_with_neighbors.exists():
            return self.chunks_with_neighbors
        if self.chunks_fallback.exists():
            return self.chunks_fallback
        raise FileNotFoundError(f"Missing {self.chunks_with_neighbors} and {self.chunks_fallback}.")

    def read_chunks_for_extraction(self, chunks_path: Path, *, skip_done: bool, max_chunks: int = 0) -> List[ChunkForExtraction]:
        done = self.load_done_chunk_ids_from_cache() if skip_done else set()
        out: List[ExtractClaimsStage.ChunkForExtraction] = []
        for rec in self._iter_jsonl(chunks_path):
            cid = rec.get("chunk_id")
            doc_id = rec.get("doc_id")
            text = rec.get("text", "")
            section = rec.get("section") or "unknown"
            if cid is None or doc_id is None:
                raise ValueError(f"Bad chunk record missing chunk_id/doc_id: {rec}")
            cid = int(cid)
            if skip_done and cid in done:
                continue
            out.append(
                self.ChunkForExtraction(
                    chunk_id=cid,
                    doc_id=str(doc_id),
                    section=str(section),
                    text=self.normalize_ws(str(text)),
                )
            )
            if max_chunks and len(out) >= max_chunks:
                break
        return out

    def build_prompt_payload(self, chunks: List[ChunkForExtraction], *, max_claims_per_chunk: int) -> str:
        payload = [
            {
                "chunk_id": ch.chunk_id,
                "doc_id": ch.doc_id,
                "section": ch.section,
                "text": ch.text,
            }
            for ch in chunks
        ]
        return json.dumps(
            {
                "instructions": {
                    "task": "Extract atomic, citation-worthy claims from research paper text chunks.",
                    "rules": [
                        f"Return 0 to {max_claims_per_chunk} claims per chunk.",
                        "Claims must be self-contained and precise (no vague 'this paper').",
                        "Evidence span must be a verbatim substring from the chunk text (short).",
                        "Confidence is 0-1. Use lower confidence if the text is ambiguous or noisy.",
                        "Tags: short snake_case topic tags (e.g., reward_hacking, rlhf, interpretability).",
                        "If the chunk is references/boilerplate/low content, return an empty claims list.",
                    ],
                },
                "chunks": payload,
            },
            ensure_ascii=False,
        )

    async def call_openai_extract_async(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        chunks: List[ChunkForExtraction],
        max_claims_per_chunk: int,
        temperature: float,
    ) -> Dict[str, Any]:
        system_msg = (
            "You are an expert at structured information extraction from AI safety research papers. "
            "Extract only what is supported by the provided text. "
            "Follow the JSON schema exactly."
        )
        user_msg = self.build_prompt_payload(chunks, max_claims_per_chunk=max_claims_per_chunk)
        resp = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "chunk_claims_extraction",
                    "strict": True,
                    "schema": self.CLAIMS_SCHEMA,
                }
            },
            temperature=temperature,
        )
        raw = resp.output_text
        if not raw:
            raise RuntimeError("Empty output_text from model response.")
        return json.loads(raw)

    @staticmethod
    def _is_rate_limit_error(err: Exception) -> bool:
        if isinstance(err, RateLimitError):
            return True
        if isinstance(err, APIStatusError):
            return int(getattr(err, "status_code", 0) or 0) == 429
        msg = str(err).lower()
        return "rate limit" in msg or "429" in msg or "too many requests" in msg

    async def _process_batch(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        batch: List[ChunkForExtraction],
        max_claims_per_chunk: int,
        temperature: float,
        semaphore: asyncio.Semaphore,
        max_retries: int,
    ) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        saw_rate_limit = False
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    return await self.call_openai_extract_async(
                        client=client,
                        model=model,
                        chunks=batch,
                        max_claims_per_chunk=max_claims_per_chunk,
                        temperature=temperature,
                    )
            except Exception as e:
                last_err = e
                if self._is_rate_limit_error(e):
                    saw_rate_limit = True
                if attempt >= max_retries:
                    break
                base = 1.2 if self._is_rate_limit_error(e) else 0.5
                await asyncio.sleep(min(8.0, base * (2**attempt)))
        tag = "RATE_LIMIT" if saw_rate_limit else "ERROR"
        raise RuntimeError(f"{tag}: Batch failed after retries: {last_err}")

    def save_cache_for_chunk(self, item: Dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        chunk_id = int(item["chunk_id"])
        doc_id = str(item.get("doc_id", "") or "")
        path = self.cache_path_for_chunk(chunk_id, doc_id=doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_claims_jsonl(self, items: List[Dict[str, Any]]) -> int:
        self.out_claims_jsonl.parent.mkdir(parents=True, exist_ok=True)
        n_written = 0
        with self.out_claims_jsonl.open("a", encoding="utf-8") as f:
            for item in items:
                chunk_id = int(item["chunk_id"])
                doc_id = str(item["doc_id"])
                section = str(item["section"])
                claims = item.get("claims", []) or []
                for i, c in enumerate(claims):
                    rec = {
                        "claim_id": f"claim:{doc_id}:{chunk_id}:{i}",
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "section": section,
                        "claim": c["claim"],
                        "type": c["type"],
                        "evidence_span": c["evidence_span"],
                        "confidence": float(c["confidence"]),
                        "tags": c["tags"],
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1
        return n_written

    async def _run_async(
        self,
        *,
        model: str,
        batch_size: int,
        max_chunks: int,
        max_claims_per_chunk: int,
        temperature: float,
        max_concurrency: int,
        min_concurrency: int,
        rate_limit_cooldown: float,
        adaptive_concurrency: bool,
        max_retries: int,
        skip_done: bool,
    ) -> Dict[str, Any]:
        chunks_path = self.choose_chunks_path()
        chunks = self.read_chunks_for_extraction(chunks_path, skip_done=skip_done, max_chunks=max_chunks)
        if not chunks:
            print("No chunks to process (maybe everything is cached).", flush=True)
            return {"api_calls": 0, "claims_written": 0}
        print(f"Chunks to extract: {len(chunks)}", flush=True)
        client = AsyncOpenAI()
        self.out_claims_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_claims_jsonl.exists():
            self.out_claims_jsonl.write_text("", encoding="utf-8")
        total_claims = 0
        batches: List[Tuple[int, List[ExtractClaimsStage.ChunkForExtraction]]] = []
        for i in range(0, len(chunks), batch_size):
            batches.append((i, chunks[i : i + batch_size]))
        total_calls = len(batches)
        completed_calls = 0
        target_concurrency = max(1, int(max_concurrency))
        min_c = max(1, int(min_concurrency))
        stable_rounds = 0
        pending_batches = list(batches)
        while pending_batches:
            window = pending_batches[:target_concurrency]
            pending_batches = pending_batches[target_concurrency:]
            semaphore = asyncio.Semaphore(max(1, target_concurrency))
            tasks = []
            names = []
            for start_idx, batch in window:
                tasks.append(
                    self._process_batch(
                        client=client,
                        model=model,
                        batch=batch,
                        max_claims_per_chunk=max_claims_per_chunk,
                        temperature=temperature,
                        semaphore=semaphore,
                        max_retries=max(0, int(max_retries)),
                    )
                )
                names.append(f"batch_{start_idx}_{start_idx + len(batch) - 1}")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            rate_limited_in_round = 0
            failed_windows: List[Tuple[int, List[ExtractClaimsStage.ChunkForExtraction]]] = []
            for (start_idx, batch), name, result in zip(window, names, results):
                if isinstance(result, Exception):
                    msg = str(result)
                    if msg.startswith("RATE_LIMIT"):
                        rate_limited_in_round += 1
                    else:
                        print(f"[ERROR] {name} failed: {result}", flush=True)
                    failed_windows.append((start_idx, batch))
                    continue
                items = result.get("items", [])
                if not isinstance(items, list):
                    failed_windows.append((start_idx, batch))
                    continue
                for item in items:
                    self.save_cache_for_chunk(item)
                total_claims += self.write_claims_jsonl(items)
                completed_calls += 1
                print(f"[call {completed_calls}/{total_calls}] {name} complete, claims written so far: {total_claims}", flush=True)
            if failed_windows:
                pending_batches.extend(failed_windows)
            if not adaptive_concurrency:
                continue
            if rate_limited_in_round > 0:
                stable_rounds = 0
                target_concurrency = max(min_c, target_concurrency - max(1, rate_limited_in_round))
                await asyncio.sleep(max(0.0, float(rate_limit_cooldown)))
            else:
                stable_rounds += 1
                if stable_rounds >= 2 and target_concurrency < int(max_concurrency):
                    target_concurrency += 1
                    stable_rounds = 0
        return {"api_calls": total_calls, "claims_written": total_claims}

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        model = os.environ.get("CLAIMS_MODEL", self.default_model())
        batch_size = int(os.environ.get("CLAIMS_BATCH_SIZE", str(self.default_batch_size())))
        max_chunks = int(os.environ.get("CLAIMS_MAX_CHUNKS", "0"))
        max_claims_per_chunk = int(os.environ.get("CLAIMS_MAX_CLAIMS_PER_CHUNK", str(self.default_max_claims_per_chunk())))
        temperature = float(os.environ.get("CLAIMS_TEMPERATURE", "0.0"))
        max_concurrency = int(os.environ.get("CLAIMS_MAX_CONCURRENCY", str(self.default_max_concurrency())))
        min_concurrency = int(os.environ.get("CLAIMS_MIN_CONCURRENCY", "2"))
        rate_limit_cooldown = float(os.environ.get("CLAIMS_RATE_LIMIT_COOLDOWN", "2.0"))
        adaptive_concurrency = os.environ.get("CLAIMS_ADAPTIVE_CONCURRENCY", "1").strip().lower() not in {"0", "false", "no"}
        max_retries = int(os.environ.get("CLAIMS_MAX_RETRIES", "2"))
        skip_done = os.environ.get("CLAIMS_SKIP_DONE", "1").strip().lower() not in {"0", "false", "no"}
        stats = asyncio.run(
            self._run_async(
                model=model,
                batch_size=batch_size,
                max_chunks=max_chunks,
                max_claims_per_chunk=max_claims_per_chunk,
                temperature=temperature,
                max_concurrency=max_concurrency,
                min_concurrency=min_concurrency,
                rate_limit_cooldown=rate_limit_cooldown,
                adaptive_concurrency=adaptive_concurrency,
                max_retries=max_retries,
                skip_done=skip_done,
            )
        )
        msg = f"Done. API calls: {stats['api_calls']}, claims written: {stats['claims_written']}"
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class BuildKGStage:
    name = "build_kg"

    def __init__(self) -> None:
        processed_dir = ROOT / "data" / "processed"
        self.docs_jsonl = processed_dir / "docs.jsonl"
        self.claims_jsonl = processed_dir / "claims.jsonl"
        self.kg_dir = processed_dir / "kg"
        self.out_graphml = self.kg_dir / "graph.graphml"
        self.out_json = self.kg_dir / "graph.json"
        self.out_stats = self.kg_dir / "stats.json"

    @staticmethod
    def _safe_str(x: Any) -> str:
        return "" if x is None else str(x)

    def _iter_jsonl(self, path: Path) -> Iterable[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _load_docs(self) -> Dict[str, Dict[str, Any]]:
        if not self.docs_jsonl.exists():
            return {}
        docs: Dict[str, Dict[str, Any]] = {}
        for d in self._iter_jsonl(self.docs_jsonl):
            did = d.get("doc_id")
            if did:
                docs[str(did)] = d
        return docs

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        add_related_by_tag = True
        max_claims_per_tag = int(os.environ.get("KG_MAX_CLAIMS_PER_TAG", "80"))
        self.kg_dir.mkdir(parents=True, exist_ok=True)
        docs = self._load_docs()
        g = nx.MultiDiGraph()
        tag_to_claims: Dict[str, List[str]] = {}
        n_claims = 0
        n_papers = 0
        n_tags = 0

        for c in self._iter_jsonl(self.claims_jsonl):
            claim_id = self._safe_str(c.get("claim_id"))
            doc_id = self._safe_str(c.get("doc_id"))
            chunk_id = c.get("chunk_id")
            section = self._safe_str(c.get("section"))
            claim_text = self._safe_str(c.get("claim"))
            claim_type = self._safe_str(c.get("type"))
            evidence_span = self._safe_str(c.get("evidence_span"))
            confidence = float(c.get("confidence", 0.0))
            tags = c.get("tags", []) or []
            if not claim_id or not doc_id or chunk_id is None or not claim_text:
                continue

            paper_node = f"paper:{doc_id}"
            claim_node = claim_id
            if not g.has_node(paper_node):
                md = docs.get(doc_id, {})
                g.add_node(
                    paper_node,
                    type="paper",
                    doc_id=doc_id,
                    title=self._safe_str(md.get("title")) or doc_id,
                    year=md.get("year"),
                    source_type=md.get("source_type"),
                    source_url=md.get("source_url"),
                )
                n_papers += 1
            if not g.has_node(claim_node):
                g.add_node(
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
            g.add_edge(paper_node, claim_node, rel="has_claim")
            for t in tags:
                tag = self._safe_str(t).strip()
                if not tag:
                    continue
                tag_node = f"tag:{tag}"
                if not g.has_node(tag_node):
                    g.add_node(tag_node, type="tag", tag=tag)
                    n_tags += 1
                g.add_edge(claim_node, tag_node, rel="has_tag")
                tag_to_claims.setdefault(tag, []).append(claim_node)

        related_edges = 0
        if add_related_by_tag:
            for tag, claim_nodes in tag_to_claims.items():
                claim_nodes = list(dict.fromkeys(claim_nodes))[:max_claims_per_tag]
                for i in range(len(claim_nodes)):
                    a = claim_nodes[i]
                    for j in range(i + 1, len(claim_nodes)):
                        b = claim_nodes[j]
                        g.add_edge(a, b, rel="related_by_tag", tag=tag)
                        g.add_edge(b, a, rel="related_by_tag", tag=tag)
                        related_edges += 2

        nx.write_graphml(g, self.out_graphml)
        nodes_json = []
        for nid, attrs in g.nodes(data=True):
            nodes_json.append(
                {
                    "id": nid,
                    **{k: (v if isinstance(v, (int, float)) else self._safe_str(v)) for k, v in attrs.items()},
                }
            )
        edges_json = []
        for u, v, key, attrs in g.edges(keys=True, data=True):
            edges_json.append(
                {
                    "source": u,
                    "target": v,
                    "key": self._safe_str(key),
                    **{k: (a if isinstance(a, (int, float)) else self._safe_str(a)) for k, a in attrs.items()},
                }
            )
        self.out_json.write_text(
            json.dumps({"nodes": nodes_json, "edges": edges_json}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        stats = {
            "num_nodes": int(g.number_of_nodes()),
            "num_edges": int(g.number_of_edges()),
            "papers": int(n_papers),
            "claims": int(n_claims),
            "tags": int(n_tags),
            "related_by_tag_edges": int(related_edges),
            "graphml_path": str(self.out_graphml.as_posix()),
            "json_path": str(self.out_json.as_posix()),
        }
        self.out_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        msg = f"Wrote KG GraphML -> {self.out_graphml}"
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class ExportChunkEmbsStage:
    name = "export_chunk_embs"

    def __init__(self) -> None:
        data_dir = ROOT / "data"
        self.processed_dir = data_dir / "processed"
        self.index_dir = data_dir / "indexes"
        self.meta_path = self.index_dir / "chunk_meta.jsonl"
        self.chunks_path = self.processed_dir / "chunks_with_neighbors.jsonl"
        self.out_embs = self.index_dir / "chunk_embs.npy"
        self.out_rowids = self.index_dir / "chunk_row_ids.json"
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    class _TorchEmbedder:
        def __init__(self, model_name: str, device: Optional[str] = None, max_length: int = 512):
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.max_length = max_length

        @torch.no_grad()
        def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
            out = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Export embeddings"):
                batch = texts[i : i + batch_size]
                inputs = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                h = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.detach().cpu().numpy().astype("float32"))
            return np.vstack(out)

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing {self.meta_path}. Run embedding stage first.")
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Missing {self.chunks_path}. Run neighbors stage first.")
        meta_by_row: Dict[int, Dict[str, Any]] = {}
        for rec in self._iter_jsonl(self.meta_path):
            meta_by_row[int(rec["row_id"])] = rec
        chunks_by_id: Dict[int, Dict[str, Any]] = {}
        for rec in self._iter_jsonl(self.chunks_path):
            chunks_by_id[int(rec["chunk_id"])] = rec
        row_ids = sorted(meta_by_row.keys())
        texts: List[str] = []
        for rid in row_ids:
            cid = int(meta_by_row[rid]["chunk_id"])
            rec = chunks_by_id.get(cid)
            texts.append(str(rec.get("text", "")) if rec else "")
        emb = self._TorchEmbedder(self.model_name)
        embs = emb.encode(texts, batch_size=64)
        self.out_embs.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.out_embs, embs)
        self.out_rowids.write_text(json.dumps(row_ids, indent=2), encoding="utf-8")
        msg = f"Wrote embeddings: {self.out_embs} shape={embs.shape}"
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class EmbedChunksStage:
    name = "embed_chunks"

    def __init__(self) -> None:
        data_dir = ROOT / "data"
        self.processed_dir = data_dir / "processed"
        self.index_dir = data_dir / "indexes"
        self.chunks_with_neighbors = self.processed_dir / "chunks_with_neighbors.jsonl"
        self.chunks_fallback = self.processed_dir / "chunks.jsonl"
        self.faiss_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "chunk_meta.jsonl"
        self.info_path = self.index_dir / "index_info.json"

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    class _TorchEmbedder:
        def __init__(self, model_name: str, device: Optional[str] = None, max_length: int = 512):
            self.model_name = model_name
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.max_length = max_length
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

        @torch.no_grad()
        def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
            embs: List[np.ndarray] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embs.append(pooled.detach().cpu().numpy().astype("float32"))
            return np.vstack(embs)

    def _choose_input_path(self) -> Path:
        if self.chunks_with_neighbors.exists():
            return self.chunks_with_neighbors
        if self.chunks_fallback.exists():
            return self.chunks_fallback
        raise FileNotFoundError(
            f"Missing {self.chunks_with_neighbors} and {self.chunks_fallback}. "
            "Run chunking and neighbors stages first."
        )

    def _read_chunks(self, path: Path, max_chunks: int = 0) -> tuple[List[Dict[str, Any]], List[str]]:
        records: List[Dict[str, Any]] = []
        texts: List[str] = []
        for rec in self._iter_jsonl(path):
            if "chunk_id" not in rec or "doc_id" not in rec or "text" not in rec:
                raise ValueError(f"Bad chunk record missing required fields: {rec}")
            records.append(rec)
            texts.append(str(rec.get("text", "")))
            if max_chunks and len(records) >= max_chunks:
                break
        return records, texts

    @staticmethod
    def _build_faiss_ip_index(embs: np.ndarray) -> faiss.Index:
        dim = int(embs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        return index

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        model_name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        batch_size = int(os.environ.get("EMBED_BATCH_SIZE", "16"))
        max_length = int(os.environ.get("EMBED_MAX_LENGTH", "512"))
        max_chunks = int(os.environ.get("EMBED_MAX_CHUNKS", "0"))
        device = os.environ.get("EMBED_DEVICE") or None

        self.index_dir.mkdir(parents=True, exist_ok=True)
        in_path = self._choose_input_path()
        records, texts = self._read_chunks(in_path, max_chunks=max_chunks)
        if not records:
            raise ValueError(f"No chunks found in {in_path}")
        print(f"Input: {in_path}", flush=True)
        print(f"Chunks: {len(records)}", flush=True)
        print(f"Embedding model: {model_name}", flush=True)

        embedder = self._TorchEmbedder(model_name=model_name, device=device, max_length=max_length)
        embs_list: List[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i : i + batch_size]
            embs_list.append(embedder.encode(batch_texts, batch_size=len(batch_texts)))
        embs = np.vstack(embs_list).astype("float32")
        if embs.shape[0] != len(records):
            raise RuntimeError("Embedding count mismatch")

        index = self._build_faiss_ip_index(embs)
        faiss.write_index(index, str(self.faiss_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for row_id, rec in enumerate(records):
                meta = {
                    "row_id": row_id,
                    "chunk_id": rec["chunk_id"],
                    "doc_id": rec["doc_id"],
                    "section": rec.get("section"),
                    "start_char": rec.get("start_char"),
                    "end_char": rec.get("end_char"),
                    "prev_chunk_id": rec.get("prev_chunk_id"),
                    "next_chunk_id": rec.get("next_chunk_id"),
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        info = {
            "input_chunks_path": str(in_path.as_posix()),
            "faiss_index_path": str(self.faiss_path.as_posix()),
            "chunk_meta_path": str(self.meta_path.as_posix()),
            "embedding_model": model_name,
            "embedding_dim": int(embs.shape[1]),
            "num_chunks_indexed": int(embs.shape[0]),
            "faiss_index_type": "IndexFlatIP (cosine via L2-normalized embeddings)",
            "batch_size": int(batch_size),
            "max_length": int(max_length),
            "device": embedder.device,
        }
        self.info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

        msg = f"Wrote FAISS index -> {self.faiss_path}"
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class DetectContradictionsStage:
    name = "detect_contradictions"

    REL_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "a_claim_id": {"type": "string"},
                        "b_claim_id": {"type": "string"},
                        "relation": {"type": "string", "enum": ["entails", "contradiction", "neutral"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "short_justification": {"type": "string"},
                    },
                    "required": ["a_claim_id", "b_claim_id", "relation", "confidence", "short_justification"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    @dataclass
    class _Claim:
        claim_id: str
        doc_id: str
        chunk_id: int
        section: str
        text: str
        confidence: float
        tags: List[str]

    class _TorchEmbedder:
        def __init__(self, model_name: str, device: Optional[str] = None, max_length: int = 256):
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.max_length = max_length
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

        @torch.no_grad()
        def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
            out = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                h = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.detach().cpu().numpy().astype("float32"))
            return np.vstack(out)

    def __init__(self) -> None:
        data_dir = ROOT / "data"
        processed_dir = data_dir / "processed"
        self.claims_jsonl = processed_dir / "claims.jsonl"
        self.out_relations_jsonl = processed_dir / "relations.jsonl"
        self.cache_dir = processed_dir / "cache" / "relations_by_pair"

    def _iter_jsonl(self, path: Path) -> Iterable[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def _pair_key(a_id: str, b_id: str) -> str:
        x, y = sorted([a_id, b_id])
        h = hashlib.sha256((x + "||" + y).encode("utf-8")).hexdigest()[:24]
        return f"{h}__{x.replace(':', '_')}__{y.replace(':', '_')}"[:220]

    def _cache_path_for_pair(self, a_id: str, b_id: str) -> Path:
        return self.cache_dir / f"{self._pair_key(a_id, b_id)}.json"

    def _load_claims(self) -> List[_Claim]:
        claims: List[DetectContradictionsStage._Claim] = []
        for r in self._iter_jsonl(self.claims_jsonl):
            claims.append(
                self._Claim(
                    claim_id=str(r["claim_id"]),
                    doc_id=str(r["doc_id"]),
                    chunk_id=int(r["chunk_id"]),
                    section=str(r.get("section", "unknown")),
                    text=str(r["claim"]),
                    confidence=float(r.get("confidence", 0.0)),
                    tags=list(r.get("tags", []) or []),
                )
            )
        if not claims:
            raise ValueError(f"No claims found in {self.claims_jsonl}. Run claim extraction first.")
        return claims

    def _generate_candidate_pairs_numpy(
        self,
        claims: List[_Claim],
        *,
        embed_model: str,
        device: Optional[str],
        top_k: int,
        min_sim: float,
        exclude_same_doc: bool,
        embed_batch_size: int,
    ) -> List[tuple[int, int, float]]:
        texts = [c.text for c in claims]
        embedder = self._TorchEmbedder(embed_model, device=device, max_length=256)
        embs = embedder.encode(texts, batch_size=embed_batch_size)
        sim = embs @ embs.T
        n = sim.shape[0]
        pairs: Dict[tuple[int, int], float] = {}
        for i in range(n):
            row = sim[i].copy()
            row[i] = -1.0
            k = min(top_k, n - 1)
            if k <= 0:
                continue
            idx = np.argpartition(-row, kth=k - 1)[:k]
            idx = idx[np.argsort(-row[idx])]
            for j in idx:
                s = float(row[j])
                if s < min_sim:
                    break
                if exclude_same_doc and claims[i].doc_id == claims[j].doc_id:
                    continue
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                if a == b:
                    continue
                if (a, b) not in pairs or s > pairs[(a, b)]:
                    pairs[(a, b)] = s
        out = [(a, b, s) for (a, b), s in pairs.items()]
        out.sort(key=lambda x: x[2], reverse=True)
        return out

    async def _call_openai_relations_async(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        batch: List[Dict[str, Any]],
        temperature: float,
    ) -> Dict[str, Any]:
        system_msg = (
            "You are an expert at natural-language inference over AI safety research claims. "
            "Given two claims, label whether A entails B, contradicts B, or is neutral. "
            "Use ONLY the claim texts provided. Be conservative; if unsure, return neutral with lower confidence. "
            "Return JSON that matches the schema exactly."
        )
        user_payload = json.dumps(
            {
                "instructions": {
                    "labels": {
                        "entails": "A logically implies/supports B (same meaning or stronger).",
                        "contradiction": "A conflicts with B (cannot both be true as stated).",
                        "neutral": "Not enough to say entailment or contradiction.",
                    },
                    "notes": [
                        "Treat differences in scope/conditions carefully: if one claim is conditional and the other is general, often neutral.",
                        "If claims address different settings/metrics, neutral.",
                        "Short justification: one sentence.",
                    ],
                },
                "pairs": batch,
            },
            ensure_ascii=False,
        )
        resp = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_payload},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "claim_relation_classification",
                    "strict": True,
                    "schema": self.REL_SCHEMA,
                }
            },
            temperature=temperature,
        )
        raw = resp.output_text
        if not raw:
            raise RuntimeError("Empty output_text from model response.")
        return json.loads(raw)

    async def _classify_batch(
        self,
        *,
        batch_idx: int,
        batch_pairs: List[tuple[int, int, float]],
        claims: List[_Claim],
        client: AsyncOpenAI,
        model: str,
        temperature: float,
        semaphore: asyncio.Semaphore,
        write_lock: asyncio.Lock,
        counters: Dict[str, int],
        progress: Any,
    ) -> None:
        batch_payload = [
            {
                "a_claim_id": claims[i].claim_id,
                "b_claim_id": claims[j].claim_id,
                "a_text": claims[i].text,
                "b_text": claims[j].text,
                "a_doc_id": claims[i].doc_id,
                "b_doc_id": claims[j].doc_id,
                "similarity": sim,
            }
            for i, j, sim in batch_pairs
        ]
        async with semaphore:
            try:
                data = await self._call_openai_relations_async(
                    client=client,
                    model=model,
                    batch=batch_payload,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"\n[ERROR] batch {batch_idx} failed: {e}", flush=True)
                traceback.print_exc()
                progress.update(1)
                return

        items = data.get("items", [])
        if not isinstance(items, list):
            print(f"\n[ERROR] batch {batch_idx}: items is not a list, got {type(items)}", flush=True)
            progress.update(1)
            return

        out_rels: List[Dict[str, Any]] = []
        for it in items:
            a_id = it["a_claim_id"]
            b_id = it["b_claim_id"]
            rec = {
                "src_claim_id": a_id,
                "dst_claim_id": b_id,
                "relation": it["relation"],
                "confidence": float(it["confidence"]),
                "method": "openai_nli",
                "short_justification": it["short_justification"],
            }
            out_rels.append(rec)
            self._cache_path_for_pair(a_id, b_id).write_text(
                json.dumps({"result": rec}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        async with write_lock:
            self.out_relations_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.out_relations_jsonl.open("a", encoding="utf-8") as f:
                for r in out_rels:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            counters["written"] += len(out_rels)
            counters["calls"] += 1
        progress.update(1)

    async def _run_async(self, *, model: str, embed_model: str, device: Optional[str], top_k: int, min_sim: float,
                         exclude_same_doc: bool, max_pairs: int, batch_size: int, concurrency: int,
                         embed_batch_size: int, temperature: float) -> Dict[str, Any]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        claims = self._load_claims()
        print(f"Loaded claims: {len(claims)} from {self.claims_jsonl}", flush=True)
        print("Generating candidate pairs with local embeddings + NumPy cosine sim...", flush=True)
        pairs = self._generate_candidate_pairs_numpy(
            claims,
            embed_model=embed_model,
            device=device,
            top_k=top_k,
            min_sim=min_sim,
            exclude_same_doc=exclude_same_doc,
            embed_batch_size=embed_batch_size,
        )
        if max_pairs and len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]
        to_classify: List[tuple[int, int, float]] = []
        skipped_cached = 0
        for i, j, sim in pairs:
            if self._cache_path_for_pair(claims[i].claim_id, claims[j].claim_id).exists():
                skipped_cached += 1
            else:
                to_classify.append((i, j, sim))
        print(f"Candidate pairs total   : {len(pairs)}", flush=True)
        print(f"Already cached (skipped): {skipped_cached}", flush=True)
        print(f"Pairs to classify now   : {len(to_classify)}", flush=True)
        if not to_classify:
            return {"calls": 0, "written": 0, "skipped_cached": skipped_cached, "pairs_total": len(pairs)}

        batches = [to_classify[s : s + batch_size] for s in range(0, len(to_classify), batch_size)]
        print(f"OpenAI model      : {model}", flush=True)
        print(f"Batch size        : {batch_size} pairs/call", flush=True)
        print(f"Concurrent calls  : {concurrency}", flush=True)
        print(f"Total API calls   : {len(batches)}", flush=True)

        client = AsyncOpenAI()
        semaphore = asyncio.Semaphore(max(1, concurrency))
        write_lock = asyncio.Lock()
        counters: Dict[str, int] = {"calls": 0, "written": 0}
        with atqdm(total=len(batches), desc="OpenAI relation batches") as progress:
            tasks = [
                self._classify_batch(
                    batch_idx=idx,
                    batch_pairs=bp,
                    claims=claims,
                    client=client,
                    model=model,
                    temperature=temperature,
                    semaphore=semaphore,
                    write_lock=write_lock,
                    counters=counters,
                    progress=progress,
                )
                for idx, bp in enumerate(batches)
            ]
            await asyncio.gather(*tasks)
        return {
            "calls": counters["calls"],
            "written": counters["written"],
            "skipped_cached": skipped_cached,
            "pairs_total": len(pairs),
        }

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        model = os.environ.get("RELATIONS_MODEL", "gpt-4o-mini")
        embed_model = os.environ.get("CLAIM_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        device = os.environ.get("REL_DEVICE") or None
        top_k = int(os.environ.get("REL_TOP_K", "25"))
        min_sim = float(os.environ.get("REL_MIN_SIM", "0.35"))
        exclude_same_doc = os.environ.get("REL_EXCLUDE_SAME_DOC", "1").strip().lower() not in {"0", "false", "no"}
        max_pairs = int(os.environ.get("REL_MAX_PAIRS", "0"))
        batch_size = int(os.environ.get("REL_BATCH_SIZE", "16"))
        concurrency = int(os.environ.get("REL_CONCURRENCY", "8"))
        embed_batch_size = int(os.environ.get("REL_EMBED_BATCH_SIZE", "64"))
        temperature = float(os.environ.get("REL_TEMPERATURE", "0.0"))
        stats = asyncio.run(
            self._run_async(
                model=model,
                embed_model=embed_model,
                device=device,
                top_k=top_k,
                min_sim=min_sim,
                exclude_same_doc=exclude_same_doc,
                max_pairs=max_pairs,
                batch_size=batch_size,
                concurrency=concurrency,
                embed_batch_size=embed_batch_size,
                temperature=temperature,
            )
        )
        msg = (
            f"Done. OpenAI calls: {stats['calls']} | Relations written: {stats['written']} | "
            f"Saved to: {self.out_relations_jsonl}"
        )
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


class MergeRelationsIntoKGStage:
    name = "merge_relations_into_kg"

    def __init__(self) -> None:
        data_dir = ROOT / "data"
        processed_dir = data_dir / "processed"
        self.kg_dir = processed_dir / "kg"
        self.in_graphml = self.kg_dir / "graph.graphml"
        self.relations_jsonl = processed_dir / "relations.jsonl"
        self.out_graphml = self.kg_dir / "graph_with_relations.graphml"
        self.out_json = self.kg_dir / "graph_with_relations.json"
        self.out_stats = self.kg_dir / "merge_stats.json"

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def _safe_str(x: Any) -> str:
        return "" if x is None else str(x)

    def run(
        self,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        stage_index: int = 0,
        stage_total: int = 0,
        stage_results: Optional[List[Dict[str, object]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> StageResult:
        started = time.time()
        keep_neutral = os.environ.get("MERGE_KEEP_NEUTRAL", "0").strip().lower() in {"1", "true", "yes"}
        min_confidence = float(os.environ.get("MERGE_MIN_CONFIDENCE", "0.70"))
        if not self.in_graphml.exists():
            raise FileNotFoundError(f"Missing {self.in_graphml}. Run build KG stage first.")
        if not self.relations_jsonl.exists():
            raise FileNotFoundError(f"Missing {self.relations_jsonl}. Run relation detection stage first.")

        g = nx.read_graphml(self.in_graphml)
        if not isinstance(g, nx.MultiDiGraph):
            g = nx.MultiDiGraph(g)
        total_rel = 0
        added = 0
        skipped_low_conf = 0
        skipped_missing_nodes = 0
        skipped_neutral = 0
        for r in self._iter_jsonl(self.relations_jsonl):
            total_rel += 1
            src = self._safe_str(r.get("src_claim_id"))
            dst = self._safe_str(r.get("dst_claim_id"))
            rel = self._safe_str(r.get("relation"))
            conf = float(r.get("confidence", 0.0))
            if not src or not dst or rel not in {"entails", "contradiction", "neutral"}:
                continue
            if conf < min_confidence:
                skipped_low_conf += 1
                continue
            if rel == "neutral" and not keep_neutral:
                skipped_neutral += 1
                continue
            if not g.has_node(src) or not g.has_node(dst):
                skipped_missing_nodes += 1
                continue
            g.add_edge(
                src,
                dst,
                rel=rel,
                confidence=conf,
                method=self._safe_str(r.get("method", "openai_nli")),
                short_justification=self._safe_str(r.get("short_justification", "")),
            )
            added += 1

        nx.write_graphml(g, self.out_graphml)
        nodes_json = []
        for nid, attrs in g.nodes(data=True):
            nodes_json.append(
                {
                    "id": nid,
                    **{k: (v if isinstance(v, (int, float)) else self._safe_str(v)) for k, v in attrs.items()},
                }
            )
        edges_json = []
        for u, v, key, attrs in g.edges(keys=True, data=True):
            edges_json.append(
                {
                    "source": u,
                    "target": v,
                    "key": self._safe_str(key),
                    **{k: (a if isinstance(a, (int, float)) else self._safe_str(a)) for k, a in attrs.items()},
                }
            )
        self.out_json.write_text(
            json.dumps({"nodes": nodes_json, "edges": edges_json}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        stats = {
            "input_graphml": str(self.in_graphml.as_posix()),
            "relations_jsonl": str(self.relations_jsonl.as_posix()),
            "output_graphml": str(self.out_graphml.as_posix()),
            "output_json": str(self.out_json.as_posix()),
            "total_relations_seen": int(total_rel),
            "edges_added": int(added),
            "skipped_low_confidence": int(skipped_low_conf),
            "skipped_missing_nodes": int(skipped_missing_nodes),
            "skipped_neutral": int(skipped_neutral),
            "min_confidence": float(min_confidence),
            "keep_neutral": bool(keep_neutral),
            "final_num_nodes": int(g.number_of_nodes()),
            "final_num_edges": int(g.number_of_edges()),
        }
        self.out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        msg = f"Wrote merged KG -> {self.out_graphml}"
        print(msg, flush=True)
        return StageResult(
            module=self.name,
            ok=True,
            return_code=0,
            elapsed_seconds=round(time.time() - started, 2),
            output_tail=msg,
        )


def build_stage(stage_name: str) -> object:
    """
    Resolve a stage identifier to a stage object.
    Supports both new stage IDs and legacy module names.
    """
    mapping = {
        "dedupe_manifest": DedupeManifestStage,
        "collect_manifest": CollectManifestStage,
        "download_sources": DownloadSourcesStage,
        "pdf_to_text": PdfToTextStage,
        "apply_neighbors": ApplyNeighborsStage,
        "html_to_text": HtmlToTextStage,
        "section_chunk": SectionChunkStage,
        "extract_claims": ExtractClaimsStage,
        "build_kg": BuildKGStage,
        "export_chunk_embs": ExportChunkEmbsStage,
        "embed_chunks": EmbedChunksStage,
        "detect_contradictions": DetectContradictionsStage,
        "merge_relations_into_kg": MergeRelationsIntoKGStage,
        "src.ingest.00c_dedupe_manifest": DedupeManifestStage,
        "src.ingest.00_collect_papers_manifest": CollectManifestStage,
        "src.ingest.00_download_sources": DownloadSourcesStage,
        "src.ingest.01_pdf_to_text": PdfToTextStage,
        "src.ingest.03_apply_neighbors": ApplyNeighborsStage,
        "src.ingest.01b_html_to_text_from_cache": HtmlToTextStage,
        "src.ingest.02_section_chunk": SectionChunkStage,
        "src.ingest.04_extract_claims_openai": ExtractClaimsStage,
        "src.ingest.05_build_kg": BuildKGStage,
        "src.ingest.03b_export_chunk_embs": ExportChunkEmbsStage,
        "src.ingest.03_embed_chunks": EmbedChunksStage,
        "src.ingest.06_detect_contradictions_openai": DetectContradictionsStage,
        "src.ingest.07_merge_relations_into_kg": MergeRelationsIntoKGStage,
    }
    cls = mapping.get(stage_name)
    if cls is not None:
        return cls()
    return ModuleStage(stage_name)

