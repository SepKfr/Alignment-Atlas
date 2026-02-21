from __future__ import annotations

import audioop
import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
from openai import OpenAI

from src.app.chat_agent import ChatState, chat_turn
from src.app.ingest_guardrails import evaluate_paper_candidate
from src.app.storage import build_storage_backend_from_env
from src.ingest.pipeline import IngestPipeline
from src.ingest.stages import IncrementalStageOps, IngestPaths
from src.retrieval.citations import CitationResolver
from src.retrieval.retriever import AlignmentAtlasRetriever

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PAPERS_JSONL = DATA_DIR / "papers.jsonl"
DOCS_JSONL = PROCESSED_DIR / "docs.jsonl"
TEXT_DIR = PROCESSED_DIR / "text"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
CHUNKS_WITH_NEIGHBORS = PROCESSED_DIR / "chunks_with_neighbors.jsonl"
NEIGHBORS_DIR = PROCESSED_DIR / "neighbors"
INDEX_DIR = DATA_DIR / "indexes"
META_PATH = INDEX_DIR / "chunk_meta.jsonl"
EMBS_NPY = INDEX_DIR / "chunk_embs.npy"
ROWIDS_JSON = INDEX_DIR / "chunk_row_ids.json"
KG_DIR = PROCESSED_DIR / "kg"
GRAPH_GRAPHML = KG_DIR / "graph.graphml"
GRAPH_JSON = KG_DIR / "graph.json"
GRAPH_WITH_REL_GRAPHML = KG_DIR / "graph_with_relations.graphml"
GRAPH_WITH_REL_JSON = KG_DIR / "graph_with_relations.json"
DEFAULT_TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "gpt-4o-transcribe")
INGEST_GUARDRAILS_ENABLED = os.environ.get("INGEST_GUARDRAILS_ENABLED", "1").strip().lower() not in {"0", "false", "no"}


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return s[:100] or f"paper_{int(time.time())}"


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
    # Drop common tracking params but preserve semantically meaningful ones.
    keep = []
    for k, v in parse_qsl(p.query or "", keep_blank_values=False):
        kl = (k or "").lower()
        if kl.startswith("utm_") or kl in {"ref", "source", "fbclid", "gclid"}:
            continue
        keep.append((k, v))
    query = urlencode(keep, doseq=True)
    canon = urlunparse((scheme, netloc, path, "", query, ""))
    return canon


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


def _normalize_title(title: str, url: str) -> tuple[str, str]:
    raw = re.sub(r"\s+", " ", (title or "").strip())
    if not raw:
        raw = _title_from_url(url)
    # Keep a readable display title, but also store a lowercase normalized form for comparisons.
    display = raw
    normalized = raw.lower().strip()
    return display, normalized


def _is_placeholder_title(title: str) -> bool:
    t = re.sub(r"\s+", " ", (title or "").strip()).lower()
    if not t:
        return True
    return bool(re.match(r"^arxiv\s+\d{4}\.\d{4,5}(?:v\d+)?$", t))


def _detect_source_type(url: str) -> str:
    u = (url or "").lower().strip()
    if u.endswith(".pdf") or "/pdf/" in u or "arxiv.org/pdf/" in u:
        return "pdf"
    return "html"


def _is_valid_source_url(url: str) -> bool:
    u = (url or "").strip()
    if not u:
        return False
    try:
        p = urlparse(u)
    except Exception:
        return False
    return p.scheme in {"http", "https"} and bool(p.netloc)


def _wav_duration_seconds(audio_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 0
            if rate <= 0:
                return 0.0
            return float(frames) / float(rate)
    except Exception:
        return 0.0


def _is_mostly_silence_wav(
    audio_bytes: bytes,
    *,
    silence_rms_threshold: int = 230,
    voiced_ratio_threshold: float = 0.08,
    window_ms: int = 30,
) -> bool:
    """
    Heuristic silence detector for WAV clips.
    Returns True if voiced windows are below threshold ratio.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            framerate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if not raw or framerate <= 0 or sample_width <= 0:
            return True

        frame_bytes = max(1, int(framerate * (window_ms / 1000.0)) * sample_width * max(1, channels))
        total_windows = 0
        voiced_windows = 0
        for i in range(0, len(raw), frame_bytes):
            chunk = raw[i : i + frame_bytes]
            if not chunk:
                continue
            total_windows += 1
            rms = audioop.rms(chunk, sample_width)
            if rms >= silence_rms_threshold:
                voiced_windows += 1
        if total_windows == 0:
            return True
        voiced_ratio = voiced_windows / float(total_windows)
        return voiced_ratio < voiced_ratio_threshold
    except Exception:
        # If parsing fails, do not block transcription.
        return False


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _upsert_paper(
    title: str, url: str, source_type: str, year: Optional[int] = None
) -> tuple[Dict[str, Any], bool]:
    rows = _iter_jsonl(PAPERS_JSONL)
    canonical_url = _canonicalize_source_url(url)
    display_title, normalized_title = _normalize_title(title, canonical_url or url)

    existing: Optional[Dict[str, Any]] = None
    for r in rows:
        r_canon = str(r.get("canonical_source_url", "") or "").strip()
        if not r_canon and r.get("source_url"):
            r_canon = _canonicalize_source_url(str(r.get("source_url")))
        if canonical_url and r_canon == canonical_url:
            existing = r
            break

    if existing is not None:
        doc_id = str(existing.get("doc_id", "")).strip() or _slugify(display_title)
        existed = True
    else:
        # Stable doc_id from canonical URL avoids duplicates from title capitalization/variants.
        basis = canonical_url or (url or "").strip()
        if basis:
            digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
            doc_id = f"doc_{digest}"
        else:
            doc_id = _slugify(display_title)
        existed = any(str(r.get("doc_id", "")) == doc_id for r in rows)

    rec = {
        "doc_id": doc_id,
        "title": display_title or doc_id,
        "title_normalized": normalized_title,
        "year": year,
        "source_type": source_type,
        "source_url": (url or "").strip(),
        "canonical_source_url": canonical_url or (url or "").strip(),
    }
    def _row_canonical_source(r: Dict[str, Any]) -> str:
        rc = str(r.get("canonical_source_url", "") or "").strip()
        if rc:
            return rc
        if r.get("source_url"):
            return _canonicalize_source_url(str(r.get("source_url")))
        return ""

    # Drop duplicate records that point to the same canonical URL but different doc_ids.
    filtered_rows: List[Dict[str, Any]] = []
    for r in rows:
        rid = str(r.get("doc_id", "")).strip()
        if canonical_url and rid and rid != doc_id and _row_canonical_source(r) == canonical_url:
            continue
        filtered_rows.append(r)

    by_id = {str(r.get("doc_id")): r for r in filtered_rows if r.get("doc_id")}
    by_id[doc_id] = rec
    merged = list(by_id.values())
    merged.sort(key=lambda x: str(x.get("doc_id", "")))
    _write_jsonl(PAPERS_JSONL, merged)
    return rec, existed


@dataclass
class IngestRunResult:
    ok: bool
    doc: Dict[str, Any]
    stage_results: List[Dict[str, Any]]
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "doc": self.doc,
            "stage_results": self.stage_results,
            "error": self.error,
        }


class AtlasService:
    def __init__(self) -> None:
        self.state = ChatState()
        self.client = OpenAI()
        self._chat_lock = threading.Lock()
        self._jobs_lock = threading.Lock()
        self._storage_lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self.storage = build_storage_backend_from_env()
        self.storage_last_sync_down: Optional[Dict[str, Any]] = None
        self.storage_last_sync_up: Optional[Dict[str, Any]] = None
        self.retriever: Optional[AlignmentAtlasRetriever] = None
        self.resolver = CitationResolver()
        self.retriever_init_error: Optional[str] = None
        self._sync_from_storage_on_startup()
        self._refresh_runtime_artifacts()
        self._incremental_ops = IncrementalStageOps(
            IngestPaths(
                docs_jsonl=DOCS_JSONL,
                text_dir=TEXT_DIR,
                chunks_jsonl=CHUNKS_JSONL,
                chunks_with_neighbors=CHUNKS_WITH_NEIGHBORS,
                neighbors_dir=NEIGHBORS_DIR,
                raw_pdfs_dir=DATA_DIR / "raw_pdfs",
                raw_html_dir=DATA_DIR / "raw_html",
            )
        )

    def runtime_info(self) -> Dict[str, Any]:
        with self._jobs_lock:
            active_jobs = len([j for j in self._jobs.values() if j.get("status") == "running"])
        retriever_info: Dict[str, Any]
        if self.retriever is not None:
            retriever_info = self.retriever.runtime_info()
        else:
            retriever_info = {
                "ready": False,
                "error": self.retriever_init_error,
            }
        return {
            "retriever": retriever_info,
            "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            "papers_manifest_path": str(PAPERS_JSONL.as_posix()),
            "active_ingest_jobs": active_jobs,
            "storage": self.storage.describe(),
            "storage_last_sync_down": self.storage_last_sync_down,
            "storage_last_sync_up": self.storage_last_sync_up,
        }

    def reset_chat(self) -> None:
        with self._chat_lock:
            self.state = ChatState()

    def transcribe_audio(
        self,
        *,
        audio_bytes: bytes,
        filename: str = "voice.wav",
        model: Optional[str] = None,
        skip_if_silence: bool = True,
    ) -> Dict[str, Any]:
        if not audio_bytes:
            return {
                "ok": False,
                "text": "",
                "error": "Empty audio payload.",
                "skipped_silence": False,
                "duration_seconds": 0.0,
                "model": model or DEFAULT_TRANSCRIBE_MODEL,
            }

        duration_sec = _wav_duration_seconds(audio_bytes)
        if skip_if_silence and filename.lower().endswith(".wav"):
            if _is_mostly_silence_wav(audio_bytes):
                return {
                    "ok": True,
                    "text": "",
                    "error": None,
                    "skipped_silence": True,
                    "duration_seconds": duration_sec,
                    "model": model or DEFAULT_TRANSCRIBE_MODEL,
                }

        use_model = (model or DEFAULT_TRANSCRIBE_MODEL).strip() or DEFAULT_TRANSCRIBE_MODEL
        try:
            buf = io.BytesIO(audio_bytes)
            buf.name = filename
            resp = self.client.audio.transcriptions.create(
                model=use_model,
                file=buf,
            )
            text = ""
            if isinstance(resp, dict):
                text = str(resp.get("text", "")).strip()
            else:
                text = str(getattr(resp, "text", "")).strip()
            return {
                "ok": True,
                "text": text,
                "error": None,
                "skipped_silence": False,
                "duration_seconds": duration_sec,
                "model": use_model,
            }
        except Exception as e:
            return {
                "ok": False,
                "text": "",
                "error": f"{type(e).__name__}: {e}",
                "skipped_silence": False,
                "duration_seconds": duration_sec,
                "model": use_model,
            }

    def chat(
        self,
        user_message: str,
        *,
        steer: float = 0.0,
        steering_mode: Optional[str] = None,
        top_k_chunks: int = 12,
        neighbor_radius: int = 2,
        include_suggestions: bool = True,
        allow_external_fallback: bool = True,
        stream_handler: Optional[Callable[[str], None]] = None,
        stage_handler: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        if self.retriever is None:
            raise RuntimeError(
                "Retriever is not ready. Ingest at least one paper or sync data from storage first. "
                f"Init error: {self.retriever_init_error}"
            )
        with self._chat_lock:
            self.state, payload = chat_turn(
                self.state,
                user_message,
                steer=steer,
                steering_mode=steering_mode,
                top_k_chunks=top_k_chunks,
                neighbor_radius=neighbor_radius,
                include_suggestions=include_suggestions,
                allow_external_fallback=allow_external_fallback,
                stream_handler=stream_handler,
                stage_handler=stage_handler,
                client=self.client,
                retriever=self.retriever,
            )
            return payload

    def evaluate_ingest_candidate(
        self,
        *,
        title: str,
        source_url: str,
        source_type: str = "auto",
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        st = source_type if source_type in {"pdf", "html"} else _detect_source_type(source_url)
        return evaluate_paper_candidate(
            self.client,
            title=title,
            source_url=source_url,
            source_type=st,
            year=year,
        )

    def ingest_source(
        self,
        *,
        title: str,
        source_url: str,
        source_type: str = "auto",
        year: Optional[int] = None,
        run_relations: bool = True,
        incremental: bool = True,
        allow_review_override: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        claims_batch_size: Optional[int] = None,
        claims_max_concurrency: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not _is_valid_source_url(source_url):
            return {
                "ok": False,
                "doc": {
                    "doc_id": "",
                    "title": (title or "").strip(),
                    "source_url": source_url,
                    "source_type": source_type,
                    "year": year,
                },
                "stage_results": [],
                "error": "Invalid source URL. Expected a valid http(s) URL.",
            }
        guardrail: Optional[Dict[str, Any]] = None
        effective_title = (title or "").strip()
        effective_year = year
        if INGEST_GUARDRAILS_ENABLED:
            guardrail = evaluate_paper_candidate(
                self.client,
                title=title,
                source_url=source_url,
                source_type=source_type,
                year=year,
            )
            decision = (
                str(((guardrail or {}).get("decision") or {}).get("decision", "")).strip().lower()
                if isinstance(guardrail, dict)
                else ""
            )
            signals = ((guardrail or {}).get("signals") or {}) if isinstance(guardrail, dict) else {}
            ss = (signals.get("semantic_scholar") or {}) if isinstance(signals, dict) else {}
            ss_title = str(ss.get("title") or "").strip()
            ss_year = ss.get("year")
            if _is_placeholder_title(effective_title) and ss_title:
                effective_title = ss_title
            if effective_year is None and ss_year is not None:
                try:
                    effective_year = int(ss_year)
                except Exception:
                    pass
            review_overridden = decision == "review" and bool(allow_review_override)
            if decision != "allow" and not review_overridden:
                return {
                    "ok": False,
                    "doc": {
                        "doc_id": "",
                        "title": (title or "").strip(),
                        "source_url": source_url,
                        "source_type": source_type,
                        "year": year,
                    },
                    "stage_results": [],
                    "error": (
                        "Blocked by ingest guardrails "
                        f"(decision={decision or 'unknown'}; only allow, or review with explicit override, can proceed)."
                    ),
                    "guardrail": guardrail,
                }

        st = source_type if source_type in {"pdf", "html"} else _detect_source_type(source_url)
        doc, existed = _upsert_paper(title=effective_title, url=source_url, source_type=st, year=effective_year)
        # Always use one-doc incremental mode when requested.
        # Existing docs will rely on step-level skip checks to avoid rework.
        use_incremental = bool(incremental)
        claims_env: Dict[str, str] = {}
        if claims_batch_size is not None and int(claims_batch_size) > 0:
            claims_env["CLAIMS_BATCH_SIZE"] = str(int(claims_batch_size))
        if claims_max_concurrency is not None and int(claims_max_concurrency) > 0:
            claims_env["CLAIMS_MAX_CONCURRENCY"] = str(int(claims_max_concurrency))
        if use_incremental:
            result = self._run_incremental_ingest_pipeline(
                run_relations=run_relations,
                doc=doc,
                progress_callback=progress_callback,
                claims_env=claims_env,
            )
        else:
            result = self._run_ingest_pipeline(
                run_relations=run_relations,
                doc=doc,
                progress_callback=progress_callback,
                claims_env=claims_env,
            )
        if result.ok:
            out = result.as_dict()
            out["ingest_mode"] = "incremental" if use_incremental else "full_rebuild"
            if guardrail is not None:
                out["guardrail"] = guardrail
                decision = str(((guardrail or {}).get("decision") or {}).get("decision", "")).strip().lower()
                if decision == "review" and allow_review_override:
                    out["guardrail_override"] = {
                        "applied": True,
                        "decision": "review",
                        "reason": "Manually approved by reviewer.",
                    }
            # If incremental run had no new chunks/claims, we skip expensive retriever refresh.
            # This avoids a "looks stuck" phase where embeddings reload despite no data changes.
            skip_refresh = False
            if use_incremental:
                mods = {str(s.get("module", "")) for s in out.get("stage_results", []) or []}
                skip_refresh = "incremental.skip_graph_updates" in mods
            if not skip_refresh:
                self._refresh_runtime_artifacts()
            else:
                out["runtime_refresh_skipped"] = True
            if self.retriever is None:
                out["warning"] = (
                    "Ingest completed, but retriever refresh failed. "
                    f"Likely missing graph merge output: {self.retriever_init_error}"
                )
            if existed and incremental:
                out["note"] = (
                    "Document slug already existed; one-doc incremental mode was used and completed steps were skipped."
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "current_stage": "finalizing.sync_storage",
                        "stage_index": int(out.get("stage_total", 0) or 0),
                        "stage_total": int(out.get("stage_total", 0) or 0),
                        "stage_results": out.get("stage_results", []) or [],
                        "detail_message": "[finalize] syncing storage backend",
                    }
                )

            sync_timeout_sec = float(os.environ.get("INGEST_SYNC_TIMEOUT_SECONDS", "20"))
            sync_box: Dict[str, Any] = {"done": False, "result": None}

            def _sync() -> None:
                try:
                    sync_box["result"] = self._sync_to_storage(
                        commit_message=f"Ingest update for {doc.get('doc_id')}"
                    )
                finally:
                    sync_box["done"] = True

            sync_thread = threading.Thread(target=_sync, daemon=True, name="ingest-sync-up")
            sync_thread.start()
            sync_thread.join(timeout=sync_timeout_sec)
            if sync_box["done"]:
                sync_up = sync_box["result"]
            else:
                sync_up = {
                    "ok": False,
                    "backend": "unknown",
                    "action": "sync_up",
                    "detail": (
                        f"Timed out after {sync_timeout_sec:.0f}s; "
                        "skipped waiting for storage sync."
                    ),
                    "elapsed_seconds": sync_timeout_sec,
                }
                out["warning"] = (
                    (out.get("warning", "") + " ").strip()
                    + "Storage sync timed out; ingest data is still present locally."
                ).strip()
            out["storage_sync_up"] = sync_up
            return out
        out_fail = result.as_dict()
        if guardrail is not None:
            out_fail["guardrail"] = guardrail
        return out_fail

    def start_ingest_job(
        self,
        *,
        title: str,
        source_url: str,
        source_type: str = "auto",
        year: Optional[int] = None,
        run_relations: bool = True,
        incremental: bool = True,
        allow_review_override: bool = False,
        claims_batch_size: Optional[int] = None,
        claims_max_concurrency: Optional[int] = None,
    ) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())
        now = time.time()
        record = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "progress": {
                "current_stage": "queued",
                "stage_index": 0,
                "stage_total": 0,
                "stage_results": [],
            },
            "request": {
                "title": title,
                "source_url": source_url,
                "source_type": source_type,
                "year": year,
                "run_relations": run_relations,
                "incremental": incremental,
                "allow_review_override": allow_review_override,
                "claims_batch_size": claims_batch_size,
                "claims_max_concurrency": claims_max_concurrency,
            },
            "result": None,
            "error": None,
        }
        with self._jobs_lock:
            self._jobs[job_id] = record

        def _runner() -> None:
            self._update_job(job_id, status="running")
            def _progress(ev: Dict[str, Any]) -> None:
                self._update_job(job_id, progress=ev)
            try:
                result = self.ingest_source(
                    title=title,
                    source_url=source_url,
                    source_type=source_type,
                    year=year,
                    run_relations=run_relations,
                    incremental=incremental,
                    allow_review_override=allow_review_override,
                    progress_callback=_progress,
                    claims_batch_size=claims_batch_size,
                    claims_max_concurrency=claims_max_concurrency,
                )
                self._update_job(job_id, status="completed", result=result)
            except Exception as e:
                self._update_job(
                    job_id,
                    status="failed",
                    error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                )

        t = threading.Thread(target=_runner, daemon=True, name=f"ingest-job-{job_id[:8]}")
        t.start()
        return self.get_ingest_job(job_id) or {"job_id": job_id, "status": "queued"}

    def _update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        progress: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._jobs_lock:
            if job_id not in self._jobs:
                return
            rec = self._jobs[job_id]
            if status is not None:
                rec["status"] = status
            if result is not None:
                rec["result"] = result
            if error is not None:
                rec["error"] = error
            if progress is not None:
                rec_progress = rec.setdefault("progress", {})
                for k, v in progress.items():
                    rec_progress[k] = v
                rec["progress_event_count"] = int(rec.get("progress_event_count", 0) or 0) + 1
            rec["updated_at"] = time.time()

    def get_ingest_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._jobs_lock:
            rec = self._jobs.get(job_id)
            return dict(rec) if rec else None

    def list_ingest_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._jobs_lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda x: float(x.get("updated_at", 0.0)), reverse=True)
        return [dict(j) for j in jobs[: max(1, limit)]]

    def _sync_from_storage_on_startup(self) -> None:
        with self._storage_lock:
            res = self.storage.sync_down().as_dict()
            self.storage_last_sync_down = res

    def _sync_to_storage(self, *, commit_message: str) -> Dict[str, Any]:
        with self._storage_lock:
            res = self.storage.sync_up(commit_message=commit_message).as_dict()
            self.storage_last_sync_up = res
            return res

    def _refresh_runtime_artifacts(self) -> None:
        try:
            self.retriever = AlignmentAtlasRetriever()
            self.resolver = CitationResolver()
            self.retriever_init_error = None
        except Exception as e:
            self.retriever = None
            self.resolver = CitationResolver()
            self.retriever_init_error = str(e)

    def _run_stage(
        self,
        module_name: str,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stage_index: Optional[int] = None,
        stage_total: Optional[int] = None,
        stage_results: Optional[List[Dict[str, Any]]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        pipeline = IngestPipeline()
        _ok, rec = pipeline.run_single_stage(
            module_name,
            progress_callback=progress_callback,
            env_overrides=env_overrides,
        )
        if stage_index is not None or stage_total is not None:
            rec["stage_index"] = stage_index or 0
            rec["stage_total"] = stage_total or 0
        return rec

    def _run_ingest_pipeline(
        self,
        run_relations: bool,
        doc: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        claims_env: Optional[Dict[str, str]] = None,
    ) -> IngestRunResult:
        pipeline = IngestPipeline(
            stages=[
                "collect_manifest",
                "download_sources",
                "pdf_to_text",
                "html_to_text",
                "section_chunk",
                "apply_neighbors",
                "embed_chunks",
                "export_chunk_embs",
                "extract_claims",
                "build_kg",
            ]
        )
        run = pipeline.run_full(
            run_relations=run_relations,
            progress_callback=progress_callback,
            claims_env=claims_env,
        )
        if not run.ok:
            return IngestRunResult(
                ok=False,
                doc=doc,
                stage_results=run.stage_results,
                error=f"Pipeline failed at {run.failed_module}",
            )
        return IngestRunResult(ok=True, doc=doc, stage_results=run.stage_results)

    def _run_incremental_ingest_pipeline(
        self,
        run_relations: bool,
        doc: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        claims_env: Optional[Dict[str, str]] = None,
    ) -> IngestRunResult:
        stage_results: List[Dict[str, Any]] = []
        stage_total = 8 if run_relations else 7
        stage_index = 0
        step_timeout_sec = float(os.environ.get("INGEST_INCREMENTAL_STEP_TIMEOUT_SECONDS", "600"))

        def _tick(stage_name: str) -> None:
            nonlocal stage_index
            stage_index += 1
            msg = f"[incremental] starting {stage_name}"
            print(msg, flush=True)
            if progress_callback is not None:
                progress_callback(
                    {
                        "current_stage": stage_name,
                        "stage_index": stage_index,
                        "stage_total": stage_total,
                        "stage_results": stage_results,
                        "detail_message": msg,
                    }
                )

        def _run_step_with_heartbeat(stage_name: str, fn: Callable[[], Any]) -> Any:
            done = threading.Event()
            box: Dict[str, Any] = {"result": None, "error": None}
            started = time.time()

            def _target() -> None:
                try:
                    box["result"] = fn()
                except Exception as e:  # pragma: no cover - defensive pass-through
                    box["error"] = e
                finally:
                    done.set()

            t = threading.Thread(target=_target, daemon=True, name=f"inc-step-{stage_name}")
            t.start()
            while not done.wait(2.0):
                elapsed = round(time.time() - started, 1)
                hb = f"[incremental] {stage_name} running for {elapsed}s..."
                print(hb, flush=True)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "current_stage": stage_name,
                            "stage_index": stage_index,
                            "stage_total": stage_total,
                            "stage_results": stage_results,
                            "detail_message": hb,
                        }
                    )
                if elapsed >= step_timeout_sec:
                    raise TimeoutError(
                        f"{stage_name} exceeded {step_timeout_sec:.0f}s; "
                        "aborting incremental step to avoid indefinite hang."
                    )
            if box["error"] is not None:
                raise box["error"]
            elapsed_done = round(time.time() - started, 1)
            done_msg = f"[incremental] finished {stage_name} in {elapsed_done}s"
            print(done_msg, flush=True)
            if progress_callback is not None:
                progress_callback(
                    {
                        "current_stage": stage_name,
                        "stage_index": stage_index,
                        "stage_total": stage_total,
                        "stage_results": stage_results,
                        "detail_message": done_msg,
                    }
                )
            return box["result"]

        def _ok_stage(module: str, payload: Dict[str, Any]) -> None:
            stage_results.append(
                {
                    "module": module,
                    "ok": True,
                    "return_code": 0,
                    "elapsed_seconds": payload.get("elapsed_seconds", 0.0),
                    "output_tail": json.dumps(payload, ensure_ascii=False, indent=2)[-6000:],
                }
            )

        _tick("collect_manifest")
        collect = self._run_stage("collect_manifest")
        stage_results.append(collect)
        if not collect["ok"]:
            return IngestRunResult(
                ok=False, doc=doc, stage_results=stage_results, error="Incremental failed at docs manifest stage"
            )

        try:
            _tick("incremental.download_single_doc")
            download_info = _run_step_with_heartbeat(
                "incremental.download_single_doc", lambda: self._download_single_doc(doc)
            )
            _ok_stage("incremental.download_single_doc", download_info)

            _tick("incremental.materialize_text")
            text_info = _run_step_with_heartbeat(
                "incremental.materialize_text", lambda: self._materialize_single_text(doc)
            )
            _ok_stage("incremental.materialize_text", text_info)

            _tick("incremental.append_chunks")
            new_chunks = _run_step_with_heartbeat(
                "incremental.append_chunks", lambda: self._append_single_doc_chunks(doc["doc_id"])
            )
            _ok_stage(
                "incremental.append_chunks",
                {"doc_id": doc["doc_id"], "num_new_chunks": len(new_chunks), "elapsed_seconds": 0.0},
            )

            _tick("incremental.append_embeddings")
            emb_info = _run_step_with_heartbeat(
                "incremental.append_embeddings", lambda: self._append_embeddings_for_chunks(new_chunks)
            )
            _ok_stage("incremental.append_embeddings", emb_info)

            _tick("incremental.extract_claims")
            claims_batch = int((claims_env or {}).get("CLAIMS_BATCH_SIZE", 10))
            claims_conc = int((claims_env or {}).get("CLAIMS_MAX_CONCURRENCY", 8))
            claims_info = _run_step_with_heartbeat(
                "incremental.extract_claims",
                lambda: self._extract_claims_for_chunks(
                    new_chunks,
                    batch_size=claims_batch,
                    max_concurrency=claims_conc,
                    progress_callback=progress_callback,
                    stage_index=stage_index,
                    stage_total=stage_total,
                    stage_results=stage_results,
                ),
            )
            _ok_stage("incremental.extract_claims", claims_info)
        except Exception as e:
            return IngestRunResult(
                ok=False,
                doc=doc,
                stage_results=stage_results,
                error=f"Incremental ingest error: {e}",
            )

        no_new_chunks = len(new_chunks) == 0
        no_new_claims = int(claims_info.get("num_claims_written", 0)) == 0
        if no_new_chunks and no_new_claims:
            _ok_stage(
                "incremental.skip_graph_updates",
                {
                    "doc_id": doc.get("doc_id"),
                    "reason": "No new chunks or claims detected; skipping KG/relation rebuild stages.",
                    "elapsed_seconds": 0.0,
                },
            )
            fallback = self._ensure_graph_with_relations_fallback()
            _ok_stage("incremental.ensure_graph_with_relations", fallback)
            return IngestRunResult(ok=True, doc=doc, stage_results=stage_results)

        _tick("build_kg")
        build_kg = self._run_stage("build_kg")
        stage_results.append(build_kg)
        if not build_kg["ok"]:
            return IngestRunResult(ok=False, doc=doc, stage_results=stage_results, error="Stage 5 KG build failed")

        if run_relations:
            _tick("detect_contradictions")
            rel = self._run_stage("detect_contradictions")
            stage_results.append(rel)
            if not rel["ok"]:
                return IngestRunResult(ok=False, doc=doc, stage_results=stage_results, error="Stage 6 failed")
            _tick("merge_relations_into_kg")
            merge = self._run_stage("merge_relations_into_kg")
            stage_results.append(merge)
            if not merge["ok"]:
                return IngestRunResult(ok=False, doc=doc, stage_results=stage_results, error="Stage 7 failed")
        else:
            _tick("incremental.ensure_graph_with_relations")
            fallback = self._ensure_graph_with_relations_fallback()
            _ok_stage("incremental.ensure_graph_with_relations", fallback)

        return IngestRunResult(ok=True, doc=doc, stage_results=stage_results)

    def _load_docs_map(self) -> Dict[str, Dict[str, Any]]:
        return self._incremental_ops._load_docs_map()

    def _download_single_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return self._incremental_ops.download_single_doc(doc)

    def _materialize_single_text(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return self._incremental_ops.materialize_single_text(doc)

    def _append_single_doc_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        return self._incremental_ops.append_single_doc_chunks(doc_id)

    def _append_embeddings_for_chunks(self, chunk_recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunk_recs:
            return {"num_new_embeddings": 0}
        if not META_PATH.exists() or not ROWIDS_JSON.exists() or not EMBS_NPY.exists():
            rebuild1 = self._run_stage("embed_chunks")
            rebuild2 = self._run_stage("export_chunk_embs")
            if not rebuild1["ok"] or not rebuild2["ok"]:
                raise RuntimeError("Missing index artifacts and full rebuild failed.")
            return {"num_new_embeddings": len(chunk_recs), "mode": "full_rebuild_fallback"}

        from src.retrieval.retriever import TorchEmbedder

        embedder = TorchEmbedder()
        texts = [str(r.get("text", "")) for r in chunk_recs]
        new_embs = np.vstack([embedder.encode_one(t) for t in texts]).astype("float32")

        old_embs = np.load(str(EMBS_NPY)).astype("float32")
        merged = np.vstack([old_embs, new_embs]).astype("float32")
        np.save(EMBS_NPY, merged)

        max_row_id = -1
        for rec in _iter_jsonl(META_PATH):
            max_row_id = max(max_row_id, int(rec.get("row_id", -1)))
        new_rows = list(range(max_row_id + 1, max_row_id + 1 + len(chunk_recs)))

        with META_PATH.open("a", encoding="utf-8") as f:
            for row_id, rec in zip(new_rows, chunk_recs):
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

        row_ids = json.loads(ROWIDS_JSON.read_text(encoding="utf-8"))
        row_ids.extend(new_rows)
        ROWIDS_JSON.write_text(json.dumps(row_ids, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"num_new_embeddings": len(chunk_recs), "embedding_device": embedder.device}

    def _extract_claims_for_chunks(
        self,
        chunk_recs: List[Dict[str, Any]],
        batch_size: int = 10,
        max_concurrency: int = 8,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stage_index: Optional[int] = None,
        stage_total: Optional[int] = None,
        stage_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self._incremental_ops.extract_claims_for_chunks(
            chunk_recs,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            progress_callback=progress_callback,
            stage_index=stage_index or 0,
            stage_total=stage_total or 0,
            stage_results=stage_results,
        )

    def _ensure_graph_with_relations_fallback(self) -> Dict[str, Any]:
        KG_DIR.mkdir(parents=True, exist_ok=True)
        copied: List[str] = []
        if GRAPH_GRAPHML.exists():
            shutil.copyfile(GRAPH_GRAPHML, GRAPH_WITH_REL_GRAPHML)
            copied.append(str(GRAPH_WITH_REL_GRAPHML.as_posix()))
        if GRAPH_JSON.exists():
            shutil.copyfile(GRAPH_JSON, GRAPH_WITH_REL_JSON)
            copied.append(str(GRAPH_WITH_REL_JSON.as_posix()))
        if not copied:
            raise FileNotFoundError("Stage 5 outputs missing; cannot create graph_with_relations fallback.")
        return {"copied_outputs": copied}


def _normalize_citations(raw: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for c in raw or []:
        if not isinstance(c, dict):
            continue
        kind = str(c.get("kind", "")).strip().lower()
        cid = str(c.get("id", "")).strip()
        if kind in {"claim", "chunk", "external"} and cid:
            out.append({"kind": kind, "id": cid})
    uniq: List[Dict[str, str]] = []
    seen = set()
    for c in out:
        key = (c["kind"], c["id"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def render_answer_markdown(
    payload: Dict[str, Any],
    *,
    resolver: Optional[CitationResolver] = None,
    snippet_chars: int = 260,
) -> str:
    ans = payload.get("answer", {}) or {}
    lines: List[str] = []
    title = (ans.get("title") or "Answer").strip()
    summary = (ans.get("summary") or "").strip()
    lines.append(f"## {title}")
    if summary:
        lines.append(summary)
    evidence_status = str(payload.get("evidence_status") or "").strip()
    if evidence_status:
        lines.append(f"\n- Evidence status: `{evidence_status}`")
    fallback_reason = str(payload.get("fallback_reason") or "").strip()
    if fallback_reason:
        lines.append(f"- Note: {fallback_reason}")

    citation_lookup = {}
    for c in ans.get("citations", []) or []:
        if not isinstance(c, dict):
            continue
        kind = str(c.get("kind", "")).strip().lower()
        cid = str(c.get("id", "")).strip()
        if kind and cid:
            citation_lookup[(kind, cid)] = c

    def _source_tag(source_type: str) -> str:
        m = {
            "atlas": "Atlas",
            "external_scholarly": "External Scholar",
            "external_web": "External Web",
        }
        return m.get((source_type or "").strip().lower(), "Source")

    def _resolve_for_display(cits_norm: List[Dict[str, str]]) -> List[Any]:
        active_resolver = resolver or CitationResolver()
        out = []
        for c in cits_norm:
            inline = citation_lookup.get((c["kind"], c["id"]))
            r = active_resolver.resolve_one(inline or c, snippet_chars=snippet_chars)
            if r:
                out.append(r)
        return out

    key_points = ans.get("key_points", []) or []
    if key_points:
        lines.append("\n### Key Points")
        for i, kp in enumerate(key_points, 1):
            lines.append(f"{i}. {kp.get('point', '').strip()}")
            cits_norm = _normalize_citations(kp.get("citations", []))
            resolved = _resolve_for_display(cits_norm)
            if resolved:
                src_parts: List[str] = []
                seen = set()
                for r in resolved:
                    key = (r.kind, r.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    tag = _source_tag(getattr(r, "source_type", "atlas"))
                    label = f"[{tag}] {r.title} - {r.section}"
                    src_parts.append(f"[{label}]({r.url})" if r.url else label)
                if src_parts:
                    lines.append(f"   - Sources: {'; '.join(src_parts)}")
                lines.append(f"   - Evidence: \"{resolved[0].snippet}\"")
            elif cits_norm:
                raw_refs = [f"{c['kind']}:{c['id']}" for c in cits_norm]
                lines.append(f"   - Sources: {'; '.join(raw_refs)}")

    debates = ans.get("debates_and_contradictions", []) or []
    if debates:
        lines.append("\n### Debates / Contradictions")
        for i, d in enumerate(debates, 1):
            lines.append(f"{i}. {d.get('debate', '').strip()}")
            cits_norm = _normalize_citations(d.get("citations", []))
            resolved = _resolve_for_display(cits_norm)
            if resolved:
                src_parts: List[str] = []
                seen = set()
                for r in resolved:
                    key = (r.kind, r.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    tag = _source_tag(getattr(r, "source_type", "atlas"))
                    label = f"[{tag}] {r.title} - {r.section}"
                    src_parts.append(f"[{label}]({r.url})" if r.url else label)
                if src_parts:
                    lines.append(f"   - Sources: {'; '.join(src_parts)}")
            elif cits_norm:
                raw_refs = [f"{c['kind']}:{c['id']}" for c in cits_norm]
                lines.append(f"   - Sources: {'; '.join(raw_refs)}")

    limitations = ans.get("limitations", []) or []
    if limitations:
        lines.append("\n### Limitations")
        for i, l in enumerate(limitations, 1):
            lines.append(f"{i}. {str(l).strip()}")

    suggestions = payload.get("suggestions", []) or []
    if suggestions:
        lines.append("\n### Suggested Follow-ups")
        for s in suggestions:
            lines.append(f"- {s}")

    return "\n".join(lines).strip()

