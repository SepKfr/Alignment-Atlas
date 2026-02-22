from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from src.app.services import AtlasService

ROOT = Path(__file__).resolve().parents[2]
DOCS_JSONL = ROOT / "data" / "processed" / "docs.jsonl"
UI_DEV_DEBUG = os.environ.get("ATLAS_UI_DEBUG", "0").strip().lower() in {"1", "true", "yes"}


GOAL_OPTIONS = {
    "Core alignment fundamentals": "foundations",
    "Reward hacking and specification gaming": "reward_hacking",
    "Oversight and evaluation methods": "oversight_eval",
    "Interpretability for alignment": "interpretability",
    "Safe deployment and operations": "deployment",
}

STEERING_OPTIONS = {
    "Risk and safety implications": "safety_first",
    "Mechanisms and interpretability": "interpretability_first",
    "Practical deployment guidance": "practical_deployment",
}

EXAMPLE_QUESTIONS_BY_GOAL = {
    "Core alignment fundamentals": [
        "What are the core disagreements in alignment research today?",
        "How do papers distinguish outer alignment vs inner alignment?",
        "What assumptions recur across major alignment frameworks?",
        "Which foundational risks are most evidence-backed vs speculative?",
    ],
    "Reward hacking and specification gaming": [
        "What are the strongest documented examples of reward hacking?",
        "How do papers categorize specification gaming failure modes?",
        "Which mitigations for reward hacking show the best evidence so far?",
        "Where do current RLHF-style methods still fail against gaming?",
    ],
    "Oversight and evaluation methods": [
        "How do current oversight methods detect latent misalignment?",
        "Which eval techniques are most predictive of deployment failures?",
        "How do papers compare red-teaming, audits, and adversarial evals?",
        "What are the biggest blind spots in current alignment evaluations?",
    ],
    "Interpretability for alignment": [
        "Which interpretability methods appear most useful for safety assurance?",
        "Where do mechanistic interpretability results actually change risk estimates?",
        "What are the limits of probes and feature-based explanations for safety?",
        "How do papers connect interpretability findings to concrete interventions?",
    ],
    "Safe deployment and operations": [
        "What deployment controls are most consistently recommended across papers?",
        "How should teams stage rollout gates for high-risk model behaviors?",
        "What monitoring signals best capture emerging alignment failures in production?",
        "How do papers balance capability, latency, and safety trade-offs in deployment?",
    ],
}

GUIDED_STEPS: List[Dict[str, Any]] = [
    {
        "id": "solvability",
        "question": "Is alignment mostly an engineering problem or fundamentally unsolved?",
        "options": [
            "Mostly engineering",
            "Mixed engineering + theory",
            "Fundamentally unsolved",
            "Unsure",
        ],
    },
    {
        "id": "reward_hacking",
        "question": "How concerned are you about reward hacking / specification gaming?",
        "options": ["Low", "Moderate", "High", "Severe"],
    },
    {
        "id": "rlhf",
        "question": "Do you think RLHF aligns intent or mostly behavior?",
        "options": [
            "Mostly intent alignment",
            "Mostly behavior shaping",
            "Helps somewhat but incomplete",
            "Unsure",
        ],
    },
    {
        "id": "dominant_risk",
        "question": "Which risk dominates in your view?",
        "options": [
            "Deception / alignment-faking",
            "Reward hacking / spec gaming",
            "Misuse / abuse",
            "Governance / coordination failure",
        ],
    },
    {
        "id": "confidence",
        "question": "How confident are you in current evaluation and oversight methods?",
        "options": ["Low", "Moderate", "High", "Unsure"],
    },
    {
        "id": "timeline",
        "question": "What deployment horizon worries you most?",
        "options": ["Current systems", "Next 2-5 years", "Long-term advanced systems", "All horizons"],
    },
]


@st.cache_resource
def get_service() -> AtlasService:
    return AtlasService()


def _inject_ui_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(1200px 500px at 10% -10%, rgba(59,130,246,0.10), transparent 55%),
                radial-gradient(1200px 500px at 90% -20%, rgba(14,165,233,0.08), transparent 50%);
        }
        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 2rem !important;
            max-width: 1100px;
        }
        h1, h2, h3 {
            letter-spacing: -0.01em;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        button[role="tab"] {
            border-radius: 12px !important;
            padding: 0.45rem 0.9rem !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            background: rgba(15, 23, 42, 0.04) !important;
        }
        button[role="tab"][aria-selected="true"] {
            border: 1px solid rgba(37, 99, 235, 0.55) !important;
            background: rgba(37, 99, 235, 0.14) !important;
        }
        div[data-testid="stTextArea"] textarea {
            border-radius: 12px !important;
            border: 1px solid rgba(148, 163, 184, 0.45) !important;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(148, 163, 184, 0.28) !important;
            border-radius: 12px !important;
            background: rgba(255, 255, 255, 0.02);
        }
        div[data-testid="stButton"] button {
            border-radius: 10px !important;
            border: 1px solid rgba(148, 163, 184, 0.35) !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
            overflow: hidden;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "draft_message" not in st.session_state:
        st.session_state.draft_message = ""
    if "goal_label" not in st.session_state:
        st.session_state.goal_label = "Core alignment fundamentals"
    if "steering_label" not in st.session_state:
        st.session_state.steering_label = "Risk and safety implications"
    if "ingest_result_json" not in st.session_state:
        st.session_state.ingest_result_json = ""
    if "ingest_stage_logs" not in st.session_state:
        st.session_state.ingest_stage_logs = ""
    if "ingest_status" not in st.session_state:
        st.session_state.ingest_status = ""
    if "ingest_job_id" not in st.session_state:
        st.session_state.ingest_job_id = None
    if "ingest_live_progress" not in st.session_state:
        st.session_state.ingest_live_progress = {}
    if "ingest_guardrail_preview" not in st.session_state:
        st.session_state.ingest_guardrail_preview = None
    if "ingest_guardrail_preview_url" not in st.session_state:
        st.session_state.ingest_guardrail_preview_url = ""
    if "ingest_review_override" not in st.session_state:
        st.session_state.ingest_review_override = False
    if "pending_draft_message" not in st.session_state:
        st.session_state.pending_draft_message = None
    if "live_token_stream" not in st.session_state:
        st.session_state.live_token_stream = True
    if "guided_idx" not in st.session_state:
        st.session_state.guided_idx = 0
    if "guided_answers" not in st.session_state:
        st.session_state.guided_answers = []
    if "guided_records" not in st.session_state:
        st.session_state.guided_records = []
    if "map_selected_doc" not in st.session_state:
        st.session_state.map_selected_doc = None
    if "stats_selected_bucket" not in st.session_state:
        st.session_state.stats_selected_bucket = None


def _normalize_citations(raw: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for c in raw or []:
        if not isinstance(c, dict):
            continue
        kind = str(c.get("kind", "")).strip().lower()
        cid = str(c.get("id", "")).strip()
        if kind in {"chunk", "claim", "external"} and cid:
            out.append({"kind": kind, "id": cid})
    seen = set()
    uniq: List[Dict[str, str]] = []
    for c in out:
        key = (c["kind"], c["id"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _is_low_quality_evidence_snippet(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return True
    # Likely truncated/mid-sentence starts.
    if t[0].islower():
        return True
    first = t.split()[0].strip(".,;:!?()[]{}\"'").lower()
    if len(first) <= 2 and first not in {"ai", "ml", "rl"}:
        return True
    return False


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


def _is_valid_ingest_url(url: str) -> bool:
    u = (url or "").strip()
    if not u:
        return False
    try:
        p = urlparse(u)
    except Exception:
        return False
    return p.scheme in {"http", "https"} and bool(p.netloc)


@st.cache_data(show_spinner=False)
def _load_docs_for_stats() -> List[Dict[str, Any]]:
    docs = _iter_jsonl(DOCS_JSONL)
    out: List[Dict[str, Any]] = []
    for d in docs:
        did = str(d.get("doc_id", "")).strip()
        if not did:
            continue
        out.append(
            {
                "doc_id": did,
                "title": str(d.get("title", did)).strip() or did,
                "year": d.get("year"),
                "source_type": str(d.get("source_type", "unknown")).strip() or "unknown",
                "source_url": str(d.get("source_url", "")).strip(),
            }
        )
    return out


def _year_bucket(year: Any) -> str:
    try:
        y = int(year)
    except Exception:
        return "Unknown year"
    if y < 2015:
        return "<2015"
    if y < 2020:
        return "2015-2019"
    if y < 2023:
        return "2020-2022"
    return "2023+"


def _render_corpus_stats() -> None:
    docs = _load_docs_for_stats()
    if not docs:
        st.caption("No docs available yet for corpus stats.")
        return

    mode = st.selectbox(
        "Stats breakdown",
        options=["Source type", "Year bucket"],
        index=0,
        key="stats_mode",
    )
    if mode == "Source type":
        get_bucket = lambda d: d["source_type"].upper()
    else:
        get_bucket = lambda d: _year_bucket(d.get("year"))

    counts: Dict[str, int] = defaultdict(int)
    for d in docs:
        counts[get_bucket(d)] += 1

    labels = sorted(counts.keys(), key=lambda x: (-counts[x], x))
    values = [counts[l] for l in labels]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} papers (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 20, "b": 0}, height=360)

    event = st.plotly_chart(
        fig,
        width="stretch",
        key="corpus_stats_donut",
        on_select="rerun",
        selection_mode="points",
    )
    if event and isinstance(event, dict):
        points = (event.get("selection") or {}).get("points", [])
        if points:
            p = points[0]
            label = p.get("label")
            if not label and isinstance(p.get("point_number"), int):
                idx = int(p.get("point_number"))
                if 0 <= idx < len(labels):
                    label = labels[idx]
            if label:
                st.session_state.stats_selected_bucket = str(label)

    selected = st.session_state.stats_selected_bucket
    if not selected:
        st.caption("Click a donut slice to list papers in that group.")
        return

    st.markdown(f"#### Papers in `{selected}`")
    bucket_docs = [d for d in docs if get_bucket(d) == selected]
    bucket_docs.sort(key=lambda x: (str(x.get("title", "")).lower(), str(x.get("doc_id", ""))))
    for d in bucket_docs:
        title = d["title"]
        did = d["doc_id"]
        yr = d.get("year")
        src = d.get("source_type", "unknown").upper()
        line = f"- **{title}** (`{did}`) - {src}" + (f", {yr}" if yr else "")
        if d.get("source_url"):
            line += f" - [Open source]({d['source_url']})"
        st.markdown(line)


def _source_tag(source_type: str) -> str:
    m = {
        "atlas": "Atlas",
        "external_scholarly": "External Scholar",
        "external_web": "External Web",
    }
    return m.get((source_type or "").strip().lower(), "Source")


def _build_evidence_rows(
    service: AtlasService,
    payload: Dict[str, Any],
    *,
    snippet_chars: int = 950,
    debug_counts: Dict[str, int] | None = None,
) -> List[Dict[str, Any]]:
    ans = payload.get("answer", {}) or {}
    citation_lookup = {}
    for c in ans.get("citations", []) or []:
        if not isinstance(c, dict):
            continue
        kind = str(c.get("kind", "")).strip().lower()
        cid = str(c.get("id", "")).strip()
        if kind and cid:
            citation_lookup[(kind, cid)] = c

    section_citations: List[Dict[str, str]] = []
    for kp in ans.get("key_points", []) or []:
        section_citations.extend(_normalize_citations(kp.get("citations", [])))
    for d in ans.get("debates_and_contradictions", []) or []:
        section_citations.extend(_normalize_citations(d.get("citations", [])))
    global_citations = _normalize_citations(ans.get("citations", []))

    def _rows_from_pool(pool: List[Dict[str, str]], *, filter_low_quality: bool = True) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen = set()
        row_index_by_fingerprint: Dict[str, int] = {}
        for c in pool:
            key = (c["kind"], c["id"])
            if key in seen:
                continue
            seen.add(key)
            inline = citation_lookup.get(key)
            r = service.resolver.resolve_one(inline or c, snippet_chars=snippet_chars)
            if not r:
                continue
            if filter_low_quality and _is_low_quality_evidence_snippet(str(r.snippet)):
                continue
            evidence_key = f"{r.kind}:{r.id}"
            fingerprint = "||".join(
                [
                    str(getattr(r, "source_type", "")).strip().lower(),
                    str(r.title).strip().lower(),
                    str(r.section).strip().lower(),
                    re.sub(r"\s+", " ", str(r.snippet)).strip().lower(),
                ]
            )
            if fingerprint in row_index_by_fingerprint:
                ridx = row_index_by_fingerprint[fingerprint]
                alias_keys = rows[ridx].setdefault("evidence_keys", [])
                if evidence_key not in alias_keys:
                    alias_keys.append(evidence_key)
                continue
            row_index_by_fingerprint[fingerprint] = len(rows)
            rows.append(
                {
                    "row": len(rows) + 1,
                    "source": f"{_source_tag(getattr(r, 'source_type', 'atlas'))} - {r.title}",
                    "title": r.title,
                    "doc_id": r.doc_id,
                    "kind": r.kind,
                    "section": r.section,
                    "evidence_key": evidence_key,
                    "evidence_keys": [evidence_key],
                    "url": r.url,
                    "snippet_full": r.snippet,
                    "snippet_preview": (r.snippet[:220] + "...") if len(r.snippet) > 220 else r.snippet,
                }
            )
        return rows

    # 1) Prefer section-level evidence.
    rows = _rows_from_pool(section_citations, filter_low_quality=True)
    if debug_counts is not None:
        debug_counts["section_citations"] = len(section_citations)
        debug_counts["global_citations"] = len(global_citations)
        debug_counts["rows_from_sections"] = len(rows)
    # 2) If section-level citations existed but yielded nothing, fallback to global answer citations.
    if not rows and global_citations:
        rows = _rows_from_pool(global_citations, filter_low_quality=True)
    if debug_counts is not None:
        debug_counts["rows_after_global_fallback"] = len(rows)
    # 3) Final safety net: if quality filter removed everything, keep at least resolvable evidence.
    if not rows:
        fallback_pool = section_citations or global_citations
        rows = _rows_from_pool(fallback_pool, filter_low_quality=False)
    if debug_counts is not None:
        debug_counts["rows_final"] = len(rows)
    return rows


def _render_answer_with_evidence_refs(payload: Dict[str, Any], rows: List[Dict[str, Any]], *, msg_idx: int) -> str:
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

    row_by_citation: Dict[str, int] = {}
    row_obj_by_citation: Dict[str, Dict[str, Any]] = {}
    row_obj_by_id: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        rid = int(r.get("row", 0))
        if rid:
            row_obj_by_id[rid] = r
        for ek in r.get("evidence_keys", []) or []:
            if ek:
                row_by_citation[str(ek)] = rid
                row_obj_by_citation[str(ek)] = r

    def _rows_for_citations(raw_cits: Any) -> List[int]:
        out: List[int] = []
        seen = set()
        for c in _normalize_citations(raw_cits):
            key = f"{c['kind']}:{c['id']}"
            rid = row_by_citation.get(key)
            if rid and rid not in seen:
                seen.add(rid)
                out.append(rid)
        return sorted(out)

    def _source_link(row_id: int) -> str:
        row = row_obj_by_id.get(row_id, {})
        title = str(row.get("title", "")).strip()
        section = str(row.get("section", "")).strip()
        tooltip = title
        if section:
            tooltip = f"{title} - {section}" if title else section
        if tooltip:
            safe_tooltip = tooltip.replace('"', "'")
            return f"[Source {row_id}](#evidence-m{msg_idx}-r{row_id} \"{safe_tooltip}\")"
        return f"[Source {row_id}](#evidence-m{msg_idx}-r{row_id})"

    def _inline_source_refs(raw_cits: Any) -> str:
        rows_here = _rows_for_citations(raw_cits)
        if not rows_here:
            return ""
        return ", ".join([_source_link(r) for r in rows_here])

    key_points = ans.get("key_points", []) or []
    if key_points:
        lines.append("\n### Key Points")
        for i, kp in enumerate(key_points, 1):
            point_text = kp.get("point", "").strip()
            refs = _inline_source_refs(kp.get("citations", []))
            if refs:
                lines.append(f"{i}. {point_text} ({refs})")
            else:
                lines.append(f"{i}. {point_text}")

    debates = ans.get("debates_and_contradictions", []) or []
    if debates:
        lines.append("\n### Debates / Contradictions")
        for i, d in enumerate(debates, 1):
            debate_text = d.get("debate", "").strip()
            refs = _inline_source_refs(d.get("citations", []))
            if refs:
                lines.append(f"{i}. {debate_text} ({refs})")
            else:
                lines.append(f"{i}. {debate_text}")

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


def _render_evidence_explorer(rows: List[Dict[str, Any]], *, msg_idx: int) -> None:
    if not rows:
        st.caption("No evidence rows available for this answer.")
        return

    st.caption("Evidence rows (source + section + link)")
    table_rows = []
    for r in rows:
        table_rows.append(
            {
                "row": r["row"],
                "source": r["source"],
                "section": r["section"],
                "link": r["url"] or "",
            }
        )
    st.dataframe(table_rows, width="stretch")
    st.markdown("### Evidence details")
    for r in rows:
        st.markdown(
            f"<a id='evidence-m{msg_idx}-r{r['row']}'></a>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Row {r['row']} - {r['source']}**")
        st.markdown(f"- Section: `{r['section']}`")
        if r.get("url"):
            st.markdown(f"- Source link: [{r['url']}]({r['url']})")
        st.markdown("---")


def _render_markdown_with_math(text: str) -> None:
    """
    Render markdown and display-math blocks without showing raw $$...$$ delimiters.
    """
    raw = str(text or "")
    if not raw.strip():
        return
    pattern = re.compile(r"\$\$(.+?)\$\$", flags=re.S)
    cursor = 0
    for m in pattern.finditer(raw):
        before = raw[cursor : m.start()]
        if before.strip():
            st.markdown(before)
        equation = (m.group(1) or "").strip()
        if equation:
            st.latex(equation)
        cursor = m.end()
    tail = raw[cursor:]
    if tail.strip():
        st.markdown(tail)


def _guided_query(step: Dict[str, Any], choice: str, free_text: str) -> str:
    q = f"{step.get('question', '')} Selection: {choice}."
    if free_text.strip():
        q += f" User note: {free_text.strip()}"
    return q


def _cit_from_row(row: Dict[str, Any]) -> str:
    return f"{row.get('source', 'Unknown')} - {row.get('section', 'unknown')}"


def _extract_challenge(service: AtlasService, evidence: Dict[str, Any]) -> Dict[str, Any]:
    rels = (evidence.get("relations", {}) or {}).get("contradiction", []) or []
    if not rels:
        return {"status": "neutral", "message": "No high-confidence contradiction found for this step."}
    best = sorted(rels, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)[0]
    src = service.resolver.resolve_one({"kind": "claim", "id": str(best.get("src", ""))}, snippet_chars=460)
    dst = service.resolver.resolve_one({"kind": "claim", "id": str(best.get("dst", ""))}, snippet_chars=460)
    if not src and not dst:
        return {"status": "neutral", "message": "Contradiction edge exists, but snippets were unavailable."}
    return {
        "status": "contradiction",
        "confidence": float(best.get("confidence", 0.0)),
        "justification": str(best.get("justification", "")),
        "src": src,
        "dst": dst,
        "src_claim_id": str(best.get("src", "")),
        "dst_claim_id": str(best.get("dst", "")),
    }


def _render_guided_tab(service: AtlasService) -> None:
    st.subheader("Guided Exploration / Stance Builder")
    st.caption("Answer 6 short prompts. Each step retrieves grounded evidence plus a challenge snippet.")

    progress = st.session_state.guided_idx / max(1, len(GUIDED_STEPS))
    st.progress(progress)

    if st.session_state.guided_idx >= len(GUIDED_STEPS):
        st.success("Guided exploration complete.")
        answers = st.session_state.guided_answers
        records = st.session_state.guided_records
        st.markdown("### Your stance summary")
        for a in answers:
            st.markdown(f"- **{a['question']}** -> {a['choice']}" + (f" ({a['note']})" if a["note"] else ""))

        st.markdown("### Supporting evidence")
        shown = 0
        seen_cits = set()
        for rec in records:
            for sn in rec.get("snippets", []):
                cit = sn.get("citation", "")
                if cit in seen_cits:
                    continue
                seen_cits.add(cit)
                shown += 1
                st.markdown(f"- {cit}")
                st.caption(sn.get("snippet", ""))
                if sn.get("url"):
                    st.markdown(f"  - [Source link]({sn['url']})")
                if shown >= 8:
                    break
            if shown >= 8:
                break

        st.markdown("### Open disagreements / uncertainties")
        any_disagreement = False
        for rec in records:
            ch = rec.get("challenge", {}) or {}
            if ch.get("status") != "contradiction":
                continue
            any_disagreement = True
            st.markdown(
                f"- Step: **{rec.get('question', '')}** "
                f"(confidence `{round(float(ch.get('confidence', 0.0)), 3)}`)"
            )
            if ch.get("justification"):
                st.caption(ch["justification"])
            src = ch.get("src")
            dst = ch.get("dst")
            if src:
                st.markdown(f"  - Claim A `{ch.get('src_claim_id')}`: {src.title} - {src.section}")
                st.caption(src.snippet)
            if dst:
                st.markdown(f"  - Claim B `{ch.get('dst_claim_id')}`: {dst.title} - {dst.section}")
                st.caption(dst.snippet)
        if not any_disagreement:
            st.caption("No strong contradiction edges found in this guided run.")

        st.markdown("### What to read next")
        doc_scores: Dict[str, float] = defaultdict(float)
        for rec in records:
            for ch in (rec.get("evidence_pack", {}) or {}).get("chunks", []) or []:
                did = str(ch.get("doc_id", ""))
                if did:
                    doc_scores[did] += max(0.0, float(ch.get("score", 0.0)))
        for did, sc in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:6]:
            title = service.resolver.doc_title(did)
            url = service.resolver.doc_url(did)
            line = f"- **{title}** (`{did}`)"
            if url:
                line += f" - [Open source]({url})"
            line += f" - relevance `{round(sc, 3)}`"
            st.markdown(line)

        if st.button("Restart guided exploration", key="guided_restart"):
            st.session_state.guided_idx = 0
            st.session_state.guided_answers = []
            st.session_state.guided_records = []
            st.rerun()
        return

    step = GUIDED_STEPS[st.session_state.guided_idx]
    st.markdown(f"### Step {st.session_state.guided_idx + 1}: {step['question']}")
    choice = st.radio(
        "Choose one option",
        options=step["options"],
        key=f"guided_choice_{st.session_state.guided_idx}",
    )
    note = st.text_input(
        "Optional note (your own nuance)",
        key=f"guided_note_{st.session_state.guided_idx}",
        placeholder="Optional free text...",
    )

    if st.button("Retrieve evidence for this step", type="primary", key=f"guided_next_{st.session_state.guided_idx}"):
        query = _guided_query(step, choice, note)
        with st.spinner("Retrieving grounded evidence..."):
            evidence = service.retriever.build_evidence_pack(query, top_k_chunks=6, neighbor_radius=0, max_claim_rel_per_claim=3)
            rows = []
            for chunk in (evidence.get("chunks", []) or [])[:4]:
                resolved = service.resolver.resolve_one({"kind": "chunk", "id": str(chunk.get("chunk_id", ""))}, snippet_chars=520)
                if not resolved:
                    continue
                rows.append(
                    {
                        "citation": _cit_from_row({"source": resolved.title, "section": resolved.section}),
                        "snippet": resolved.snippet,
                        "url": resolved.url,
                        "chunk_id": chunk.get("chunk_id"),
                        "doc_id": chunk.get("doc_id"),
                    }
                )
            challenge = _extract_challenge(service, evidence)

        st.session_state.guided_answers.append(
            {
                "step_id": step["id"],
                "question": step["question"],
                "choice": choice,
                "note": note.strip(),
                "query": query,
            }
        )
        st.session_state.guided_records.append(
            {
                "step_id": step["id"],
                "question": step["question"],
                "query": query,
                "snippets": rows,
                "challenge": challenge,
                "evidence_pack": evidence,
            }
        )
        st.session_state.guided_idx += 1
        st.rerun()

    # Show latest retrieved step details if available
    if st.session_state.guided_records:
        rec = st.session_state.guided_records[-1]
        st.markdown("### Latest evidence")
        for i, sn in enumerate(rec.get("snippets", [])[:4], 1):
            st.markdown(f"{i}. **{sn.get('citation', 'citation')}**")
            st.caption(sn.get("snippet", ""))
            if sn.get("url"):
                st.markdown(f"   - [Source link]({sn['url']})")
        st.markdown("### Challenge snippet")
        ch = rec.get("challenge", {}) or {}
        if ch.get("status") == "contradiction":
            st.warning(f"Found contradiction edge (confidence `{round(float(ch.get('confidence', 0.0)), 3)}`)")
            if ch.get("justification"):
                st.caption(ch["justification"])
            if ch.get("dst"):
                st.markdown(f"- Opposing claim `{ch.get('dst_claim_id')}`: {ch['dst'].title} - {ch['dst'].section}")
                st.caption(ch["dst"].snippet)
        else:
            st.info(ch.get("message", "Neutral / uncertain."))


@st.cache_data(show_spinner=False)
def _build_doc_aggregate_edges() -> Dict[str, Any]:
    service = get_service()
    kg = service.retriever.kg
    edge_stats: Dict[tuple, Dict[str, Any]] = {}
    claim_counts: Dict[str, int] = defaultdict(int)
    for nid, attrs in kg.nodes(data=True):
        if attrs.get("type") == "claim":
            did = str(attrs.get("doc_id", ""))
            if did:
                claim_counts[did] += 1

    for src, dst, attrs in kg.edges(data=True):
        rel = str(attrs.get("rel", ""))
        if rel not in {"entails", "contradiction"}:
            continue
        src_doc = str(kg.nodes[src].get("doc_id", ""))
        dst_doc = str(kg.nodes[dst].get("doc_id", ""))
        if not src_doc or not dst_doc:
            continue
        key = (src_doc, dst_doc, rel)
        rec = edge_stats.setdefault(
            key,
            {"src_doc": src_doc, "dst_doc": dst_doc, "rel": rel, "count": 0, "sum_conf": 0.0, "examples": []},
        )
        conf = float(attrs.get("confidence", 0.0))
        rec["count"] += 1
        rec["sum_conf"] += conf
        if len(rec["examples"]) < 8:
            rec["examples"].append(
                {
                    "src_claim_id": str(src),
                    "dst_claim_id": str(dst),
                    "confidence": conf,
                    "justification": str(attrs.get("short_justification", "")),
                }
            )

    edges = []
    for rec in edge_stats.values():
        rec["avg_conf"] = (rec["sum_conf"] / max(1, rec["count"]))
        edges.append(rec)

    return {"edges": edges, "claim_counts": dict(claim_counts)}


def _render_alignment_map_tab(service: AtlasService) -> None:
    st.subheader("Alignment Map")
    st.caption("Interactive doc-level map aggregated from claim-level entailment/contradiction edges.")

    relation_mode = st.selectbox(
        "Relation type",
        options=["both", "entails_only", "contradiction_only"],
        index=0,
    )
    conf_threshold = st.slider("Minimum average edge confidence", 0.5, 0.95, 0.7, 0.01)
    topic_query = st.text_input("Topic search (highlights relevant docs)", placeholder="e.g., reward hacking")
    show_claim_nodes = st.checkbox("Show claim-node details for selected doc", value=False)

    agg = _build_doc_aggregate_edges()
    edges = agg["edges"]
    claim_counts = agg["claim_counts"]
    if relation_mode == "entails_only":
        edges = [e for e in edges if e["rel"] == "entails"]
    elif relation_mode == "contradiction_only":
        edges = [e for e in edges if e["rel"] == "contradiction"]
    edges = [e for e in edges if float(e.get("avg_conf", 0.0)) >= conf_threshold]

    G = nx.DiGraph()
    for e in edges:
        label = "supports/builds-on" if e["rel"] == "entails" else "contradicts"
        G.add_edge(e["src_doc"], e["dst_doc"], rel=e["rel"], label=label, avg_conf=e["avg_conf"], count=e["count"], examples=e["examples"])

    if G.number_of_nodes() == 0:
        st.info("No edges match current filters.")
        return

    highlight_docs = set()
    q = (topic_query or "").strip()
    if q:
        try:
            top = service.retriever.vector_retrieve(q, top_k=20)
            for cid, _ in top:
                ch = service.retriever.chunks_by_id.get(int(cid), {})
                did = str(ch.get("doc_id", ""))
                if did:
                    highlight_docs.add(did)
        except Exception:
            pass
        for did in G.nodes():
            title = service.resolver.doc_title(did).lower()
            if q.lower() in title:
                highlight_docs.add(did)

    pos = nx.spring_layout(G, seed=42, k=0.8)
    edge_traces: List[go.Scatter] = []
    for rel, color in (("entails", "#2ca02c"), ("contradiction", "#d62728")):
        x_vals: List[float] = []
        y_vals: List[float] = []
        for u, v, attrs in G.edges(data=True):
            if attrs.get("rel") != rel:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            x_vals.extend([x0, x1, None])
            y_vals.extend([y0, y1, None])
        edge_traces.append(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line={"width": 1.6, "color": color},
                hoverinfo="none",
                showlegend=False,
            )
        )

    node_x, node_y, node_text, node_size, node_color, node_custom = [], [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        deg = G.degree(n)
        title = service.resolver.doc_title(n)
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{title}<br>{n}<br>degree={deg} claims={claim_counts.get(n, 0)}")
        node_size.append(12 + min(20, deg * 2))
        node_color.append("#ffbf00" if n in highlight_docs else "#4c78a8")
        node_custom.append(n)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[service.resolver.doc_title(n)[:26] for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker={"size": node_size, "color": node_color, "line": {"width": 1, "color": "#1f2937"}},
        customdata=node_custom,
        name="docs",
    )
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=560,
    )

    event = st.plotly_chart(fig, width="stretch", key="alignment_map_plot", on_select="rerun", selection_mode="points")
    if event and isinstance(event, dict):
        pts = (event.get("selection") or {}).get("points", [])
        if pts:
            cd = pts[0].get("customdata")
            if cd:
                st.session_state.map_selected_doc = str(cd)

    selected_doc = st.session_state.map_selected_doc
    if not selected_doc:
        st.caption("Click a node to inspect paper details.")
        return
    st.markdown(f"### Selected paper: {service.resolver.doc_title(selected_doc)}")
    url = service.resolver.doc_url(selected_doc)
    st.markdown(f"- Doc ID: `{selected_doc}`")
    if url:
        st.markdown(f"- Source: [Open paper]({url})")
    st.markdown(f"- Claims in KG: `{claim_counts.get(selected_doc, 0)}`")

    # Top claims
    claim_nodes = []
    for nid, attrs in service.retriever.kg.nodes(data=True):
        if attrs.get("type") == "claim" and str(attrs.get("doc_id", "")) == selected_doc:
            claim_nodes.append(
                {
                    "claim_id": str(nid),
                    "confidence": float(attrs.get("confidence", 0.0)),
                    "claim": str(attrs.get("claim", "")),
                    "section": str(attrs.get("section", "unknown")),
                }
            )
    claim_nodes = sorted(claim_nodes, key=lambda x: x["confidence"], reverse=True)
    st.markdown("#### Top claims")
    for c in claim_nodes[:8]:
        st.markdown(f"- `{c['claim_id']}` ({round(c['confidence'], 3)}) - {c['section']}")
        st.caption(c["claim"])

    # Contradictions touching selected doc
    st.markdown("#### Top contradictions")
    contrad_rows = []
    for e in edges:
        if e["rel"] != "contradiction":
            continue
        if e["src_doc"] != selected_doc and e["dst_doc"] != selected_doc:
            continue
        other_doc = e["dst_doc"] if e["src_doc"] == selected_doc else e["src_doc"]
        contrad_rows.append(
            {
                "other_doc": service.resolver.doc_title(other_doc),
                "other_doc_id": other_doc,
                "avg_conf": round(float(e["avg_conf"]), 3),
                "count": int(e["count"]),
                "examples": e["examples"],
            }
        )
    contrad_rows = sorted(contrad_rows, key=lambda x: x["avg_conf"], reverse=True)[:6]
    if not contrad_rows:
        st.caption("No contradiction edges above threshold for this paper.")
    for row in contrad_rows:
        st.markdown(f"- With **{row['other_doc']}** (`{row['other_doc_id']}`), conf `{row['avg_conf']}`, edges `{row['count']}`")
        if row["examples"]:
            ex = row["examples"][0]
            st.caption(ex.get("justification", ""))
            st.markdown(f"  - claim ids: `{ex.get('src_claim_id')}` -> `{ex.get('dst_claim_id')}`")

    if show_claim_nodes:
        st.markdown("#### Claim-node neighborhood (selected doc)")
        sub = service.retriever.kg.copy()
        keep_nodes = {c["claim_id"] for c in claim_nodes[:20]}
        neighbor_claims = set()
        for n in list(keep_nodes):
            if not sub.has_node(n):
                continue
            for _, dst, attrs in sub.out_edges(n, data=True):
                if attrs.get("rel") in {"entails", "contradiction"}:
                    neighbor_claims.add(dst)
        keep_nodes |= neighbor_claims
        st.caption(f"Showing `{len(keep_nodes)}` claim nodes (top local neighborhood).")
        for cid in list(keep_nodes)[:40]:
            r = service.resolver.resolve_one({"kind": "claim", "id": str(cid)}, snippet_chars=280)
            if not r:
                continue
            st.markdown(f"- `{cid}` - {r.title} / {r.section}")
            st.caption(r.snippet)


def _render_chat_tab(service: AtlasService) -> None:
    if st.session_state.pending_draft_message is not None:
        st.session_state.draft_message = str(st.session_state.pending_draft_message)
        st.session_state.pending_draft_message = None

    st.markdown(
        "Alignment Atlas is a research assistant for AI alignment and safety literature.\n\n"
        "Use it to explore questions across reward hacking, interpretability, oversight, and deployment risk.\n\n"
        "How it works:\n"
        "1) Pick what you want to learn.\n"
        "2) Ask your question in plain language.\n"
        "3) Get evidence-grounded answers from the Atlas corpus (with labeled external sources only when needed)."
    )

    goal_label = st.selectbox(
        "What are you here to learn?",
        options=list(GOAL_OPTIONS.keys()),
        key="goal_label",
    )
    steering_label = st.selectbox(
        "How should answers be framed?",
        options=list(STEERING_OPTIONS.keys()),
        key="steering_label",
    )
    # User-facing UI: keep advanced engineering knobs hidden.
    with st.expander("Example questions", expanded=False):
        goal_examples = EXAMPLE_QUESTIONS_BY_GOAL.get(goal_label, [])
        for q in goal_examples:
            if st.button(q, key=f"example_{q}"):
                st.session_state.pending_draft_message = q
                st.rerun()

    for msg_idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            if msg.get("role") == "assistant" and msg.get("payload"):
                answer_md = _render_answer_with_evidence_refs(
                    msg["payload"],
                    msg.get("evidence_rows", []),
                    msg_idx=msg_idx,
                )
                _render_markdown_with_math(answer_md)
            else:
                st.markdown(msg["content"])
            if msg.get("role") == "assistant" and msg.get("evidence_rows") is not None:
                with st.expander("Evidence Explorer", expanded=False):
                    _render_evidence_explorer(msg["evidence_rows"], msg_idx=msg_idx)

    # Reserve a live-response area above the composer so streaming output
    # appears in chat flow (not below the input controls).
    live_response_area = st.container()
    stage_placeholder = live_response_area.empty()
    stream_placeholder = None

    st.markdown("### Ask")
    st.text_area(
        "Message",
        key="draft_message",
        height=90,
        placeholder="Write your question (or choose an example), then click Send.",
    )

    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        send_draft = st.button("Send", type="primary", width="stretch")
    with action_col2:
        clear = st.button("Clear Chat", width="stretch")

    if clear:
        service.reset_chat()
        st.session_state.chat_history = []
        st.session_state.pending_draft_message = ""
        st.rerun()

    user_text = st.session_state.draft_message.strip() if send_draft else ""
    if user_text:
        stream_chunks: List[str] = []
        if st.session_state.live_token_stream:
            with live_response_area:
                with st.chat_message("assistant"):
                    stream_placeholder = st.empty()

        def _on_stream(delta: str) -> None:
            if stream_placeholder is None:
                return
            if not delta:
                return
            stream_chunks.append(delta)
            with stream_placeholder.container():
                _render_markdown_with_math("".join(stream_chunks))

        def _on_stage(stage: str) -> None:
            stage_placeholder.info(f"Research pipeline: {stage}")

        # Keep retrieval context stable across modes; fast mode mainly reduces follow-up work.
        top_k_chunks = 12
        neighbor_radius = 2
        include_suggestions = True
        allow_external_fallback = True
        _on_stage("Gathering evidence")
        evidence_debug: Dict[str, int] = {}
        with live_response_area:
            with st.spinner("Research pipeline: gathering evidence..."):
                payload = service.chat(
                    user_message=user_text,
                    steer=0.0,
                    steering_mode=STEERING_OPTIONS[steering_label],
                    top_k_chunks=top_k_chunks,
                    neighbor_radius=neighbor_radius,
                    include_suggestions=include_suggestions,
                    allow_external_fallback=allow_external_fallback,
                    stream_handler=_on_stream if st.session_state.live_token_stream else None,
                    stage_handler=_on_stage,
                )
                evidence_rows = _build_evidence_rows(service, payload, debug_counts=evidence_debug)
                if UI_DEV_DEBUG:
                    stage_placeholder.caption(
                        "Evidence debug: "
                        f"section cits={evidence_debug.get('section_citations', 0)}, "
                        f"global cits={evidence_debug.get('global_citations', 0)}, "
                        f"rows(section)={evidence_debug.get('rows_from_sections', 0)}, "
                        f"rows(after global)={evidence_debug.get('rows_after_global_fallback', 0)}, "
                        f"rows(final)={evidence_debug.get('rows_final', 0)}"
                    )
        stage_placeholder.empty()
        if stream_placeholder is not None:
            stream_placeholder.empty()
        st.session_state.chat_history.extend(
            [
                {"role": "user", "content": user_text},
                {
                    "role": "assistant",
                    "payload": payload,
                    "evidence_rows": evidence_rows,
                },
            ]
        )
        st.session_state.pending_draft_message = ""
        st.rerun()


def _stage_logs_md(result: Dict[str, Any]) -> str:
    stage_logs = []
    for stg in result.get("stage_results", []) or []:
        stage_logs.append(
            f"### {stg.get('module')}\n"
            f"- ok: `{stg.get('ok')}`\n"
            f"- return_code: `{stg.get('return_code')}`\n"
            f"- elapsed_seconds: `{stg.get('elapsed_seconds')}`\n"
            f"```\n{stg.get('output_tail', '')}\n```"
        )
    return "\n\n".join(stage_logs) if stage_logs else "No stage logs."


def _render_ingest_tab(service: AtlasService) -> None:
    st.markdown(
        "Help grow Alignment Atlas by proposing a source.\n\n"
        "Add your proposed paper, blog post, or technical write-up. We first evaluate quality and relevance; "
        "if it passes, we ingest it, extract claims, add relations, and connect it into the graph."
    )

    title = st.text_input(
        "Title (optional)",
        placeholder="Optional: leave blank to auto-generate from link",
    )
    source_url = st.text_input("Source URL (paper or blog)", placeholder="https://...")
    if source_url.strip() != str(st.session_state.ingest_guardrail_preview_url or "").strip():
        st.session_state.ingest_guardrail_preview = None
        st.session_state.ingest_guardrail_preview_url = source_url.strip()
        st.session_state.ingest_review_override = False

    if st.button("Check paper quality", width="stretch"):
        if not source_url.strip():
            st.warning("Please provide a source URL first.")
        elif not _is_valid_ingest_url(source_url.strip()):
            st.warning("Please provide a valid http(s) URL (not a question or plain text).")
        else:
            with st.spinner("Evaluating quality and relevance..."):
                preview = service.evaluate_ingest_candidate(
                    title=title.strip(),
                    source_url=source_url.strip(),
                    source_type="auto",
                    year=None,
                )
                st.session_state.ingest_guardrail_preview = preview
                st.session_state.ingest_guardrail_preview_url = source_url.strip()

    preview = st.session_state.ingest_guardrail_preview or {}
    if preview:
        decision_obj = preview.get("decision", {}) or {}
        decision = str(decision_obj.get("decision", "review")).strip().lower()
        tier = str(decision_obj.get("tier", "somewhat_relevant"))
        conf = float(decision_obj.get("confidence", 0.0))
        if decision == "allow":
            st.success(f"Pre-check: {tier} - {decision} (confidence {conf:.2f})")
        elif decision == "review":
            st.warning(f"Pre-check: {tier} - {decision} (confidence {conf:.2f})")
        else:
            st.error(f"Pre-check: {tier} - {decision} (confidence {conf:.2f})")
        st.caption(str(decision_obj.get("reasoning", "")).strip())
        signals = preview.get("signals", {}) or {}
        ss = (signals.get("semantic_scholar") or {}) if isinstance(signals, dict) else {}
        cols = st.columns([1, 1, 1, 1])
        cols[0].metric("Citations", int(signals.get("citation_count") or 0))
        cols[1].metric("Influential citations", int(ss.get("influential_citation_count") or 0))
        cols[2].metric("Trusted domain", "Yes" if signals.get("trusted_domain") else "No")
        cols[3].metric("Lookup", "Found" if ss.get("ok") else "Not found")
        flags = decision_obj.get("flags", []) or []
        if flags:
            st.caption("Flags: " + ", ".join([str(f) for f in flags]))
        if decision == "review":
            st.checkbox(
                "I reviewed this source and approve ingest",
                key="ingest_review_override",
                help="Manual reviewer approval for review-tier sources only.",
            )

    if st.button("Ingest Into Graph", type="primary"):
        if not source_url.strip():
            st.warning("Please provide a source URL.")
        elif not _is_valid_ingest_url(source_url.strip()):
            st.warning("Please provide a valid http(s) URL before ingesting.")
        else:
            # Enforce quality gate before creating a background ingest job.
            preview = st.session_state.ingest_guardrail_preview or {}
            preview_url = str(st.session_state.ingest_guardrail_preview_url or "").strip()
            if preview_url != source_url.strip() or not preview:
                with st.spinner("Running pre-ingest quality gate..."):
                    preview = service.evaluate_ingest_candidate(
                        title=title.strip(),
                        source_url=source_url.strip(),
                        source_type="auto",
                        year=None,
                    )
                st.session_state.ingest_guardrail_preview = preview
                st.session_state.ingest_guardrail_preview_url = source_url.strip()
            decision_obj = (preview or {}).get("decision", {}) or {}
            decision = str(decision_obj.get("decision", "review")).strip().lower()
            approved_review = decision == "review" and bool(st.session_state.ingest_review_override)
            if decision != "allow" and not approved_review:
                if decision == "review":
                    st.error(
                        "Ingest is in `review`. Check 'I reviewed this source and approve ingest' to proceed."
                    )
                else:
                    st.error(
                        "Ingest blocked by quality gate. "
                        f"Decision: `{decision}`. Only `allow` (or approved `review`) can start ingest."
                    )
                return
            if decision == "review" and approved_review:
                st.info("Reviewer override applied: proceeding with ingest for `review` decision.")
            if decision not in {"allow", "review"}:
                st.error(
                    "Ingest blocked by quality gate. "
                    f"Decision: `{decision}`. Only `allow` (or approved `review`) can start ingest."
                )
                return
            job = service.start_ingest_job(
                title=title.strip(),
                source_url=source_url.strip(),
                source_type="auto",
                year=None,
                run_relations=True,
                incremental=True,
                allow_review_override=bool(st.session_state.ingest_review_override),
            )
            st.session_state.ingest_job_id = job.get("job_id")
            st.session_state.ingest_live_progress = {}
            st.session_state.ingest_status = "Ingest job started."
            st.rerun()

    if st.session_state.ingest_job_id:
        st.markdown("### Ingest Progress")
        progress_box = st.empty()
        detail_box = st.empty()
        status_box = st.empty()
        claim_box = st.empty()
        job = service.get_ingest_job(st.session_state.ingest_job_id)
        if not job:
            status_box.error("Ingest job disappeared. Please start again.")
        else:
            status = str(job.get("status", "unknown"))
            progress = job.get("progress", {}) or {}
            current_stage = str(progress.get("current_stage", status))
            stage_idx = int(progress.get("stage_index", 0) or 0)
            stage_total = int(progress.get("stage_total", 0) or 0)
            updated_at = float(job.get("updated_at", 0.0) or 0.0)
            age_sec = max(0.0, time.time() - updated_at) if updated_at else 0.0
            progress_events = int(job.get("progress_event_count", 0) or 0)
            if status in {"completed", "failed"}:
                frac = 1.0
            else:
                frac = (stage_idx / stage_total) if stage_total > 0 else (0.0 if status == "queued" else 0.2)
            progress_box.progress(min(1.0, max(0.0, frac)))
            finalizing_hint = ""
            if status == "running" and stage_total > 0 and stage_idx >= stage_total:
                finalizing_hint = " | Finalizing ingest result..."
            status_box.info(
                f"Status: **{status}** | Stage: `{current_stage}` ({stage_idx}/{stage_total}){finalizing_hint}"
            )
            st.caption(
                f"Last progress update: {age_sec:.1f}s ago | Progress events: {progress_events}"
            )
            if status == "running" and age_sec > 15:
                st.warning(
                    "No progress events for more than 15 seconds. "
                    "If this keeps increasing, check terminal output for claim heartbeats."
                )

            stage_rows = progress.get("stage_results", []) or []
            if stage_rows:
                lines = []
                for i, stg in enumerate(stage_rows, 1):
                    ok = "ok" if stg.get("ok") else "failed"
                    lines.append(f"{i}. `{stg.get('module')}` - **{ok}** ({stg.get('elapsed_seconds')}s)")
                detail_box.markdown("#### Completed stages\n" + "\n".join(lines))
            else:
                detail_box.caption("No completed stages yet.")

            claim_done = progress.get("claim_calls_done")
            claim_total = progress.get("claim_calls_total")
            claim_written = progress.get("claims_written")
            detail_msg = str(progress.get("detail_message", "") or "").strip()
            if claim_done is not None and claim_total is not None:
                claim_box.info(
                    f"Claim extraction progress: call {claim_done}/{claim_total}"
                    + (f" | claims written: {claim_written}" if claim_written is not None else "")
                )
            elif detail_msg:
                claim_box.caption(detail_msg)
            else:
                claim_box.empty()

            if status in {"completed", "failed"}:
                result = job.get("result") or {}
                if result:
                    st.session_state.ingest_result_json = json.dumps(result, indent=2, ensure_ascii=False)
                    st.session_state.ingest_stage_logs = _stage_logs_md(result)
                    st.session_state.ingest_status = (
                        "Ingest succeeded." if result.get("ok") else f"Ingest failed: {result.get('error')}"
                    )
                elif job.get("error"):
                    st.session_state.ingest_status = f"Ingest failed: {job.get('error')}"
            else:
                time.sleep(1.0)
                st.rerun()

    if st.session_state.ingest_status:
        st.write(st.session_state.ingest_status)
        st.code(st.session_state.ingest_result_json, language="json")
        st.markdown(st.session_state.ingest_stage_logs)

def run() -> None:
    st.set_page_config(page_title="Alignment Atlas", page_icon="🧭", layout="wide")
    _inject_ui_theme()
    _init_state()
    service = get_service()

    st.title("Alignment Atlas")
    chat_tab, ingest_tab = st.tabs(["Chat", "Ingest"])
    with chat_tab:
        _render_chat_tab(service)
    with ingest_tab:
        _render_ingest_tab(service)

