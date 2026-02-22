# src/app/chat_agent.py
"""
Stage 10 — Chat orchestration layer (multi-turn) + citations + steering slider (CLI)

Run:
  PYTHONUNBUFFERED=1 uv run python -m src.app.chat_agent --steer 0.0
  PYTHONUNBUFFERED=1 uv run python -m src.app.chat_agent --steer 0.8

Env:
  export OPENAI_API_KEY="..."
Optional:
  export REWRITE_MODEL="gpt-4o-mini"
  export ANSWER_MODEL="gpt-4o-mini"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

from src.retrieval.retriever import AlignmentAtlasRetriever
from src.retrieval.generate_answer_openai import generate_britannica_answer
from src.retrieval.citations import CitationResolver
from src.retrieval.external_fallback import retrieve_external_evidence

REWRITE_MODEL = os.environ.get("REWRITE_MODEL", "gpt-4o-mini")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "gpt-4o-mini")
FALLBACK_MIN_TOP_SCORE = float(os.environ.get("ATLAS_FALLBACK_MIN_TOP_SCORE", "0.33"))
FALLBACK_MIN_MEAN_TOP3 = float(os.environ.get("ATLAS_FALLBACK_MIN_MEAN_TOP3", "0.28"))
FALLBACK_MIN_CHUNKS = int(os.environ.get("ATLAS_FALLBACK_MIN_CHUNKS", "3"))
FALLBACK_MIN_CLAIMS = int(os.environ.get("ATLAS_FALLBACK_MIN_CLAIMS", "2"))
FALLBACK_MAX_SCHOLARLY = int(os.environ.get("ATLAS_FALLBACK_MAX_SCHOLARLY", "5"))
FALLBACK_MAX_WEB = int(os.environ.get("ATLAS_FALLBACK_MAX_WEB", "2"))
ROOT = Path(__file__).resolve().parents[2]
DOCS_JSONL = ROOT / "data" / "processed" / "docs.jsonl"


# -------------------- State --------------------

@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ChatState:
    history: List[Turn] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=lambda: {
        "topic": "",
        "key_terms": [],
        "recent_doc_ids": [],
    })
    last_evidence_pack: Optional[Dict[str, Any]] = None
    last_answer: Optional[Dict[str, Any]] = None


# -------------------- Rewrite schema --------------------

REWRITE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "standalone_query": {"type": "string"},
        "intent": {
            "type": "string",
            "enum": ["explain", "compare", "find_evidence", "definition", "critique", "list_corpus", "other"],
        },
        "key_terms": {"type": "array", "items": {"type": "string"}},
        "topic": {"type": "string"},
        "answer_mode": {"type": "string", "enum": ["strict", "balanced", "expansive"]},
        "tool_plan": {
            "type": "object",
            "properties": {
                "retrieval_profile": {"type": "string", "enum": ["focused", "standard", "deep"]},
                "external_fallback_preference": {"type": "string", "enum": ["auto", "prefer", "avoid"]},
                "why": {"type": "string"},
            },
            "required": ["retrieval_profile", "external_fallback_preference", "why"],
            "additionalProperties": False,
        },
    },
    "required": ["standalone_query", "intent", "key_terms", "topic", "answer_mode", "tool_plan"],
    "additionalProperties": False,
}

SUGGEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"suggestions": {"type": "array", "items": {"type": "string"}}},
    "required": ["suggestions"],
    "additionalProperties": False,
}


def _compact_history(history: List[Turn], max_chars: int = 1600) -> str:
    pieces = []
    total = 0
    for t in reversed(history[-10:]):
        chunk = f"{t.role.upper()}: {t.content.strip()}\n"
        if total + len(chunk) > max_chars:
            break
        pieces.append(chunk)
        total += len(chunk)
    return "".join(reversed(pieces)).strip()


def rewrite_followup_to_standalone(
    client: OpenAI,
    user_message: str,
    state: ChatState,
    steering_profile: Optional[Dict[str, Any]] = None,
    *,
    model: str = REWRITE_MODEL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    history_text = _compact_history(state.history)
    mem = state.memory or {}

    system = (
        "You are a turn planner for a research assistant. "
        "You must produce both (a) a standalone retrieval query and (b) an execution plan "
        "that decides retrieval depth, fallback preference, and answer style. "
        "Choose plans from user intent, ambiguity, and question complexity. "
        "Do not add facts. "
        "If the user message is unrelated to AI alignment/safety (including greetings/smalltalk), set intent='other' "
        "and keep the standalone query minimal."
    )

    user = json.dumps(
        {
            "conversation_context": history_text,
            "memory": mem,
            "steering_profile": steering_profile or {},
            "latest_user_message": user_message,
            "output_requirements": {
                "standalone_query": "A single sentence query suitable for semantic retrieval.",
                "intent": (
                    "Classify intent (explain, compare, find_evidence, definition, critique, list_corpus, other). "
                    "Use 'other' for out-of-scope requests or greetings/smalltalk."
                ),
                "key_terms": "List of key terms (snake_case preferred).",
                "topic": "Short topic label (<= 8 words).",
                "answer_mode": "strict for citation-tight asks, balanced by default, expansive for deep/clarifying asks.",
                "tool_plan": {
                    "retrieval_profile": "focused (precise), standard, or deep (broader context for complex asks).",
                    "external_fallback_preference": "auto, prefer, or avoid.",
                    "why": "One short rationale for plan choices.",
                },
            },
        },
        ensure_ascii=False,
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "query_rewrite",
                "strict": True,
                "schema": REWRITE_SCHEMA,
            }
        },
        temperature=temperature,
    )
    raw = resp.output_text
    if not raw:
        raise RuntimeError("Empty rewrite output_text.")
    return json.loads(raw)


def generate_followup_suggestions(
    client: OpenAI,
    *,
    question: str,
    answer: Dict[str, Any],
    evidence_pack: Dict[str, Any],
    steering_profile: Optional[Dict[str, Any]] = None,
    model: str = REWRITE_MODEL,
    temperature: float = 0.35,
) -> List[str]:
    profile = steering_profile or {}
    label = str(profile.get("label", "britannica_neutral"))
    tone = str(profile.get("tone", "neutral, encyclopedic"))
    emphasis = str(profile.get("emphasis", ""))
    system = (
        "You propose useful follow-up questions for a user exploring AI alignment literature. "
        "Suggestions should be short and actionable. "
        "Adapt suggestions to the steering profile. "
        "Each suggestion must be distinct in purpose: one clarification, one evidence check, "
        "one limitations/challenge question, and one next-step/action question."
    )

    key_points = [kp.get("point", "") for kp in (answer.get("key_points", []) or [])][:6]
    debates = [d.get("debate", "") for d in (answer.get("debates_and_contradictions", []) or [])][:4]
    limitations = [str(x) for x in (answer.get("limitations", []) or [])][:4]
    doc_ids = list({c.get("doc_id", "") for c in (evidence_pack.get("chunks", []) or []) if c.get("doc_id")})[:8]

    user = json.dumps(
        {
            "question": question,
            "steering_profile": {
                "label": label,
                "tone": tone,
                "emphasis": emphasis,
            },
            "key_points": key_points,
            "debates": debates,
            "limitations": limitations,
            "doc_ids_in_context": doc_ids,
            "constraints": {
                "num_suggestions": 4,
                "no_fluff": True,
                "no_redundant_questions": True,
                "keep_each_under_16_words": True,
                "vary_opening_phrases": True,
                "must_change_with_steering_profile": True,
            },
        },
        ensure_ascii=False,
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {"type": "json_schema", "name": "followup_suggestions", "strict": True, "schema": SUGGEST_SCHEMA}
        },
        temperature=temperature,
    )

    raw = resp.output_text
    if not raw:
        return []
    obj = json.loads(raw)
    sug = obj.get("suggestions", []) or []
    cleaned = []
    for s in sug:
        s = re.sub(r"\s+", " ", str(s)).strip()
        if s and s not in cleaned:
            cleaned.append(s)
    return cleaned[:5]


# -------------------- Steering slider --------------------

def steering_profile(steer: float, mode: Optional[str] = None) -> Dict[str, Any]:
    """
    steer in [0,1]:
      0.0 -> Britannica-neutral: conservative, minimal speculation, fewer sections
      0.5 -> Practical: more mitigations + engineering framing
      1.0 -> Safety-first: emphasize risk, scaling, controls; still grounded & cited
    """
    mode_norm = (mode or "").strip().lower()
    if mode_norm == "safety_first":
        s = 0.9
        label = "safety_first"
        emphasis = "Emphasize alignment risk, scaling concerns, and control measures; be explicit about uncertainty; avoid hype."
        extras = ["risks", "scaling considerations", "controls/guardrails", "reward hacking risk"]
        cite_pressure = "high"
        tone = "serious, safety-oriented, still precise"
    elif mode_norm == "interpretability_first":
        s = 0.55
        label = "interpretability_first"
        emphasis = "Prioritize mechanistic clarity, model internals, representations, and evidence traceability."
        extras = ["mechanistic explanations", "feature-level evidence", "causal interpretability", "limitations of probes"]
        cite_pressure = "high"
        tone = "analytical, mechanistic, evidence-traceable"
    elif mode_norm == "practical_deployment":
        s = 0.45
        label = "practical_deployment"
        emphasis = "Focus on deployment constraints, monitoring, mitigations, and operational decision trade-offs."
        extras = ["deployment checklists", "evaluation metrics", "failure handling", "cost/performance trade-offs"]
        cite_pressure = "medium_high"
        tone = "practical, technical, actionable"
    else:
        s = max(0.0, min(1.0, float(steer)))

    if mode_norm:
        pass
    elif s < 0.34:
        label = "britannica_neutral"
        emphasis = "Define terms clearly; be conservative; avoid speculation; ground every key point in citations."
        extras = ["brief definitions", "high-level importance"]
        cite_pressure = "medium"
        tone = "neutral, encyclopedic"
    elif s < 0.67:
        label = "practical_engineering"
        emphasis = "Focus on concrete mechanisms, failure modes, detection and mitigations; give checklists where possible."
        extras = ["mitigations", "engineering checklist", "failure modes"]
        cite_pressure = "medium_high"
        tone = "practical, technical, actionable"
    else:
        label = "safety_first"
        emphasis = "Emphasize alignment risk, scaling concerns, and control measures; be explicit about uncertainty; avoid hype."
        extras = ["risks", "scaling considerations", "controls/guardrails"]
        cite_pressure = "high"
        tone = "serious, safety-oriented, still precise"

    return {
        "steer": s,
        "label": label,
        "emphasis": emphasis,
        "extras": extras,
        "cite_pressure": cite_pressure,
        "tone": tone,
    }


def _atlas_quality_snapshot(evidence: Dict[str, Any]) -> Dict[str, Any]:
    chunks = evidence.get("chunks", []) or []
    claims = evidence.get("claims", []) or []
    top_score = float(chunks[0].get("score", 0.0)) if chunks else 0.0
    top3 = chunks[:3]
    mean_top3 = (sum(float(c.get("score", 0.0)) for c in top3) / len(top3)) if top3 else 0.0
    return {
        "num_chunks": len(chunks),
        "num_claims": len(claims),
        "top_score": top_score,
        "mean_top3_score": mean_top3,
    }


def _should_use_external_fallback(evidence: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    stats = _atlas_quality_snapshot(evidence)
    score_fail = (stats["top_score"] < FALLBACK_MIN_TOP_SCORE) and (stats["mean_top3_score"] < FALLBACK_MIN_MEAN_TOP3)
    structure_fail = (stats["num_chunks"] < FALLBACK_MIN_CHUNKS) or (stats["num_claims"] < FALLBACK_MIN_CLAIMS)
    if score_fail and structure_fail:
        return True, "Atlas confidence and evidence coverage are both low.", stats
    if score_fail:
        return True, "Atlas retrieval scores are low for this query.", stats
    if structure_fail:
        return True, "Atlas retrieved limited supporting evidence for this query.", stats
    return False, "", stats


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


def _load_corpus_docs() -> List[Dict[str, Any]]:
    rows = _iter_jsonl(DOCS_JSONL)
    docs: List[Dict[str, Any]] = []
    for r in rows:
        doc_id = str(r.get("doc_id", "")).strip()
        if not doc_id:
            continue
        docs.append(
            {
                "doc_id": doc_id,
                "title": str(r.get("title", doc_id)).strip() or doc_id,
                "year": r.get("year"),
                "source_type": str(r.get("source_type", "unknown")).strip() or "unknown",
                "source_url": str(r.get("source_url", "")).strip(),
            }
        )
    docs.sort(key=lambda d: (str(d.get("title", "")).lower(), str(d.get("doc_id", ""))))
    return docs


def _looks_like_inventory_request(user_message: str) -> bool:
    t = (user_message or "").strip().lower()
    signals = [
        "what papers",
        "which papers",
        "list papers",
        "dataset papers",
        "corpus papers",
        "papers do you have",
        "show all papers",
        "paper list",
    ]
    return any(s in t for s in signals)


def _looks_like_formula_request(user_message: str) -> bool:
    t = (user_message or "").strip().lower()
    triggers = [
        "formula",
        "equation",
        "latex",
        "objective",
        "loss function",
        "mathematical form",
    ]
    return any(tok in t for tok in triggers)


def _build_scope_guardrail_answer() -> Dict[str, Any]:
    return {
        "title": "Alignment Atlas Scope",
        "summary": (
            "I am here to help with AI alignment and safety topics, such as reward modeling, "
            "reward hacking, oversight/evals, interpretability, and deployment risk.\n\n"
            "If you share a question in those areas, I can provide evidence-grounded answers and cite sources."
        ),
        "key_points": [],
        "debates_and_contradictions": [],
        "limitations": [],
        "citations": [],
    }


CORPUS_GROUP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "description": {"type": "string"},
                    "paper_doc_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["theme", "description", "paper_doc_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "groups"],
    "additionalProperties": False,
}


def _fallback_groups(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[str]] = {}
    for d in docs:
        by_type.setdefault(d.get("source_type", "unknown"), []).append(d["doc_id"])
    groups = []
    for k, ids in sorted(by_type.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        groups.append(
            {
                "theme": f"{str(k).upper()} sources",
                "description": f"Papers ingested from {k} sources.",
                "paper_doc_ids": ids,
            }
        )
    return groups[:6]


def _stratify_corpus_docs_with_model(
    client: OpenAI,
    *,
    user_message: str,
    docs: List[Dict[str, Any]],
    model: str = REWRITE_MODEL,
) -> Dict[str, Any]:
    system = (
        "You organize a known paper inventory into practical themes. "
        "Do not invent papers or IDs. Use only provided doc_ids."
    )
    payload = json.dumps(
        {
            "user_request": user_message,
            "docs": [
                {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "year": d.get("year"),
                    "source_type": d.get("source_type"),
                }
                for d in docs
            ],
            "requirements": {
                "num_groups": "4 to 8 themes",
                "include_all_doc_ids_if_possible": True,
                "concise_theme_descriptions": True,
            },
        },
        ensure_ascii=False,
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "corpus_inventory_groups",
                "strict": True,
                "schema": CORPUS_GROUP_SCHEMA,
            }
        },
        temperature=0.1,
    )
    raw = resp.output_text
    if not raw:
        raise RuntimeError("Empty corpus grouping output.")
    return json.loads(raw)


def _build_corpus_inventory_answer(
    *,
    user_message: str,
    docs: List[Dict[str, Any]],
    grouping: Dict[str, Any],
) -> Dict[str, Any]:
    by_id = {d["doc_id"]: d for d in docs}
    paper_lines: List[str] = []
    for d in docs:
        year = f"{d['year']}" if d.get("year") is not None else "year unknown"
        src = str(d.get("source_type", "unknown")).upper()
        if d.get("source_url"):
            paper_lines.append(f"- [{d['title']}]({d['source_url']}) - {year}, {src}")
        else:
            paper_lines.append(f"- {d['title']} - {year}, {src}")

    key_points: List[Dict[str, Any]] = []
    for g in (grouping.get("groups", []) or [])[:8]:
        ids = [str(x) for x in (g.get("paper_doc_ids", []) or []) if str(x) in by_id]
        titles = [by_id[i]["title"] for i in ids][:8]
        more = max(0, len(ids) - len(titles))
        tail = f" (+{more} more)" if more else ""
        point = f"{g.get('theme', 'Theme')}: {g.get('description', '').strip()} Papers: " + ", ".join(titles) + tail
        key_points.append({"point": point.strip(), "citations": []})

    summary_parts = [
        f"Atlas currently contains **{len(docs)} papers**.",
        grouping.get("summary", "").strip(),
        "\n### Full paper list",
        "\n".join(paper_lines),
    ]
    summary = "\n\n".join([p for p in summary_parts if p])
    return {
        "title": "Atlas Paper Inventory",
        "summary": summary,
        "key_points": key_points,
        "debates_and_contradictions": [],
        "limitations": [],
        "citations": [],
    }


def _apply_retrieval_profile(
    *,
    requested_top_k_chunks: int,
    requested_neighbor_radius: int,
    retrieval_profile: str,
) -> Tuple[int, int]:
    """
    Map model-selected retrieval profile to concrete retriever settings.
    """
    profile = (retrieval_profile or "standard").strip().lower()
    top_k = max(1, int(requested_top_k_chunks))
    radius = max(0, int(requested_neighbor_radius))
    if profile == "focused":
        return max(4, min(top_k, 6)), min(radius, 1)
    if profile == "deep":
        return max(top_k, 12), max(radius, 2)
    return top_k, radius


# -------------------- Turn function --------------------

def chat_turn(
    state: ChatState,
    user_message: str,
    *,
    steer: float = 0.0,
    steering_mode: Optional[str] = None,
    top_k_chunks: int = 12,
    neighbor_radius: int = 2,
    max_claim_rel_per_claim: int = 4,
    include_suggestions: bool = True,
    allow_external_fallback: bool = True,
    stream_handler: Optional[Callable[[str], None]] = None,
    stage_handler: Optional[Callable[[str], None]] = None,
    client: Optional[OpenAI] = None,
    retriever: Optional[AlignmentAtlasRetriever] = None,
) -> Tuple[ChatState, Dict[str, Any]]:
    client = client or OpenAI()
    retriever = retriever or AlignmentAtlasRetriever()

    profile = steering_profile(steer, mode=steering_mode)
    if stage_handler is not None:
        stage_handler("Rewriting query")
    rewrite = rewrite_followup_to_standalone(client, user_message, state, steering_profile=profile)
    standalone_query = rewrite["standalone_query"]
    intent = rewrite["intent"]
    topic = rewrite["topic"]
    key_terms = rewrite["key_terms"]
    answer_mode = rewrite.get("answer_mode", "balanced")
    tool_plan = rewrite.get("tool_plan", {}) or {}
    retrieval_profile = str(tool_plan.get("retrieval_profile", "standard"))
    external_pref = str(tool_plan.get("external_fallback_preference", "auto"))
    planned_top_k, planned_neighbor_radius = _apply_retrieval_profile(
        requested_top_k_chunks=top_k_chunks,
        requested_neighbor_radius=neighbor_radius,
        retrieval_profile=retrieval_profile,
    )

    state.memory["topic"] = topic
    state.memory["key_terms"] = list(dict.fromkeys(key_terms))[:20]

    # Scope guardrail (model-decided): rely on planner intent for out-of-scope turns.
    if intent == "other":
        if stage_handler is not None:
            stage_handler("Applying scope guardrail")
        ans = _build_scope_guardrail_answer()
        suggestions = [
            "What is reward modeling in RLHF?",
            "How do researchers detect deceptive alignment?",
            "What are strong mitigations for reward hacking?",
            "Which interpretability methods are most useful for safety?",
        ]
        state.history.append(Turn(role="user", content=user_message))
        state.history.append(Turn(role="assistant", content=(ans.get("summary", "") or "")))
        payload = {
            "rewritten_query": standalone_query,
            "intent": intent,
            "topic": topic,
            "answer_mode": answer_mode,
            "tool_plan": tool_plan,
            "steering_profile": profile,
            "evidence_status": "Scope guardrail",
            "fallback_reason": "Query appears outside Alignment Atlas scope.",
            "fallback_used": False,
            "atlas_quality": {"num_chunks": 0, "num_claims": 0, "top_score": 0.0, "mean_top3_score": 0.0},
            "external_sources_used": 0,
            "external_errors": [],
            "answer": ans,
            "suggestions": suggestions,
            "evidence_pack": {"question": user_message, "chunks": [], "claims": [], "relations": {}},
        }
        return state, payload

    if intent == "list_corpus" or _looks_like_inventory_request(user_message):
        if stage_handler is not None:
            stage_handler("Building corpus inventory")
        docs = _load_corpus_docs()
        try:
            grouping = _stratify_corpus_docs_with_model(
                client,
                user_message=user_message,
                docs=docs,
            )
        except Exception:
            grouping = {
                "summary": "Grouped using source metadata fallback.",
                "groups": _fallback_groups(docs),
            }
        ans = _build_corpus_inventory_answer(
            user_message=user_message,
            docs=docs,
            grouping=grouping,
        )
        suggestions: List[str] = []
        if include_suggestions:
            suggestions = [
                "Group these papers by reward hacking vs oversight vs interpretability.",
                "Show only papers from 2023 and later.",
                "Which papers in this list most directly contradict each other?",
                "Rank this inventory by practical deployment relevance.",
            ]
        state.history.append(Turn(role="user", content=user_message))
        state.history.append(Turn(role="assistant", content=(ans.get("summary", "") or "")))
        payload = {
            "rewritten_query": standalone_query,
            "intent": "list_corpus",
            "topic": topic,
            "answer_mode": answer_mode,
            "tool_plan": tool_plan,
            "steering_profile": profile,
            "evidence_status": "Corpus inventory",
            "fallback_reason": "",
            "fallback_used": False,
            "atlas_quality": {"num_chunks": 0, "num_claims": 0, "top_score": 0.0, "mean_top3_score": 0.0},
            "external_sources_used": 0,
            "external_errors": [],
            "answer": ans,
            "suggestions": suggestions,
            "evidence_pack": {"question": user_message, "chunks": [], "claims": [], "relations": {}},
        }
        return state, payload

    if stage_handler is not None:
        stage_handler("Retrieving evidence")
    evidence = retriever.build_evidence_pack(
        standalone_query,
        top_k_chunks=planned_top_k,
        neighbor_radius=planned_neighbor_radius,
        max_claim_rel_per_claim=max_claim_rel_per_claim,
    )
    fallback_used, fallback_reason, atlas_quality = _should_use_external_fallback(evidence)
    if external_pref == "prefer":
        fallback_used = True
        fallback_reason = "Planner preference: use external fallback to broaden context."
    elif external_pref == "avoid":
        fallback_used = False
        fallback_reason = ""
    external = {"results": [], "used_general_web": False, "errors": []}
    if fallback_used and allow_external_fallback:
        external = retrieve_external_evidence(
            standalone_query,
            max_scholarly=FALLBACK_MAX_SCHOLARLY,
            max_general=FALLBACK_MAX_WEB,
            min_scholarly_before_web=2,
        )
        if external.get("results"):
            evidence["external_results"] = external["results"]
        else:
            fallback_used = False
            fallback_reason = ""

    doc_ids = []
    for ch in evidence.get("chunks", []) or []:
        d = ch.get("doc_id")
        if d and d not in doc_ids:
            doc_ids.append(d)
    state.memory["recent_doc_ids"] = doc_ids[:12]

    if stage_handler is not None:
        stage_handler("Generating answer")
    ans = generate_britannica_answer(
        evidence,
        model=ANSWER_MODEL,
        perspective=profile["label"],          # for logging/debug
        steering_profile=profile,              # <-- NEW: passed into stage 9 prompt
        formula_requested=_looks_like_formula_request(user_message),
        answer_mode=answer_mode,
        temperature=0.2,
        max_chunks=10,
        max_claims=18,
        max_relations=14,
        client=client,
        stream_handler=stream_handler,
    )

    suggestions: List[str] = []
    if include_suggestions:
        try:
            suggestions = generate_followup_suggestions(
                client,
                question=standalone_query,
                answer=ans,
                evidence_pack=evidence,
                steering_profile=profile,
            )
        except Exception:
            suggestions = []

    state.history.append(Turn(role="user", content=user_message))
    state.history.append(Turn(role="assistant", content=(ans.get("summary", "") or "")))

    state.last_evidence_pack = evidence
    state.last_answer = ans

    payload = {
        "rewritten_query": standalone_query,
        "intent": intent,
        "topic": topic,
        "answer_mode": answer_mode,
        "tool_plan": tool_plan,
        "steering_profile": profile,
        "evidence_status": "Atlas + external fallback" if fallback_used else "Atlas grounded",
        "fallback_reason": fallback_reason,
        "fallback_used": fallback_used,
        "atlas_quality": atlas_quality,
        "external_sources_used": len((external or {}).get("results", []) or []),
        "external_errors": (external or {}).get("errors", []) or [],
        "answer": ans,
        "suggestions": suggestions,
        "evidence_pack": evidence,
    }
    return state, payload


# -------------------- Citation normalization + printing --------------------

def _normalize_citations(raw: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not raw:
        return out

    items = raw if isinstance(raw, list) else [raw]
    for it in items:
        if isinstance(it, dict):
            kind = str(it.get("kind", "")).strip()
            cid = str(it.get("id", "")).strip()
            if kind in {"claim", "chunk", "external"} and cid:
                out.append({"kind": kind, "id": cid})
            continue

        if isinstance(it, str):
            s = it.strip()
            if not s:
                continue

            if s.startswith("claim:"):
                # Normalize accidental double-prefix: "claim:claim:..." -> "claim:..."
                s2 = s
                if s2.startswith("claim:claim:"):
                    s2 = "claim:" + s2[len("claim:claim:"):]
                out.append({"kind": "claim", "id": s2})
                continue

            if s.startswith("chunk:"):
                maybe = s.split("chunk:", 1)[1].strip()
                out.append({"kind": "chunk", "id": maybe if maybe.isdigit() else s})
                continue
            if s.startswith("external:"):
                out.append({"kind": "external", "id": s.split("external:", 1)[1].strip() or s})
                continue

            out.append({"kind": "claim", "id": s})

    seen = set()
    uniq: List[Dict[str, str]] = []
    for c in out:
        key = (c["kind"], c["id"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _pretty_print_answer(ans: Dict[str, Any], resolver: CitationResolver, *, max_snippet_chars: int = 260) -> None:
    print("\n" + "=" * 80)
    print(ans.get("title", ""))
    print("=" * 80)
    summary = (ans.get("summary", "") or "").strip()
    if summary:
        print(summary, "\n")

    def resolve_cits(cits: List[Dict[str, str]]):
        resolved = resolver.resolve_many(cits, snippet_chars=max_snippet_chars)
        by_key = {(r.kind, r.id): r for r in resolved}
        return by_key

    def render_sources(cits_norm: List[Dict[str, str]], by_key: Dict[tuple, Any]) -> str:
        parts: List[str] = []
        for c in cits_norm:
            k = (c.get("kind"), c.get("id"))
            r = by_key.get(k)
            if not r:
                parts.append(f"{c.get('kind')}:{c.get('id')}")
            else:
                parts.append(f"{r.title} — {r.section}")
        return "; ".join(list(dict.fromkeys(parts)))

    print("Key points:")
    for i, kp in enumerate(ans.get("key_points", []) or [], 1):
        cits_norm = _normalize_citations(kp.get("citations", []))
        by_key = resolve_cits(cits_norm)

        print(f"  {i}. {kp.get('point','').strip()}")
        if cits_norm:
            print(f"     Sources: {render_sources(cits_norm, by_key)}")
            c0 = cits_norm[0]
            r0 = by_key.get((c0.get("kind"), c0.get("id")))
            if r0:
                print(f"     Evidence: “{r0.snippet}”")

    debates = ans.get("debates_and_contradictions", []) or []
    if debates:
        print("\nDebates / contradictions:")
        for i, d in enumerate(debates, 1):
            cits_norm = _normalize_citations(d.get("citations", []))
            by_key = resolve_cits(cits_norm)

            print(f"  {i}. {d.get('debate','').strip()}")
            if cits_norm:
                print(f"     Sources: {render_sources(cits_norm, by_key)}")

    lim = ans.get("limitations", []) or []
    if lim:
        print("\nLimitations:")
        for i, l in enumerate(lim, 1):
            print(f"  {i}. {l.strip()}")


# -------------------- CLI --------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steer", type=float, default=0.0, help="Steering slider in [0,1]. 0=neutral, 1=safety-first.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--no-suggestions", action="store_true", default=False)
    args = parser.parse_args()

    print("Alignment Atlas Chat (CLI). Type 'exit' to quit.")
    print(f"Steering: {max(0.0, min(1.0, args.steer))}\n")

    state = ChatState()
    client = OpenAI()
    retriever = AlignmentAtlasRetriever()
    resolver = CitationResolver()

    while True:
        user = input("\nYou: ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        state, payload = chat_turn(
            state,
            user,
            steer=args.steer,
            top_k_chunks=args.top_k,
            neighbor_radius=args.neighbor_radius,
            include_suggestions=(not args.no_suggestions),
            client=client,
            retriever=retriever,
        )

        _pretty_print_answer(payload["answer"], resolver)

        if payload.get("suggestions"):
            print("\nSuggested follow-ups:")
            for s in payload["suggestions"]:
                print(f"  - {s}")


if __name__ == "__main__":
    main()
