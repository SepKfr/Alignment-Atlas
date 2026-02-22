# src/retrieval/generate_answer_openai.py
"""
Stage 9 — Generate a Britannica-style answer from an evidence pack using OpenAI (Responses API + Structured Outputs)

Inputs:
- evidence_pack dict produced by Stage 8 (AlignmentAtlasRetriever.build_evidence_pack)

Outputs:
- dict with:
    - title
    - summary
    - key_points (bullets with citations)
    - debates_and_contradictions (bullets with citations)
    - limitations
    - citations (normalized list)

This file provides:
- generate_britannica_answer(evidence_pack, ...)
- a small CLI to test:
    python -m src.retrieval.generate_answer_openai --question "What is reward hacking?"

Env:
  export OPENAI_API_KEY="..."
Optional:
  export ANSWER_MODEL="gpt-4o-mini"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

# ---- Defaults ----
DEFAULT_MODEL = os.environ.get("ANSWER_MODEL", "gpt-4o-mini")
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "answers"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- Structured output schema ----
ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string", "minLength": 300},
        "key_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": ["chunk", "claim", "external"]},
                                "id": {"type": "string"},
                            },
                            "required": ["kind", "id"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["point", "citations"],
                "additionalProperties": False,
            },
        },
        "debates_and_contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "debate": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": ["chunk", "claim", "external"]},
                                "id": {"type": "string"},
                            },
                            "required": ["kind", "id"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["debate", "citations"],
                "additionalProperties": False,
            },
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["chunk", "claim", "external"]},
                    "id": {"type": "string"},
                    "doc_id": {"type": "string"},
                    "section": {"type": "string"},
                    "snippet": {"type": "string"},
                },
                "required": ["kind", "id", "doc_id", "section", "snippet"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "summary", "key_points", "debates_and_contradictions", "limitations", "citations"],
    "additionalProperties": False,
}


REPHRASE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "key_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"point": {"type": "string"}},
                "required": ["point"],
                "additionalProperties": False,
            },
        },
        "debates_and_contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"debate": {"type": "string"}},
                "required": ["debate"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["key_points", "debates_and_contradictions"],
    "additionalProperties": False,
}


def _looks_fragmentary(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if t[0].islower():
        return True
    first = t.split()[0].strip(".,;:!?()[]{}\"'").lower()
    if len(first) <= 2 and first not in {"ai", "ml", "rl"}:
        return True
    return False


def _normalize_bullet_sentence(text: str, *, prefix: str) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip(" \t\n\r-•")
    if not t:
        return f"{prefix} unavailable."
    # If the model still returns a fragment-like start, enforce readable wrapper.
    if _looks_fragmentary(t):
        t = f"{prefix} {t}"
    t = t[0].upper() + t[1:] if t else t
    if t and t[-1] not in ".!?":
        t += "."
    return t


def _maybe_rephrase_bullets(
    client: OpenAI,
    *,
    model: str,
    answer_obj: Dict[str, Any],
) -> Dict[str, Any]:
    key_points = answer_obj.get("key_points", []) or []
    debates = answer_obj.get("debates_and_contradictions", []) or []
    if not key_points and not debates:
        return answer_obj

    needs_rephrase = False
    for kp in key_points:
        if _looks_fragmentary(str(kp.get("point", ""))):
            needs_rephrase = True
            break
    if not needs_rephrase:
        for d in debates:
            if _looks_fragmentary(str(d.get("debate", ""))):
                needs_rephrase = True
                break
    if not needs_rephrase:
        return answer_obj

    system = (
        "You are editing bullet text for readability. "
        "Rewrite each bullet into a complete, clean, standalone sentence. "
        "Do not add new facts. Keep meaning and scope unchanged."
    )
    user = json.dumps(
        {
            "instructions": [
                "Rewrite only wording; preserve meaning.",
                "Make each bullet a complete sentence.",
                "No sentence fragments or mid-sentence starts.",
                "Do not add claims not present in the original bullet.",
                "Return same number/order of bullets.",
            ],
            "key_points": [{"point": str(k.get("point", ""))} for k in key_points],
            "debates_and_contradictions": [{"debate": str(d.get("debate", ""))} for d in debates],
        },
        ensure_ascii=False,
    )
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "answer_bullet_rephrase",
                    "strict": True,
                    "schema": REPHRASE_SCHEMA,
                }
            },
            temperature=0.0,
        )
        raw = resp.output_text
        if not raw:
            return answer_obj
        data = json.loads(raw)
        new_kp = data.get("key_points", []) or []
        new_db = data.get("debates_and_contradictions", []) or []
        if len(new_kp) != len(key_points) or len(new_db) != len(debates):
            return answer_obj
        for i, item in enumerate(new_kp):
            cleaned = _normalize_bullet_sentence(
                str(item.get("point", "")),
                prefix="Key finding:",
            )
            answer_obj["key_points"][i]["point"] = cleaned
        for i, item in enumerate(new_db):
            cleaned = _normalize_bullet_sentence(
                str(item.get("debate", "")),
                prefix="Debate point:",
            )
            answer_obj["debates_and_contradictions"][i]["debate"] = cleaned
        return answer_obj
    except Exception:
        return answer_obj


def _build_evidence_digest(
    evidence_pack: Dict[str, Any],
    *,
    max_chunks: int = 10,
    max_claims: int = 18,
    max_relations: int = 14,
) -> Dict[str, Any]:
    """
    Create a smaller, model-friendly evidence object:
    - keeps full atlas evidence text (no truncation)
    - keeps ids needed for citation
    """
    chunks = evidence_pack.get("chunks", []) or []
    claims = evidence_pack.get("claims", []) or []
    relations = evidence_pack.get("relations", {}) or {}
    external_results = evidence_pack.get("external_results", []) or []

    # Keep top chunks by score (already sorted), but cap
    chunks_small = []
    for c in chunks[:max_chunks]:
        chunks_small.append(
            {
                "chunk_id": int(c["chunk_id"]),
                "doc_id": str(c.get("doc_id", "")),
                "section": str(c.get("section", "unknown")),
                "score": float(c.get("score", 0.0)),
                "text": str(c.get("text", "")),
            }
        )

    # Keep higher-confidence claims first
    claims_sorted = sorted(claims, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    claims_small = []
    for cl in claims_sorted[:max_claims]:
        claims_small.append(
            {
                "claim_id": str(cl["claim_id"]),
                "doc_id": str(cl.get("doc_id", "")),
                "chunk_id": int(cl.get("chunk_id", -1)),
                "section": str(cl.get("section", "unknown")),
                "confidence": float(cl.get("confidence", 0.0)),
                "claim": str(cl.get("claim", "")),
            }
        )

    def take_rels(rel_key: str) -> List[Dict[str, Any]]:
        items = relations.get(rel_key, []) or []
        # sort by confidence desc
        items = sorted(items, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        out = []
        for r in items[:max_relations]:
            out.append(
                {
                    "src": str(r.get("src", "")),
                    "dst": str(r.get("dst", "")),
                    "confidence": float(r.get("confidence", 0.0)),
                    "justification": str(r.get("justification", "")),
                }
            )
        return out

    return {
        "question": str(evidence_pack.get("question", "")),
        "chunks": chunks_small,
        "claims": claims_small,
        "external_results": [
            {
                "ext_id": str(r.get("id", "")),
                "source_type": str(r.get("source_type", "external_scholarly")),
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "snippet": str(r.get("snippet", "")),
                "score": float(r.get("score", 0.0)),
            }
            for r in external_results
            if str(r.get("id", "")).strip()
        ],
        "relations": {
            "entails": take_rels("entails"),
            "contradiction": take_rels("contradiction"),
        },
    }


def _perspective_instructions(perspective: str) -> str:
    """
    Controls "steering" at answer time: what to emphasize and how to frame tradeoffs.
    """
    p = (perspective or "balanced").lower().strip()
    if p in {"safety_first", "safety"}:
        return (
            "Perspective: safety-first. Emphasize failure modes, misuse/deception risks, and mitigations. "
            "Prefer conservative claims; include caveats."
        )
    if p in {"interpretability_first", "interpretability"}:
        return (
            "Perspective: interpretability-first. Emphasize mechanistic explanations, observability, and "
            "how interpretability helps assurance. Include limitations."
        )
    if p in {"practical_deployment", "deployment", "product"}:
        return (
            "Perspective: practical deployment. Emphasize monitoring, evaluation, red-teaming, "
            "and operational guidance."
        )
    return "Perspective: balanced. Cover definitions, main arguments, evidence, and tradeoffs."


def _extract_json_string_value_partial(text: str, key: str) -> Optional[str]:
    marker = f'"{key}"'
    i = text.find(marker)
    if i < 0:
        return None
    j = text.find(":", i + len(marker))
    if j < 0:
        return None
    k = text.find('"', j + 1)
    if k < 0:
        return None

    out_chars: List[str] = []
    esc = False
    p = k + 1
    while p < len(text):
        ch = text[p]
        if esc:
            # Keep escaped char as-is for a readable partial preview.
            out_chars.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            break
        else:
            out_chars.append(ch)
        p += 1
    return "".join(out_chars)


def _contains_latex_math(text: str) -> bool:
    t = str(text or "")
    if not t.strip():
        return False
    if "$$" in t:
        return True
    if re.search(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)", t, flags=re.S):
        return True
    if "\\(" in t or "\\[" in t:
        return True
    return False


def _maybe_add_formula_fallback(answer_obj: Dict[str, Any], *, question: str) -> Dict[str, Any]:
    """
    Last-resort formula insertion for explicit formula asks when the model produced no math.
    Keeps scope narrow to reward-modeling style asks.
    """
    q = (question or "").strip().lower()
    wants_reward_model_formula = (
        ("reward model" in q)
        or ("reward modeling" in q)
        or ("preference model" in q)
        or ("preference learning" in q)
    )
    if not wants_reward_model_formula:
        return answer_obj

    summary = str(answer_obj.get("summary", "") or "")
    if _contains_latex_math(summary):
        return answer_obj
    for kp in answer_obj.get("key_points", []) or []:
        if _contains_latex_math(str(kp.get("point", "") or "")):
            return answer_obj

    formula_block = (
        "\n\nMathematical form (pairwise reward modeling / Bradley-Terry):\n"
        "$$P(y_w \\succ y_l\\mid x)=\\sigma\\left(r_{\\theta}(x,y_w)-r_{\\theta}(x,y_l)\\right)$$\n"
        "$$\\mathcal{L}(\\theta)=-\\mathbb{E}_{(x,y_w,y_l)}\\left[\\log\\sigma\\left(r_{\\theta}(x,y_w)-r_{\\theta}(x,y_l)\\right)\\right]$$"
    )
    answer_obj["summary"] = summary.rstrip() + formula_block
    return answer_obj


def generate_britannica_answer(
    evidence_pack: Dict[str, Any],
    *,
    model: str,
    perspective: str = "balanced",
    steering_profile: Optional[Dict[str, Any]] = None,  # <-- add
    formula_requested: bool = False,
    answer_mode: str = "balanced",
    temperature: float = 0.2,
    max_chunks: int = 10,
    max_claims: int = 18,
    max_relations: int = 14,
    client: Optional["OpenAI"] = None,
    stream_handler: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Returns a structured Britannica-style answer with citations.

    Citations rule:
    - cite using {"kind":"chunk","id":"<chunk_id as string>"} or {"kind":"claim","id":"<claim_id>"}
    - All bullets must have at least 1 citation.
    """
    client = client or OpenAI()

    digest = _build_evidence_digest(
        evidence_pack,
        max_chunks=max_chunks,
        max_claims=max_claims,
        max_relations=max_relations,
    )

    profile = steering_profile or {
        "label": perspective,
        "tone": "neutral, encyclopedic",
        "emphasis": "Be conservative, grounded, and cite evidence.",
        "extras": [],
        "cite_pressure": "medium",
        "steer": 0.0,
    }

    mode = (answer_mode or "balanced").strip().lower()
    if mode == "expansive":
        response_mode_instruction = (
            "Response mode: expansive. Provide deeper explanation, unpack assumptions, and connect implications "
            "across cited evidence while staying grounded."
        )
    elif mode == "strict":
        response_mode_instruction = (
            "Response mode: strict. Keep claims tightly tied to explicit evidence and avoid broader inference."
        )
    else:
        response_mode_instruction = (
            "Response mode: balanced. Be grounded first, but provide useful explanatory inference when supported."
        )

    system = (
        "You write Britannica-style entries grounded in the provided evidence pack.\n"
        "Rules:\n"
        "1) Do not invent facts or citations.\n"
        "2) Every key point must have at least one citation.\n"
        "3) If evidence is weak, say so in Limitations.\n"
        "4) If there are contradictions/uncertainties, surface them explicitly and state what each side claims.\n"
        "5) Source hierarchy is mandatory: atlas chunk/claim evidence first, external scholarly second, external web last.\n"
        "6) If external evidence is used, label uncertainty and keep it subordinate to atlas evidence.\n\n"
        "7) You may infer implications from cited evidence, but do not invent new facts, numbers, experiments, or citations.\n"
        "8) If the user explicitly asks for a formula/equation, include at least one LaTeX equation in the summary.\n"
        "9) Prefer equations grounded in cited evidence. If no explicit equation is present but the method has a "
        "widely used canonical objective, you may include that canonical form and label it as canonical.\n"
        "10) Use Markdown-compatible LaTeX delimiters: inline $...$ or block $$...$$.\n\n"
        "11) The summary must be substantial (around 120-200 words), coherent, and rewritten in clean prose.\n"
        "12) Do not copy rough fragment text from chunks verbatim; synthesize and normalize phrasing.\n\n"
        f"{response_mode_instruction}\n\n"
        f"STEERING PROFILE (controls tone + emphasis; do not change factual grounding):\n"
        f"- label: {profile.get('label')}\n"
        f"- steer: {profile.get('steer')}\n"
        f"- tone: {profile.get('tone')}\n"
        f"- emphasis: {profile.get('emphasis')}\n"
        f"- extras to include if relevant: {profile.get('extras')}\n"
        f"- citation strictness: {profile.get('cite_pressure')}\n"
    )
    user = json.dumps(
        {
            "task": "Write a Britannica-style entry answering the question using ONLY the evidence provided.",
            "style": [
                "Clear, factual, and as detailed as needed for the user's intent.",
                "No hype, no personal opinions.",
                "Prefer claim-level citations when available; otherwise chunk-level.",
                "Do not invent paper names, results, or metrics not in evidence.",
                "When helpful, explain implications inferred from the evidence and present uncertainty explicitly.",
                "Write an expanded opening summary before bullet sections (not just 1-2 short sentences).",
                "If formula_requested is true, include at least one LaTeX equation in the summary.",
            ],
            "citation_rules": [
                'Cite chunks by {"kind":"chunk","id":"<chunk_id>"} (chunk_id is an integer rendered as a string).',
                'Cite claims by {"kind":"claim","id":"<claim_id>"} (exact claim_id from evidence).',
                'Cite external items by {"kind":"external","id":"<ext_id>"} (exact ext_id from external_results).',
                "Citations must refer to IDs present in the evidence digest.",
            ],
            "perspective": _perspective_instructions(perspective),
            "formula_requested": bool(formula_requested),
            "evidence_digest": digest,
        },
        ensure_ascii=False,
    )

    request_kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "britannica_answer",
                "strict": True,
                "schema": ANSWER_SCHEMA,
            }
        },
        "temperature": temperature,
    }

    raw = ""
    if stream_handler is None:
        resp = client.responses.create(**request_kwargs)
        raw = resp.output_text
    else:
        json_buffer = ""
        emitted_summary_len = 0
        with client.responses.stream(**request_kwargs) as stream:
            for event in stream:
                if getattr(event, "type", "") == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    json_buffer += delta
                    summary_partial = _extract_json_string_value_partial(json_buffer, "summary")
                    if summary_partial is not None and len(summary_partial) > emitted_summary_len:
                        stream_handler(summary_partial[emitted_summary_len:])
                        emitted_summary_len = len(summary_partial)
            final_resp = stream.get_final_response()
            raw = final_resp.output_text

    if not raw:
        raise RuntimeError("Empty output_text from model.")
    out = json.loads(raw)
    out = _maybe_rephrase_bullets(client, model=model, answer_obj=out)
    if formula_requested:
        out = _maybe_add_formula_fallback(out, question=str(evidence_pack.get("question", "")))

    # Optional: validate citations refer to known IDs (light sanity)
    known_chunk_ids = {str(c["chunk_id"]) for c in digest["chunks"]}
    known_claim_ids = {c["claim_id"] for c in digest["claims"]}
    known_external_ids = {str(c.get("ext_id", "")) for c in (digest.get("external_results", []) or [])}

    def _filter_citations(cits: List[Dict[str, str]]) -> List[Dict[str, str]]:
        good = []
        for c in cits or []:
            kind = c.get("kind")
            cid = c.get("id")
            if kind == "chunk" and cid in known_chunk_ids:
                good.append(c)
            elif kind == "claim" and cid in known_claim_ids:
                good.append(c)
            elif kind == "external" and cid in known_external_ids:
                good.append(c)
        return good

    for kp in out.get("key_points", []) or []:
        kp["citations"] = _filter_citations(kp.get("citations", []))
    for d in out.get("debates_and_contradictions", []) or []:
        d["citations"] = _filter_citations(d.get("citations", []))

    # Normalize citations list by filling doc_id/section/snippet from digest when possible
    # (Keeps your UI simple: you can display a snippet for each citation.)
    chunk_lookup = {str(c["chunk_id"]): c for c in digest["chunks"]}
    claim_lookup = {c["claim_id"]: c for c in digest["claims"]}
    external_lookup = {str(c["ext_id"]): c for c in (digest.get("external_results", []) or [])}

    normalized = []
    seen = set()
    citation_pool: List[Dict[str, str]] = []
    citation_pool.extend(out.get("citations", []) or [])
    for kp in out.get("key_points", []) or []:
        citation_pool.extend(kp.get("citations", []) or [])
    for d in out.get("debates_and_contradictions", []) or []:
        citation_pool.extend(d.get("citations", []) or [])

    for c in citation_pool:
        kind = c.get("kind")
        cid = c.get("id")
        key = (kind, cid)
        if key in seen:
            continue
        seen.add(key)

        if kind == "chunk" and cid in chunk_lookup:
            ch = chunk_lookup[cid]
            normalized.append(
                {
                    "kind": "chunk",
                    "id": cid,
                    "doc_id": ch["doc_id"],
                    "section": ch["section"],
                    "snippet": ch["text"],
                    "source_type": "atlas",
                    "url": None,
                    "title": ch["doc_id"],
                }
            )
        elif kind == "claim" and cid in claim_lookup:
            cl = claim_lookup[cid]
            normalized.append(
                {
                    "kind": "claim",
                    "id": cid,
                    "doc_id": cl["doc_id"],
                    "section": cl["section"],
                    "snippet": cl["claim"],
                    "source_type": "atlas",
                    "url": None,
                    "title": cl["doc_id"],
                }
            )
        elif kind == "external" and cid in external_lookup:
            ex = external_lookup[cid]
            normalized.append(
                {
                    "kind": "external",
                    "id": cid,
                    "doc_id": "",
                    "section": ex.get("source_type", "external_scholarly"),
                    "snippet": ex.get("snippet", ""),
                    "source_type": ex.get("source_type", "external_scholarly"),
                    "url": ex.get("url"),
                    "title": ex.get("title", "External source"),
                }
            )

    out["citations"] = normalized
    out["_meta"] = {
        "model": model,
        "perspective": perspective,
        "formula_requested": bool(formula_requested),
        "temperature": temperature,
        "external_results_count": len(digest.get("external_results", []) or []),
        "digest_limits": {"max_chunks": max_chunks, "max_claims": max_claims, "max_relations": max_relations},
    }
    return out


# ---------------- CLI for quick testing ----------------

def _save_answer(answer: Dict[str, Any], question: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", question.strip().lower()).strip("_")[:80] or "question"
    path = OUT_DIR / f"{slug}.json"
    path.write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--perspective", type=str, default="balanced")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k-chunks", type=int, default=8)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    args = parser.parse_args()

    # Lazy import to avoid circular imports if you reorganize modules
    from src.retrieval.retriever import AlignmentAtlasRetriever as Retriever

    r = Retriever()
    evidence = r.build_evidence_pack(
        args.question,
        top_k_chunks=args.top_k_chunks,
        neighbor_radius=args.neighbor_radius,
        max_claim_rel_per_claim=4,
    )

    ans = generate_britannica_answer(
        evidence,
        model=args.model,
        perspective=args.perspective,
        temperature=args.temperature,
    )

    out_path = _save_answer(ans, args.question)
    print(f"Saved answer -> {out_path}")

    # Print a human-readable preview
    print("\nTITLE:", ans["title"])
    print("\nSUMMARY:", ans["summary"])
    print("\nKEY POINTS:")
    for i, kp in enumerate(ans["key_points"], 1):
        cits = ", ".join([f"{c['kind']}:{c['id']}" for c in kp.get("citations", [])])
        print(f"  {i}. {kp['point']} [{cits}]")

    if ans["debates_and_contradictions"]:
        print("\nDEBATES / CONTRADICTIONS:")
        for i, d in enumerate(ans["debates_and_contradictions"], 1):
            cits = ", ".join([f"{c['kind']}:{c['id']}" for c in d.get("citations", [])])
            print(f"  {i}. {d['debate']} [{cits}]")

    if ans["limitations"]:
        print("\nLIMITATIONS:")
        for i, l in enumerate(ans["limitations"], 1):
            print(f"  {i}. {l}")


if __name__ == "__main__":
    main()
