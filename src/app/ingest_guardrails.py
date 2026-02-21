from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests
from openai import OpenAI

GUARDRAIL_MODEL = os.environ.get("INGEST_GUARDRAIL_MODEL", "gpt-4o-mini")
SEMANTIC_SCHOLAR_TIMEOUT_SECONDS = float(os.environ.get("SEMANTIC_SCHOLAR_TIMEOUT_SECONDS", "8"))
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()

TRUSTED_DOMAINS = {
    "arxiv.org",
    "openai.com",
    "anthropic.com",
    "deepmind.google",
    "neurips.cc",
    "icml.cc",
    "iclr.cc",
    "aclweb.org",
    "aclanthology.org",
}


GUARDRAIL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "tier": {"type": "string", "enum": ["highly_relevant", "somewhat_relevant", "unrelated"]},
        "decision": {"type": "string", "enum": ["allow", "review", "reject"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
        "flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tier", "decision", "confidence", "reasoning", "flags"],
    "additionalProperties": False,
}


def _extract_domain(url: str) -> str:
    t = (url or "").strip().lower()
    t = re.sub(r"^https?://", "", t)
    t = t.split("/", 1)[0]
    if t.startswith("www."):
        t = t[4:]
    return t


def _is_trusted_domain(domain: str) -> bool:
    d = (domain or "").strip().lower()
    if not d:
        return False
    if d in TRUSTED_DOMAINS:
        return True
    return any(d.endswith("." + td) for td in TRUSTED_DOMAINS)


def _arxiv_id_from_url(url: str) -> str:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", (url or "").lower())
    if not m:
        return ""
    return m.group(1).replace(".pdf", "")


def _semantic_scholar_lookup(url: str) -> Dict[str, Any]:
    headers = {"User-Agent": "alignment-atlas/0.1"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    fields = "title,year,citationCount,influentialCitationCount,venue,url"
    result: Dict[str, Any] = {"ok": False, "source": "none"}
    try:
        u = f"{SEMANTIC_SCHOLAR_API}/URL:{quote(url, safe='')}"
        r = requests.get(
            u,
            headers=headers,
            params={"fields": fields},
            timeout=SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
        )
        if r.status_code == 200:
            data = r.json()
            result.update(
                {
                    "ok": True,
                    "source": "url",
                    "title": data.get("title"),
                    "year": data.get("year"),
                    "citationCount": data.get("citationCount"),
                    "influentialCitationCount": data.get("influentialCitationCount"),
                    "venue": data.get("venue"),
                    "url": data.get("url"),
                }
            )
            return result
    except Exception as e:
        result["error"] = f"url_lookup:{type(e).__name__}"

    arxiv_id = _arxiv_id_from_url(url)
    if not arxiv_id:
        return result
    try:
        u = f"{SEMANTIC_SCHOLAR_API}/ARXIV:{quote(arxiv_id, safe='')}"
        r = requests.get(
            u,
            headers=headers,
            params={"fields": fields},
            timeout=SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
        )
        if r.status_code == 200:
            data = r.json()
            result.update(
                {
                    "ok": True,
                    "source": "arxiv_id",
                    "title": data.get("title"),
                    "year": data.get("year"),
                    "citationCount": data.get("citationCount"),
                    "influentialCitationCount": data.get("influentialCitationCount"),
                    "venue": data.get("venue"),
                    "url": data.get("url"),
                }
            )
    except Exception as e:
        result["error"] = f"arxiv_lookup:{type(e).__name__}"
    return result


def _fallback_decision(features: Dict[str, Any]) -> Dict[str, Any]:
    citations = int(features.get("citation_count") or 0)
    trusted = bool(features.get("trusted_domain"))
    src = str(features.get("source_type", "unknown")).lower()
    flags = []
    if citations < 3:
        flags.append("low_citations")
    if not trusted:
        flags.append("untrusted_domain")
    if src not in {"pdf", "html"}:
        flags.append("unknown_source_type")

    if trusted and citations >= 20:
        return {
            "tier": "highly_relevant",
            "decision": "allow",
            "confidence": 0.7,
            "reasoning": "Trusted source with strong citation signal.",
            "flags": flags,
        }
    if trusted or citations >= 5:
        return {
            "tier": "somewhat_relevant",
            "decision": "allow",
            "confidence": 0.62,
            "reasoning": "Moderate source/citation quality; accepted with caution.",
            "flags": flags,
        }
    return {
        "tier": "unrelated",
        "decision": "reject",
        "confidence": 0.66,
        "reasoning": "Weak quality signals and low evidence of relevance.",
        "flags": flags,
    }


def evaluate_paper_candidate(
    client: OpenAI,
    *,
    title: str,
    source_url: str,
    source_type: str,
    year: Optional[int],
) -> Dict[str, Any]:
    domain = _extract_domain(source_url)
    trusted = _is_trusted_domain(domain)
    ss = _semantic_scholar_lookup(source_url)
    citation_count = int(ss.get("citationCount") or 0) if ss.get("ok") else 0
    influential = int(ss.get("influentialCitationCount") or 0) if ss.get("ok") else 0
    venue = str(ss.get("venue") or "")
    found_title = str(ss.get("title") or "")
    found_year = ss.get("year")

    features = {
        "input_title": (title or "").strip(),
        "input_year": year,
        "source_url": source_url,
        "source_type": source_type,
        "domain": domain,
        "trusted_domain": trusted,
        "semantic_scholar": {
            "ok": bool(ss.get("ok")),
            "source": ss.get("source"),
            "title": found_title,
            "year": found_year,
            "venue": venue,
            "citation_count": citation_count,
            "influential_citation_count": influential,
        },
        "citation_count": citation_count,
    }

    system = (
        "You are a strict paper-ingest quality gate for an AI alignment corpus. "
        "Classify paper relevance/quality and decide allow/review/reject."
    )
    user = json.dumps(
        {
            "task": "Classify submission quality and relevance for AI alignment corpus ingestion.",
            "tiers": {
                "highly_relevant": "Directly about AI alignment/safety, strong source and/or strong citation/venue signal.",
                "somewhat_relevant": "Related to alignment/safety but weaker evidence/quality signal.",
                "unrelated": "Not clearly about alignment/safety or low-trust/low-value source.",
            },
            "decision_policy": {
                "allow": "Accept into corpus.",
                "review": "Borderline; could be manually reviewed.",
                "reject": "Do not ingest.",
            },
            "few_shots": [
                {
                    "input": {
                        "source_url": "https://arxiv.org/pdf/1606.06565",
                        "domain": "arxiv.org",
                        "citation_count": 1000,
                        "topic_hint": "AI safety problems",
                    },
                    "output": {"tier": "highly_relevant", "decision": "allow"},
                },
                {
                    "input": {
                        "source_url": "https://randomblog.example.com/post/llm-hacks",
                        "domain": "randomblog.example.com",
                        "citation_count": 0,
                        "topic_hint": "generic AI tips",
                    },
                    "output": {"tier": "unrelated", "decision": "reject"},
                },
                {
                    "input": {
                        "source_url": "https://arxiv.org/pdf/2305.18290",
                        "domain": "arxiv.org",
                        "citation_count": 8,
                        "topic_hint": "alignment-adjacent RLHF optimization",
                    },
                    "output": {"tier": "somewhat_relevant", "decision": "allow"},
                },
            ],
            "features": features,
            "constraints": [
                "Do not hallucinate external facts.",
                "Use citation/source signals only as supporting evidence, not sole criterion.",
                "Be conservative: if unsure between somewhat and unrelated, choose review or reject.",
            ],
        },
        ensure_ascii=False,
    )

    try:
        resp = client.responses.create(
            model=GUARDRAIL_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ingest_guardrail",
                    "strict": True,
                    "schema": GUARDRAIL_SCHEMA,
                }
            },
            temperature=0.0,
        )
        raw = resp.output_text
        if not raw:
            raise RuntimeError("Empty guardrail response")
        decision = json.loads(raw)
        return {
            "ok": True,
            "model": GUARDRAIL_MODEL,
            "decision": decision,
            "signals": features,
        }
    except Exception as e:
        fallback = _fallback_decision(features)
        return {
            "ok": False,
            "model": GUARDRAIL_MODEL,
            "error": f"{type(e).__name__}: {e}",
            "decision": fallback,
            "signals": features,
        }

