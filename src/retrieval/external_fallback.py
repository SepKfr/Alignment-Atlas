from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests


USER_AGENT = "alignment-atlas/0.1 (+https://example.com)"
REQUEST_TIMEOUT_SECONDS = 7


def _clean_text(text: str, max_chars: int = 420) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    if cut and cut[-1] not in ".!?":
        cut += "."
    return cut


def _openalex_abstract(inv: Optional[Dict[str, List[int]]]) -> str:
    if not isinstance(inv, dict) or not inv:
        return ""
    positions: Dict[int, str] = {}
    for token, idxs in inv.items():
        if not isinstance(idxs, list):
            continue
        for i in idxs:
            if isinstance(i, int):
                positions[i] = token
    if not positions:
        return ""
    words = [positions[k] for k in sorted(positions.keys())]
    return " ".join(words)


def _fetch_openalex(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    url = f"https://api.openalex.org/works?search={quote_plus(query)}&per-page={max_results}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    out: List[Dict[str, Any]] = []
    for i, item in enumerate((data.get("results") or []), 1):
        title = str(item.get("display_name") or "").strip()
        landing = (
            ((item.get("primary_location") or {}).get("landing_page_url"))
            or ((item.get("primary_location") or {}).get("pdf_url"))
            or item.get("doi")
            or item.get("id")
        )
        abstract = _openalex_abstract(item.get("abstract_inverted_index"))
        year = item.get("publication_year")
        venue = ((item.get("primary_location") or {}).get("source") or {}).get("display_name")
        prefix = f"{year} - {venue}. " if year and venue else ""
        snippet = _clean_text(prefix + (abstract or "Scholarly result relevant to the query."), max_chars=430)
        out.append(
            {
                "kind": "external",
                "id": f"ext_scholar_openalex_{i}",
                "doc_id": "",
                "section": "external_scholarly",
                "title": title or f"OpenAlex result {i}",
                "snippet": snippet,
                "url": str(landing) if landing else None,
                "source_type": "external_scholarly",
                "provider": "openalex",
                "score": max(0.0, 1.0 - (i * 0.08)),
            }
        )
    return out


def _fetch_arxiv(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{quote_plus(query)}&start=0&max_results={max_results}"
    )
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: List[Dict[str, Any]] = []
    for i, entry in enumerate(root.findall("a:entry", ns), 1):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        link = None
        for ln in entry.findall("a:link", ns):
            href = ln.attrib.get("href")
            rel = ln.attrib.get("rel", "")
            if href and rel in {"alternate", ""}:
                link = href
                break
        out.append(
            {
                "kind": "external",
                "id": f"ext_scholar_arxiv_{i}",
                "doc_id": "",
                "section": "external_scholarly",
                "title": title or f"arXiv result {i}",
                "snippet": _clean_text(summary or "arXiv result relevant to the query.", max_chars=430),
                "url": link,
                "source_type": "external_scholarly",
                "provider": "arxiv",
                "score": max(0.0, 0.8 - (i * 0.07)),
            }
        )
    return out


def _fetch_wikipedia(query: str, max_results: int = 2) -> List[Dict[str, Any]]:
    search_url = (
        "https://en.wikipedia.org/w/api.php?"
        f"action=query&list=search&srsearch={quote_plus(query)}&utf8=1&format=json&srlimit={max_results}"
    )
    resp = requests.get(search_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    out: List[Dict[str, Any]] = []
    for i, item in enumerate((data.get("query", {}) or {}).get("search", []), 1):
        title = str(item.get("title") or "").strip()
        snippet_html = str(item.get("snippet") or "")
        snippet = re.sub(r"<[^>]+>", "", snippet_html)
        out.append(
            {
                "kind": "external",
                "id": f"ext_web_wikipedia_{i}",
                "doc_id": "",
                "section": "external_web",
                "title": title or f"Wikipedia result {i}",
                "snippet": _clean_text(snippet or "General web result relevant to the query.", max_chars=380),
                "url": f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}" if title else None,
                "source_type": "external_web",
                "provider": "wikipedia",
                "score": max(0.0, 0.55 - (i * 0.06)),
            }
        )
    return out


def retrieve_external_evidence(
    query: str,
    *,
    max_scholarly: int = 5,
    max_general: int = 2,
    min_scholarly_before_web: int = 2,
) -> Dict[str, Any]:
    scholarly: List[Dict[str, Any]] = []
    general_web: List[Dict[str, Any]] = []
    errors: List[str] = []

    for provider, fn in (("openalex", _fetch_openalex), ("arxiv", _fetch_arxiv)):
        try:
            scholarly.extend(fn(query, max_results=max_scholarly))
        except Exception as e:
            errors.append(f"{provider}:{type(e).__name__}")
        if len(scholarly) >= max_scholarly:
            break

    scholarly = scholarly[:max_scholarly]
    used_general_web = len(scholarly) < min_scholarly_before_web
    if used_general_web:
        try:
            general_web = _fetch_wikipedia(query, max_results=max_general)
        except Exception as e:
            errors.append(f"wikipedia:{type(e).__name__}")

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in scholarly + general_web:
        key = (item.get("title"), item.get("url"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return {
        "results": deduped,
        "used_general_web": used_general_web,
        "errors": errors,
    }
