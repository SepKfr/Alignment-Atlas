"""Vision API client for figure explanation (OpenAI-style)."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests

from src.alignment_atlas.figures.schemas import VisionResult
from src.alignment_atlas.figures.utils import sha256_bytes

logger = logging.getLogger(__name__)

# Default endpoint: Chat Completions (vision-capable)
DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def make_cache_key(
    img_bytes: bytes,
    caption: str,
    model: str,
    prompt_version: str,
) -> str:
    """SHA256 of (image + caption + model + prompt_version) for cache key."""
    payload = (
        img_bytes
        + caption.encode("utf-8")
        + model.encode("utf-8")
        + prompt_version.encode("utf-8")
    )
    return sha256_bytes(payload)


def extract_text_from_openai_response(data: dict[str, Any]) -> str:
    """
    Extract the assistant text from an OpenAI API response.
    Handles both Chat Completions and Responses-style shapes.
    """
    # Chat Completions: choices[0].message.content
    try:
        choices = data.get("choices")
        if choices and len(choices) > 0:
            msg = choices[0].get("message") or choices[0].get("delta")
            if msg and isinstance(msg.get("content"), str):
                return msg["content"].strip()
    except (IndexError, KeyError, TypeError):
        pass

    # Responses API: output_text or output[].content[].text
    if "output_text" in data and isinstance(data["output_text"], str):
        return data["output_text"].strip()

    out = []
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") in ("output_text", "text") and "text" in c:
                out.append(c["text"])
    if out:
        return "\n".join(out).strip()

    return ""


def parse_structured_output(raw_text: str) -> dict[str, Any] | None:
    """
    Try to parse JSON from model output.
    Looks for a JSON object in the text (handles markdown code blocks).
    """
    text = raw_text.strip()
    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from ```json ... ``` or ``` ... ```
    for pattern in (r"```(?:json)?\s*([\s\S]*?)```", r"(\{[\s\S]*\})"):
        match = re.search(pattern, text)
        if match:
            snippet = (match.group(1) or match.group(0)).strip()
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
    return None


class VisionAPIClient:
    """Client for vision-to-text API (OpenAI Chat Completions style)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        prompt_version: Optional[str] = None,
        cache_dir: Optional[str] | Path = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.getenv("FIGURE_VLM_MODEL", "gpt-4o-mini")
        self.endpoint = endpoint or os.getenv("FIGURE_VLM_ENDPOINT", DEFAULT_ENDPOINT)
        self.prompt_version = prompt_version or os.getenv("FIGURE_PROMPT_VERSION", "v1")
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _get_system_prompt(self) -> str:
        return "You are an expert at reading scientific figures."

    def _get_user_prompt(self, caption: str) -> str:
        cap = caption or "None"
        return f"""Caption: {cap}
Task:
- Explain what the figure shows and what conclusion it supports, grounded in visible elements.
- If it's a chart: name axes, trends, comparisons, and key takeaways.
- If it's an architecture/diagram: list components and data flow.
Return JSON ONLY with this schema:
{{"explanation_md": "...", "facts_json": {{"figure_type": "...", "entities": [{{"name": "...", "type": "..."}}], "relations": [{{"source": "...", "relation": "...", "target": "..."}}], "claims": [{{"claim": "...", "confidence_0_1": 0.0}}], "evidence": ["..."]}}}}"""

    def _load_cached(self, cache_key: str) -> VisionResult | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{cache_key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return VisionResult(
                raw_text=data["raw_text"],
                structured_json=data.get("structured_json"),
                model=data.get("model", self.model),
                usage=data.get("usage"),
            )
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", cache_key[:8], e)
            return None

    def _save_cached(self, cache_key: str, result: VisionResult) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{cache_key}.json"
        try:
            path.write_text(
                json.dumps({
                    "raw_text": result.raw_text,
                    "structured_json": result.structured_json,
                    "model": result.model,
                    "usage": result.usage,
                }, indent=0, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", cache_key[:8], e)

    def explain(
        self,
        image_path: str | Path,
        caption: Optional[str] = None,
        *,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> VisionResult:
        """
        Get a structured explanation for the figure.
        Uses cache when use_cache=True and force_refresh=False.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_bytes = image_path.read_bytes()
        cap = caption or "None"
        cache_key = make_cache_key(img_bytes, cap, self.model, self.prompt_version)

        if use_cache and not force_refresh and self.cache_dir:
            cached = self._load_cached(cache_key)
            if cached is not None:
                logger.debug("Vision cache hit: %s", cache_key[:12])
                return cached

        result = self._call_api(img_bytes, cap)
        if use_cache and self.cache_dir:
            self._save_cached(cache_key, result)
        return result

    def _call_api(self, img_bytes: bytes, caption: str) -> VisionResult:
        """Perform the API request with retries and backoff."""
        import base64

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        # Determine MIME from magic bytes if needed; default png
        mime = "image/png"
        if img_bytes[:2] == b"\xff\xd8":
            mime = "image/jpeg"
        data_url = f"data:{mime};base64,{b64}"

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._get_user_prompt(caption)},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        backoff = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(6):
            try:
                r = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                if r.status_code in (429, 500, 503):
                    logger.warning(
                        "Vision API %s (attempt %s), backoff %.1fs",
                        r.status_code,
                        attempt + 1,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                r.raise_for_status()
                data = r.json()
                text = extract_text_from_openai_response(data)
                if not text:
                    logger.warning("Empty response text from vision API")
                parsed = parse_structured_output(text)
                usage = None
                if "usage" in data:
                    usage = data["usage"]
                return VisionResult(
                    raw_text=text,
                    structured_json=parsed,
                    model=self.model,
                    usage=usage,
                )
            except requests.RequestException as e:
                last_exc = e
                if hasattr(e, "response") and e.response is not None:
                    status = e.response.status_code
                    if status in (429, 500, 503):
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                raise

        raise RuntimeError(
            f"Vision API failed after retries: {last_exc}"
        ) from last_exc
