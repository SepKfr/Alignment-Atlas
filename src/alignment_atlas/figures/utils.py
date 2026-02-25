"""Shared utilities for the figures pipeline."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def sha256_bytes(b: bytes) -> str:
    """Return hex digest of SHA256 hash of bytes."""
    return hashlib.sha256(b).hexdigest()


def ensure_dir(path: Path | str) -> Path:
    """Create directory (and parents) if it does not exist. Return path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_image_extension(ext: str) -> str:
    """Normalize image extension to .png or .jpg for saving."""
    ext = (ext or "").strip().lower()
    if ext in (".png", "png"):
        return ".png"
    if ext in (".jpg", ".jpeg", "jpg", "jpeg"):
        return ".jpg"
    return ".png"
