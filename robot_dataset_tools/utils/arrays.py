"""Array and JSON serialization helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def to_numpy(value: Any, dtype: Any = float) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def sanitize_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return safe or "episode"
