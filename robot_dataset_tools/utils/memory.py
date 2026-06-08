"""Memory measurement helpers."""

from __future__ import annotations


def current_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)
