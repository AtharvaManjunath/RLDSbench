"""Streaming benchmark runner."""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Any

from robot_dataset_tools import __version__
from robot_dataset_tools.registry import register_builtin_adapters
from robot_dataset_tools.utils.memory import current_rss_bytes


def run_benchmark(
    path: str | Path,
    format_name: str = "auto",
    max_episodes: int | None = None,
    warmup_episodes: int = 0,
    repeat: int = 1,
) -> dict[str, Any]:
    reg = register_builtin_adapters()
    adapter = reg.resolve(path, format_name)
    if warmup_episodes > 0:
        for _ in adapter.read_episodes(path, limit=warmup_episodes):
            pass
    repeats = []
    for _ in range(repeat):
        start_mem = current_rss_bytes()
        peak = start_mem
        start = time.perf_counter()
        ep_count = 0
        step_count = 0
        for ep in adapter.read_episodes(path, limit=max_episodes):
            ep_count += 1
            step_count += len(ep.steps)
            rss = current_rss_bytes()
            if rss is not None:
                peak = max(peak or rss, rss)
        wall = max(time.perf_counter() - start, 1e-12)
        repeats.append(
            {
                "episodes": ep_count,
                "steps": step_count,
                "wall_clock_seconds": wall,
                "episodes_per_sec": ep_count / wall,
                "steps_per_sec": step_count / wall,
                "peak_rss_delta_bytes": None if start_mem is None or peak is None else max(0, peak - start_mem),
            }
        )
    first = repeats[0] if repeats else {}
    return {
        "format": adapter.format_name,
        "path": str(path),
        "episodes_per_sec": first.get("episodes_per_sec", 0),
        "steps_per_sec": first.get("steps_per_sec", 0),
        "bytes_read_estimate": _bytes_estimate(path),
        "peak_rss_delta_bytes": first.get("peak_rss_delta_bytes"),
        "wall_clock_seconds": first.get("wall_clock_seconds", 0),
        "per_repeat": repeats,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "package_version": __version__,
        },
    }


def _bytes_estimate(path: str | Path) -> int | None:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return None
