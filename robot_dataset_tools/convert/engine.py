"""Conversion engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot_dataset_tools.convert.correctness import compare_episode_streams
from robot_dataset_tools.registry import register_builtin_adapters
from robot_dataset_tools.utils.arrays import json_safe


def convert_dataset(
    input_path: str | Path,
    output_path: str | Path,
    src: str = "auto",
    dst: str = "jsonl",
    overwrite: bool = False,
    max_episodes: int | None = None,
    verify: bool = False,
    verify_samples: int | None = None,
    lossy_ok: bool = False,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    reg = register_builtin_adapters()
    src_adapter = reg.resolve(input_path, src)
    dst_adapter = reg.get(dst)
    episodes = list(src_adapter.read_episodes(input_path, limit=max_episodes))
    dst_adapter.write_episodes(
        output_path, episodes, metadata={"source_format": src_adapter.format_name}, overwrite=overwrite
    )
    report: dict[str, Any] = {
        "source_format": src_adapter.format_name,
        "destination_format": dst_adapter.format_name,
        "episode_count": len(episodes),
        "step_count": sum(len(ep.steps) for ep in episodes),
        "verified": False,
    }
    if verify:
        limit = verify_samples
        src_eps = episodes[:limit] if limit else episodes
        dst_eps = list(dst_adapter.read_episodes(output_path, limit=limit))
        correctness = compare_episode_streams(src_eps, dst_eps, lossy_ok=lossy_ok)
        report["verified"] = True
        report["correctness"] = correctness
    if report_path is not None:
        Path(report_path).write_text(
            json.dumps(json_safe(report), allow_nan=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return json_safe(report)
