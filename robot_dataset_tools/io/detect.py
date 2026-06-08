"""Format detection helpers."""

from __future__ import annotations

from pathlib import Path

from robot_dataset_tools.registry import DetectionResult, register_builtin_adapters


def detect_format(path: str | Path) -> DetectionResult:
    return register_builtin_adapters().detect(path)
