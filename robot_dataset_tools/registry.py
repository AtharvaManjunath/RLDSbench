"""Adapter registry and detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robot_dataset_tools.errors import UnknownFormatError
from robot_dataset_tools.io.base import DatasetAdapter


@dataclass
class DetectionResult:
    format: str
    confidence: float
    adapter: DatasetAdapter | None
    reason: str


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}

    def register(self, adapter: DatasetAdapter) -> None:
        self._adapters[adapter.format_name] = adapter

    def get(self, name: str) -> DatasetAdapter:
        if name == "auto":
            raise ValueError("Use detect_adapter for auto format.")
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise UnknownFormatError(f"Unknown format '{name}'.") from exc

    def detect(self, path: str | Path) -> DetectionResult:
        p = Path(path)
        best: DetectionResult | None = None
        for name, adapter in self._adapters.items():
            try:
                ok = adapter.can_read(p)
            except Exception:
                ok = False
            if ok:
                confidence = 0.95
                if name in {"jsonl", "lerobot"} and best is not None:
                    confidence = 0.8
                result = DetectionResult(name, confidence, adapter, f"{name} adapter matched")
                if best is None or result.confidence > best.confidence:
                    best = result
        if best is None:
            return DetectionResult("unknown", 0.0, None, "No adapter recognized the path")
        return best

    def resolve(self, path: str | Path, format_name: str = "auto") -> DatasetAdapter:
        if format_name != "auto":
            return self.get(format_name)
        result = self.detect(path)
        if result.adapter is None:
            raise UnknownFormatError(f"Could not detect dataset format for {path}.")
        return result.adapter

    @property
    def names(self) -> list[str]:
        return sorted(self._adapters)


registry = AdapterRegistry()


def register_builtin_adapters() -> AdapterRegistry:
    if registry.names:
        return registry
    from robot_dataset_tools.io.hdf5 import HDF5Adapter
    from robot_dataset_tools.io.jsonl import JSONLAdapter
    from robot_dataset_tools.io.lerobot import LeRobotAdapter
    from robot_dataset_tools.io.rlds import RLDSAdapter
    from robot_dataset_tools.io.robodm import RoboDMAdapter

    for adapter in [HDF5Adapter(), LeRobotAdapter(), RLDSAdapter(), RoboDMAdapter(), JSONLAdapter()]:
        registry.register(adapter)
    return registry
