"""Canonical data models used by all adapters and pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

Severity = Literal["info", "warning", "error"]


@dataclass
class Issue:
    severity: Severity
    code: str
    message: str
    episode_id: str | None = None
    step_index: int | None = None
    field: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "field": self.field,
            "suggestion": self.suggestion,
        }
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class Step:
    index: int
    observation: dict[str, Any] = field(default_factory=dict)
    action: Any | None = None
    reward: float | None = None
    discount: float | None = None
    is_first: bool | None = None
    is_last: bool | None = None
    is_terminal: bool | None = None
    language_instruction: str | None = None
    timestamp: float | int | None = None

    def action_array(self) -> np.ndarray | None:
        if self.action is None:
            return None
        try:
            arr = np.asarray(self.action, dtype=float)
        except (TypeError, ValueError):
            return None
        return arr


@dataclass
class Episode:
    episode_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)

    def normalized(self) -> Episode:
        for i, step in enumerate(self.steps):
            if step.index is None:
                step.index = i
        if self.steps:
            if self.steps[0].is_first is None:
                self.steps[0].is_first = True
            if self.steps[-1].is_last is None:
                self.steps[-1].is_last = True
        return self


@dataclass
class DatasetSummary:
    format: str
    path: str
    num_episodes: int
    num_steps: int
    observation_keys: list[str] = field(default_factory=list)
    action_shape: list[int] | None = None
    language_instruction_fields: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_episodes(
        cls,
        format_name: str,
        path: str | Path,
        episodes: list[Episode],
        metadata: dict[str, Any] | None = None,
    ) -> DatasetSummary:
        observation_keys: set[str] = set()
        action_shape: list[int] | None = None
        has_lang = False
        steps_count = 0
        for ep in episodes:
            steps_count += len(ep.steps)
            for step in ep.steps:
                observation_keys.update(flatten_keys(step.observation))
                arr = step.action_array()
                if arr is not None and action_shape is None:
                    action_shape = list(arr.shape)
                if step.language_instruction is not None:
                    has_lang = True
        return cls(
            format=format_name,
            path=str(path),
            num_episodes=len(episodes),
            num_steps=steps_count,
            observation_keys=sorted(observation_keys),
            action_shape=action_shape,
            language_instruction_fields=["language_instruction"] if has_lang else [],
            metadata=metadata or {},
        )


def flatten_keys(data: dict[str, Any], prefix: str = "") -> list[str]:
    keys: list[str] = []
    for key, value in (data or {}).items():
        full = f"{prefix}.{key}" if prefix else str(key)
        keys.append(full)
        if isinstance(value, dict):
            keys.extend(flatten_keys(value, full))
    return keys
