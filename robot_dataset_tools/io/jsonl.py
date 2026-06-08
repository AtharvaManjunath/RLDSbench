"""Canonical JSONL adapter."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from robot_dataset_tools.errors import OverwriteError
from robot_dataset_tools.io.base import DatasetAdapter
from robot_dataset_tools.models import DatasetSummary, Episode, Step
from robot_dataset_tools.utils.arrays import json_safe


class JSONLAdapter(DatasetAdapter):
    format_name = "jsonl"

    def _episodes_file(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_file() else p / "episodes.jsonl"

    def can_read(self, path: str | Path) -> bool:
        p = Path(path)
        if p.is_file() and p.suffix == ".jsonl" and ".robodm" not in p.name:
            return True
        return (p / "episodes.jsonl").is_file() and not (p / "dataset_info.json").is_file()

    def read_episodes(self, path: str | Path, limit: int | None = None, sample: int | None = None) -> Iterator[Episode]:
        file_path = self._episodes_file(path)
        count = 0
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                yield episode_from_dict(payload).normalized()
                count += 1
                if limit is not None and count >= limit:
                    break

    def summarize(self, path: str | Path) -> DatasetSummary:
        metadata = {}
        p = Path(path)
        meta_path = (p.parent if p.is_file() else p) / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return DatasetSummary.from_episodes(self.format_name, path, list(self.read_episodes(path)), metadata=metadata)

    def write_episodes(
        self,
        path: str | Path,
        episodes: Iterable[Episode],
        metadata: dict | None = None,
        overwrite: bool = False,
    ) -> None:
        p = Path(path)
        if p.exists():
            if not overwrite:
                raise OverwriteError(f"Output path already exists: {p}")
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        if p.suffix == ".jsonl":
            p.parent.mkdir(parents=True, exist_ok=True)
            file_path = p
            meta_dir = p.parent
        else:
            p.mkdir(parents=True, exist_ok=True)
            file_path = p / "episodes.jsonl"
            meta_dir = p
        if metadata is not None:
            (meta_dir / "metadata.json").write_text(
                json.dumps(json_safe(metadata), indent=2, sort_keys=True), encoding="utf-8"
            )
        with file_path.open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode_to_dict(episode), allow_nan=False, sort_keys=True) + "\n")


def step_from_dict(payload: dict[str, Any]) -> Step:
    return Step(
        index=int(payload.get("index", 0)),
        observation=payload.get("observation") or {},
        action=payload.get("action"),
        reward=payload.get("reward"),
        discount=payload.get("discount"),
        is_first=payload.get("is_first"),
        is_last=payload.get("is_last"),
        is_terminal=payload.get("is_terminal"),
        language_instruction=payload.get("language_instruction"),
        timestamp=payload.get("timestamp"),
    )


def episode_from_dict(payload: dict[str, Any]) -> Episode:
    return Episode(
        episode_id=str(payload.get("episode_id", payload.get("id", ""))),
        metadata=payload.get("metadata") or {},
        steps=[step_from_dict(step) for step in payload.get("steps", [])],
    )


def step_to_dict(step: Step) -> dict[str, Any]:
    return json_safe(
        {
            "index": step.index,
            "observation": json_safe(step.observation),
            "action": json_safe(step.action),
            "reward": step.reward,
            "discount": step.discount,
            "is_first": step.is_first,
            "is_last": step.is_last,
            "is_terminal": step.is_terminal,
            "language_instruction": step.language_instruction,
            "timestamp": step.timestamp,
        }
    )


def episode_to_dict(episode: Episode) -> dict[str, Any]:
    return json_safe(
        {
            "episode_id": episode.episode_id,
            "metadata": json_safe(episode.metadata),
            "steps": [step_to_dict(step) for step in episode.steps],
        }
    )
