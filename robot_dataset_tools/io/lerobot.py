"""Pragmatic local LeRobot-style adapter."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from robot_dataset_tools.errors import OptionalDependencyError, OverwriteError
from robot_dataset_tools.io.base import DatasetAdapter
from robot_dataset_tools.io.jsonl import episode_from_dict
from robot_dataset_tools.models import DatasetSummary, Episode, Step
from robot_dataset_tools.utils.arrays import json_safe


class LeRobotAdapter(DatasetAdapter):
    format_name = "lerobot"

    def can_read(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.is_dir():
            return False
        if (p / "meta" / "info.json").is_file() and (p / "data").exists():
            return True
        return (p / "meta" / "info.json").is_file() or ((p / "data").exists() and (p / "meta").exists())

    def read_episodes(self, path: str | Path, limit: int | None = None, sample: int | None = None) -> Iterator[Episode]:
        p = Path(path)
        jsonl_files = list((p / "data").glob("*.jsonl")) + list((p / "data").glob("**/*.jsonl"))
        if (p / "episodes.jsonl").exists():
            jsonl_files.insert(0, p / "episodes.jsonl")
        if jsonl_files:
            yield from self._read_jsonl_rows(jsonl_files[0], limit)
            return
        parquet_files = sorted((p / "data").glob("**/*.parquet"))
        if parquet_files:
            yield from self._read_parquet_rows(parquet_files, limit)
            return
        raise FileNotFoundError(f"No LeRobot-style JSONL or Parquet step table found under {p}")

    def summarize(self, path: str | Path) -> DatasetSummary:
        metadata = {}
        info = Path(path) / "meta" / "info.json"
        if info.exists():
            metadata = json.loads(info.read_text(encoding="utf-8"))
        return DatasetSummary.from_episodes(self.format_name, path, list(self.read_episodes(path)), metadata)

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
            shutil.rmtree(p)
        (p / "meta").mkdir(parents=True, exist_ok=True)
        (p / "data").mkdir(parents=True, exist_ok=True)
        info = {"format": "lerobot-portable", "version": 1, **(metadata or {})}
        (p / "meta" / "info.json").write_text(
            json.dumps(json_safe(info), allow_nan=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        rows = [row for ep in episodes for row in _episode_rows(ep)]
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            df.to_parquet(p / "data" / "episodes.parquet", index=False)
        except (ImportError, ValueError):
            with (p / "data" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(json_safe(row), allow_nan=False, sort_keys=True) + "\n")

    def _read_jsonl_rows(self, path: Path, limit: int | None) -> Iterator[Episode]:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if rows and "steps" in rows[0]:
            for count, row in enumerate(rows, start=1):
                yield episode_from_dict(row).normalized()
                if limit is not None and count >= limit:
                    break
            return
        yield from _episodes_from_rows(rows, limit)

    def _read_parquet_rows(self, paths: list[Path], limit: int | None) -> Iterator[Episode]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise OptionalDependencyError("pandas/pyarrow", "reading LeRobot-style Parquet tables") from exc
        frames = [pd.read_parquet(path) for path in paths]
        rows = [row.dropna().to_dict() for frame in frames for _, row in frame.iterrows()]
        yield from _episodes_from_rows(rows, limit)


def _episode_rows(episode: Episode) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in episode.steps:
        row = {
            "episode_id": episode.episode_id,
            "episode_index": episode.metadata.get("episode_index", episode.episode_id),
            "frame_index": step.index,
            "timestamp": step.timestamp,
            "action": json_safe(step.action),
            "language_instruction": step.language_instruction,
            "reward": step.reward,
            "discount": step.discount,
            "is_first": step.is_first,
            "is_last": step.is_last,
            "is_terminal": step.is_terminal,
        }
        for key, value in (step.observation or {}).items():
            row[f"observation.{key}"] = json_safe(value)
        rows.append(row)
    return rows


def _episodes_from_rows(rows: list[dict[str, Any]], limit: int | None) -> Iterator[Episode]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        episode_id = str(row.get("episode_id", row.get("episode_index", "0")))
        grouped.setdefault(episode_id, []).append(row)
    for count, (episode_id, ep_rows) in enumerate(grouped.items()):
        if limit is not None and count >= limit:
            break
        steps: list[Step] = []
        for i, row in enumerate(sorted(ep_rows, key=lambda r: int(r.get("frame_index", r.get("index", 0))))):
            observation = {}
            for key, value in row.items():
                if key.startswith("observation."):
                    observation[key.removeprefix("observation.")] = _maybe_parse(value)
            if "state" in row:
                observation["state"] = _maybe_parse(row["state"])
            if "observation.state" in row:
                observation["state"] = _maybe_parse(row["observation.state"])
            action = row.get("action")
            if action is None:
                action_keys = sorted(
                    [k for k in row if k.startswith("action.")],
                    key=_action_key_sort,
                )
                action = [row[k] for k in action_keys] if action_keys else None
            steps.append(
                Step(
                    index=int(row.get("frame_index", row.get("index", i))),
                    observation=observation,
                    action=_maybe_parse(action),
                    reward=_maybe_float(row.get("reward")),
                    discount=_maybe_float(row.get("discount")),
                    is_first=_maybe_bool(row.get("is_first")),
                    is_last=_maybe_bool(row.get("is_last")),
                    is_terminal=_maybe_bool(row.get("is_terminal")),
                    language_instruction=row.get("language_instruction") or row.get("task") or row.get("instruction"),
                    timestamp=_maybe_float(row.get("timestamp")),
                )
            )
        yield Episode(episode_id=episode_id, metadata={}, steps=steps).normalized()


def _maybe_parse(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in {"[", "{"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _action_key_sort(key: str) -> tuple[int, str]:
    suffix = key.rsplit(".", 1)[-1]
    try:
        return (int(suffix), key)
    except ValueError:
        return (10_000, key)
