"""Portable Robo-DM-style adapter with optional native hook."""

from __future__ import annotations

import json
import shutil
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from robot_dataset_tools.errors import OptionalDependencyError, OverwriteError
from robot_dataset_tools.io.base import DatasetAdapter
from robot_dataset_tools.io.jsonl import episode_from_dict, episode_to_dict
from robot_dataset_tools.models import DatasetSummary, Episode
from robot_dataset_tools.utils.arrays import json_safe, sanitize_id


class RoboDMAdapter(DatasetAdapter):
    format_name = "robodm"

    def can_read(self, path: str | Path) -> bool:
        p = Path(path)
        return (
            p.name.endswith(".robodm.jsonl")
            or p.name.endswith(".robodm.zip")
            or (p.is_dir() and (p / "robodm_metadata.json").exists())
        )

    def read_episodes(self, path: str | Path, limit: int | None = None, sample: int | None = None) -> Iterator[Episode]:
        p = Path(path)
        if p.name.endswith(".robodm.jsonl"):
            yield from self._read_jsonl(p, limit)
            return
        if p.name.endswith(".robodm.zip"):
            yield from self._read_zip(p, limit)
            return
        try:
            import robodm  # noqa: F401
        except ImportError as exc:
            raise OptionalDependencyError("robodm", "native Robo-DM reading") from exc
        raise OptionalDependencyError("robodm public dataset API", "native Robo-DM reading")

    def summarize(self, path: str | Path) -> DatasetSummary:
        metadata = {}
        p = Path(path)
        if p.name.endswith(".robodm.zip"):
            with zipfile.ZipFile(p) as zf:
                if "metadata.json" in zf.namelist():
                    metadata = json.loads(zf.read("metadata.json"))
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
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.name.endswith(".robodm.jsonl"):
            with p.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"metadata": json_safe(metadata or {})}, allow_nan=False, sort_keys=True) + "\n"
                )
                for ep in episodes:
                    handle.write(json.dumps(episode_to_dict(ep), allow_nan=False, sort_keys=True) + "\n")
            return
        if not p.name.endswith(".robodm.zip"):
            p = p.with_suffix(".robodm.zip")
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "metadata.json",
                json.dumps(json_safe({"format": "robodm-portable", **(metadata or {})}), allow_nan=False, indent=2),
            )
            for ep in episodes:
                ep_id = sanitize_id(ep.episode_id)
                zf.writestr(f"episodes/{ep_id}.json", json.dumps(episode_to_dict(ep), allow_nan=False, sort_keys=True))
                actions = [step.action for step in ep.steps if step.action is not None]
                if len(actions) == len(ep.steps) and actions:
                    with zf.open(f"arrays/{ep_id}/actions.npy", "w") as handle:
                        np.save(handle, np.asarray(actions, dtype=float))
                states = [
                    step.observation.get("state")
                    for step in ep.steps
                    if step.observation and step.observation.get("state") is not None
                ]
                if len(states) == len(ep.steps) and states:
                    with zf.open(f"arrays/{ep_id}/states.npy", "w") as handle:
                        np.save(handle, np.asarray(states, dtype=float))

    def _read_jsonl(self, path: Path, limit: int | None) -> Iterator[Episode]:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if line_no == 0 and "metadata" in payload and "steps" not in payload:
                    continue
                yield episode_from_dict(payload).normalized()
                count += 1
                if limit is not None and count >= limit:
                    break

    def _read_zip(self, path: Path, limit: int | None) -> Iterator[Episode]:
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist() if n.startswith("episodes/") and n.endswith(".json"))
            for count, name in enumerate(names):
                if limit is not None and count >= limit:
                    break
                yield episode_from_dict(json.loads(zf.read(name))).normalized()
