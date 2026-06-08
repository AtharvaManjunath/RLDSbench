"""RLDS and portable RLDS-lite adapter."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

from robot_dataset_tools.errors import OptionalDependencyError, OverwriteError
from robot_dataset_tools.io.base import DatasetAdapter
from robot_dataset_tools.io.jsonl import episode_from_dict, episode_to_dict
from robot_dataset_tools.models import DatasetSummary, Episode, Issue
from robot_dataset_tools.utils.arrays import json_safe


class RLDSAdapter(DatasetAdapter):
    format_name = "rlds"

    def can_read(self, path: str | Path) -> bool:
        p = Path(path)
        return p.is_dir() and (p / "dataset_info.json").is_file()

    def read_episodes(self, path: str | Path, limit: int | None = None, sample: int | None = None) -> Iterator[Episode]:
        p = Path(path)
        lite = p / "episodes.jsonl"
        if lite.exists():
            count = 0
            with lite.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    yield episode_from_dict(json.loads(line)).normalized()
                    count += 1
                    if limit is not None and count >= limit:
                        break
            return
        try:
            import tensorflow_datasets as tfds  # noqa: F401
        except ImportError as exc:
            raise OptionalDependencyError("tensorflow-datasets", "native RLDS/TFDS reading") from exc
        raise OptionalDependencyError(
            "project-specific TFDS builder",
            "native RLDS parsing for arbitrary local TFDS directories",
        )

    def summarize(self, path: str | Path) -> DatasetSummary:
        metadata = json.loads((Path(path) / "dataset_info.json").read_text(encoding="utf-8"))
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
        p.mkdir(parents=True, exist_ok=True)
        info = {"format": "rlds-lite", "version": 1, **(metadata or {})}
        (p / "dataset_info.json").write_text(
            json.dumps(json_safe(info), allow_nan=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        with (p / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for ep in episodes:
                handle.write(json.dumps(episode_to_dict(ep), allow_nan=False, sort_keys=True) + "\n")

    def validate(self, path: str | Path) -> list[Issue]:
        return super().validate(path)
