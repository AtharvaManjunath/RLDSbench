"""Adapter protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path

from robot_dataset_tools.models import DatasetSummary, Episode, Issue


class DatasetAdapter(ABC):
    format_name: str

    @abstractmethod
    def can_read(self, path: str | Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_episodes(self, path: str | Path, limit: int | None = None, sample: int | None = None) -> Iterator[Episode]:
        raise NotImplementedError

    def summarize(self, path: str | Path) -> DatasetSummary:
        episodes = list(self.read_episodes(path))
        return DatasetSummary.from_episodes(self.format_name, path, episodes)

    @abstractmethod
    def write_episodes(
        self,
        path: str | Path,
        episodes: Iterable[Episode],
        metadata: dict | None = None,
        overwrite: bool = False,
    ) -> None:
        raise NotImplementedError

    def validate(self, path: str | Path) -> list[Issue]:
        from robot_dataset_tools.audit.schema import validate_episodes

        return validate_episodes(self.read_episodes(path))
