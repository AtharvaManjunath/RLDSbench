"""HDF5 adapter for practical robot-learning layouts."""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_dataset_tools.errors import OverwriteError
from robot_dataset_tools.io.base import DatasetAdapter
from robot_dataset_tools.models import DatasetSummary, Episode, Step
from robot_dataset_tools.utils.arrays import json_safe


class HDF5Adapter(DatasetAdapter):
    format_name = "hdf5"

    def can_read(self, path: str | Path) -> bool:
        p = Path(path)
        if p.suffix.lower() not in {".h5", ".hdf5"} or not p.is_file():
            return False
        try:
            with h5py.File(p, "r"):
                return True
        except OSError:
            return False

    def read_episodes(self, path: str | Path, limit: int | None = None, sample: int | None = None) -> Iterator[Episode]:
        with h5py.File(path, "r") as handle:
            groups = _episode_groups(handle)
            for count, (episode_id, group) in enumerate(groups, start=1):
                yield _read_episode(episode_id, group).normalized()
                if limit is not None and count >= limit:
                    break

    def summarize(self, path: str | Path) -> DatasetSummary:
        with h5py.File(path, "r") as handle:
            metadata = {k: _decode(v) for k, v in handle.attrs.items()}
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
        p.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype("utf-8")
        with h5py.File(p, "w") as handle:
            for key, value in (metadata or {}).items():
                _write_attr(handle.attrs, key, value)
            root = handle.create_group("episodes")
            for episode in episodes:
                ep_group = root.create_group(str(episode.episode_id))
                for key, value in episode.metadata.items():
                    _write_attr(ep_group.attrs, key, value)
                steps = episode.steps
                actions = [step.action for step in steps if step.action is not None]
                if len(actions) == len(steps) and actions:
                    ep_group.create_dataset("actions", data=np.asarray(actions, dtype=float))
                _write_optional_vector(ep_group, "rewards", [s.reward for s in steps])
                _write_optional_vector(ep_group, "discounts", [s.discount for s in steps])
                _write_optional_vector(ep_group, "terminals", [s.is_terminal for s in steps])
                _write_optional_vector(ep_group, "timestamps", [s.timestamp for s in steps])
                instructions = [s.language_instruction for s in steps]
                if any(x is not None for x in instructions):
                    data = ["" if x is None else str(x) for x in instructions]
                    ep_group.create_dataset(
                        "language_instruction", data=np.asarray(data, dtype=object), dtype=string_dtype
                    )
                obs_group = ep_group.create_group("observations")
                obs_by_key: dict[str, list[Any]] = {}
                for step in steps:
                    flat = _flatten_observation(step.observation or {})
                    for key, value in flat.items():
                        obs_by_key.setdefault(key, []).append(value)
                for key, values in obs_by_key.items():
                    _write_observation_dataset(obs_group, key, values)


def _episode_groups(handle: h5py.File) -> list[tuple[str, h5py.Group]]:
    for root_name in ("episodes", "data"):
        if root_name in handle and isinstance(handle[root_name], h5py.Group):
            root = handle[root_name]
            return [(str(name), group) for name, group in root.items() if isinstance(group, h5py.Group)]
    return [("episode_0", handle)]


def _read_episode(episode_id: str, group: h5py.Group) -> Episode:
    n = _infer_length(group)
    metadata = {k: _decode(v) for k, v in group.attrs.items()}
    actions = _dataset(group, "actions")
    rewards = _dataset(group, "rewards")
    discounts = _dataset(group, "discounts")
    terminals = _first_dataset(group, "terminals", "dones")
    timestamps = _dataset(group, "timestamps")
    language = _first_dataset(group, "language_instruction", "language_instructions")
    observations = _read_observations(group, n)
    steps: list[Step] = []
    for i in range(n):
        steps.append(
            Step(
                index=i,
                observation=_observation_at(observations, i),
                action=_item(actions, i),
                reward=_float_item(rewards, i),
                discount=_float_item(discounts, i),
                is_first=i == 0,
                is_last=i == n - 1,
                is_terminal=_bool_item(terminals, i),
                language_instruction=_str_item(language, i),
                timestamp=_float_item(timestamps, i),
            )
        )
    return Episode(episode_id=episode_id, metadata=metadata, steps=steps)


def _infer_length(group: h5py.Group) -> int:
    for name in (
        "actions",
        "rewards",
        "discounts",
        "terminals",
        "dones",
        "timestamps",
        "language_instruction",
        "language_instructions",
    ):
        data = _dataset(group, name)
        length = _safe_len(data)
        if length is not None:
            return length
    if "observations" in group:
        for value in group["observations"].values():
            if isinstance(value, h5py.Dataset):
                length = _safe_len(value[()])
                if length is not None:
                    return length
            if isinstance(value, h5py.Group):
                for nested in value.values():
                    if isinstance(nested, h5py.Dataset):
                        length = _safe_len(nested[()])
                        if length is not None:
                            return length
    return 0


def _dataset(group: h5py.Group, name: str) -> np.ndarray | None:
    if name in group and isinstance(group[name], h5py.Dataset):
        return group[name][()]
    return None


def _first_dataset(group: h5py.Group, *names: str) -> np.ndarray | None:
    for name in names:
        data = _dataset(group, name)
        if data is not None:
            return data
    return None


def _read_observations(group: h5py.Group, n: int) -> dict[str, list[Any]]:
    obs: dict[str, list[Any]] = {}
    if "observations" not in group:
        return obs
    _collect_obs(group["observations"], "", obs, n)
    return obs


def _collect_obs(group: h5py.Group, prefix: str, out: dict[str, list[Any]], n: int) -> None:
    for name, value in group.items():
        key = f"{prefix}.{name}" if prefix else str(name)
        if isinstance(value, h5py.Dataset):
            raw = value[()]
            out[key] = [_decode(_item(raw, i)) for i in range(n)]
        elif isinstance(value, h5py.Group):
            _collect_obs(value, key, out, n)


def _item(values: Any, index: int) -> Any:
    if values is None:
        return None
    length = _safe_len(values)
    if length is None:
        return _decode(values)
    if index >= length:
        return None
    return _decode(values[index])


def _float_item(values: Any, index: int) -> float | None:
    value = _item(values, index)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_item(values: Any, index: int) -> bool | None:
    value = _item(values, index)
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return bool(value)


def _str_item(values: Any, index: int) -> str | None:
    value = _item(values, index)
    if value is None:
        return None
    return str(value)


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "O", "U"}:
            return [_decode(x) for x in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_len(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bytes, str)):
        return None
    if isinstance(value, np.ndarray) and value.shape == ():
        return None
    try:
        return int(len(value))
    except TypeError:
        return None


def _write_attr(attrs: h5py.AttributeManager, key: str, value: Any) -> None:
    safe = json_safe(value)
    try:
        attrs[key] = safe
    except TypeError:
        import json

        attrs[key] = json.dumps(safe, allow_nan=False, sort_keys=True)


def _flatten_observation(observation: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in observation.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_observation(value, full))
        else:
            flat[full] = value
    return flat


def _observation_at(observations: dict[str, list[Any]], index: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, values in observations.items():
        if index < len(values):
            _set_nested(out, key.split("."), values[index])
    return out


def _set_nested(target: dict[str, Any], parts: list[str], value: Any) -> None:
    current = target
    for part in parts[:-1]:
        existing = current.setdefault(part, {})
        if not isinstance(existing, dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _write_optional_vector(group: h5py.Group, name: str, values: list[Any]) -> None:
    if not values or not any(value is not None for value in values):
        return
    clean = [np.nan if value is None else value for value in values]
    group.create_dataset(name, data=np.asarray(clean))


def _write_observation_dataset(group: h5py.Group, key: str, values: list[Any]) -> None:
    parts = key.split(".")
    target = group
    for part in parts[:-1]:
        target = target.require_group(part)
    name = parts[-1]
    try:
        target.create_dataset(name, data=np.asarray(values))
    except (TypeError, ValueError):
        string_dtype = h5py.string_dtype("utf-8")
        target.create_dataset(name, data=np.asarray([str(v) for v in values], dtype=object), dtype=string_dtype)
