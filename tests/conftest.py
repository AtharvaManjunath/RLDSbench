from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from robot_dataset_tools.io.hdf5 import HDF5Adapter
from robot_dataset_tools.io.jsonl import JSONLAdapter
from robot_dataset_tools.io.rlds import RLDSAdapter
from robot_dataset_tools.io.robodm import RoboDMAdapter
from robot_dataset_tools.models import Episode, Step


@pytest.fixture()
def episodes() -> list[Episode]:
    out = []
    for ep_i, length in enumerate([3, 5, 2]):
        steps = []
        for i in range(length):
            steps.append(
                Step(
                    index=i,
                    observation={"state": [ep_i, i, i + 0.5]},
                    action=[float(i), float(i + 1)],
                    reward=float(i),
                    discount=1.0,
                    is_first=i == 0,
                    is_last=i == length - 1,
                    is_terminal=i == length - 1,
                    language_instruction=f"pick object {ep_i}",
                    timestamp=float(i),
                )
            )
        out.append(Episode(f"ep_{ep_i}", {"split": "train"}, steps))
    return out


@pytest.fixture()
def hdf5_path(tmp_path: Path, episodes: list[Episode]) -> Path:
    path = tmp_path / "demo.hdf5"
    HDF5Adapter().write_episodes(path, episodes, metadata={"name": "demo"}, overwrite=True)
    return path


@pytest.fixture()
def bad_hdf5_path(tmp_path: Path) -> Path:
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        root = f.create_group("episodes")
        ep = root.create_group("bad")
        ep.create_dataset("actions", data=np.asarray([[1.0, np.nan], [2.0, 3.0]]))
        ep.create_group("observations").create_dataset("state", data=np.asarray([[0.0, 1.0], [1.0, 2.0]]))
        dt = h5py.string_dtype("utf-8")
        ep.create_dataset("language_instruction", data=np.asarray(["", "go"], dtype=object), dtype=dt)
        root.create_group("empty")
    return path


@pytest.fixture()
def jsonl_path(tmp_path: Path, episodes: list[Episode]) -> Path:
    path = tmp_path / "jsonl_ds"
    JSONLAdapter().write_episodes(path, episodes, metadata={"kind": "jsonl"}, overwrite=True)
    return path


@pytest.fixture()
def lerobot_path(tmp_path: Path, episodes: list[Episode]) -> Path:
    path = tmp_path / "lerobot_ds"
    (path / "meta").mkdir(parents=True)
    (path / "data").mkdir()
    (path / "meta" / "info.json").write_text(json.dumps({"format": "lerobot"}), encoding="utf-8")
    rows = []
    for ep in episodes:
        for s in ep.steps:
            rows.append(
                {
                    "episode_id": ep.episode_id,
                    "frame_index": s.index,
                    "timestamp": s.timestamp,
                    "action": s.action,
                    "observation.state": s.observation["state"],
                    "language_instruction": s.language_instruction,
                    "reward": s.reward,
                    "discount": s.discount,
                    "is_first": s.is_first,
                    "is_last": s.is_last,
                    "is_terminal": s.is_terminal,
                }
            )
    with (path / "data" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


@pytest.fixture()
def rlds_path(tmp_path: Path, episodes: list[Episode]) -> Path:
    path = tmp_path / "rlds_lite"
    RLDSAdapter().write_episodes(path, episodes, metadata={"name": "rlds"}, overwrite=True)
    return path


@pytest.fixture()
def robodm_path(tmp_path: Path, episodes: list[Episode]) -> Path:
    path = tmp_path / "portable.robodm.zip"
    RoboDMAdapter().write_episodes(path, episodes, metadata={"name": "robodm"}, overwrite=True)
    return path


@pytest.fixture()
def robodm_jsonl_path(tmp_path: Path, episodes: list[Episode]) -> Path:
    path = tmp_path / "portable.robodm.jsonl"
    RoboDMAdapter().write_episodes(path, episodes, overwrite=True)
    return path


@pytest.fixture()
def unknown_path(tmp_path: Path) -> Path:
    path = tmp_path / "unknown.txt"
    path.write_text("not a dataset", encoding="utf-8")
    return path
