from copy import deepcopy

import pytest

from robot_dataset_tools.convert.correctness import compare_episode_streams
from robot_dataset_tools.convert.engine import convert_dataset
from robot_dataset_tools.errors import OverwriteError
from robot_dataset_tools.io.hdf5 import HDF5Adapter
from robot_dataset_tools.io.jsonl import JSONLAdapter
from robot_dataset_tools.io.robodm import RoboDMAdapter


def test_hdf5_jsonl_hdf5_roundtrip(hdf5_path, tmp_path):
    jsonl = tmp_path / "out_jsonl"
    hdf5 = tmp_path / "out.h5"
    r1 = convert_dataset(hdf5_path, jsonl, dst="jsonl", overwrite=True, verify=True)
    assert r1["correctness"]["passed"]
    r2 = convert_dataset(jsonl, hdf5, dst="hdf5", overwrite=True, verify=True)
    assert r2["correctness"]["passed"]
    assert len(list(HDF5Adapter().read_episodes(hdf5))) == 3


def test_lerobot_to_hdf5_correctness(lerobot_path, tmp_path):
    out = tmp_path / "out.hdf5"
    result = convert_dataset(lerobot_path, out, dst="hdf5", overwrite=True, verify=True)
    assert result["correctness"]["passed"]


def test_rlds_to_robodm_correctness(rlds_path, tmp_path):
    out = tmp_path / "out.robodm.zip"
    result = convert_dataset(rlds_path, out, dst="robodm", overwrite=True, verify=True)
    assert result["correctness"]["passed"]
    assert len(list(RoboDMAdapter().read_episodes(out))) == 3


def test_overwrite_protection(jsonl_path, tmp_path):
    out = tmp_path / "existing"
    JSONLAdapter().write_episodes(out, [], overwrite=True)
    with pytest.raises(OverwriteError):
        convert_dataset(jsonl_path, out, dst="jsonl", overwrite=False)


def test_compare_detects_action_drop(episodes):
    src = [deepcopy(episodes[0])]
    dst = [deepcopy(episodes[0])]
    dst[0].steps[0].action = None
    report = compare_episode_streams(src, dst)
    assert not report["passed"]
