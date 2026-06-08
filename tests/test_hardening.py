from __future__ import annotations

import json
import zipfile
from copy import deepcopy
from importlib.util import find_spec

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from robot_dataset_tools.audit.report import build_audit_report, write_json_report
from robot_dataset_tools.benchmark.runner import run_benchmark
from robot_dataset_tools.cli import app
from robot_dataset_tools.convert.correctness import compare_episode_streams
from robot_dataset_tools.io.hdf5 import HDF5Adapter
from robot_dataset_tools.io.jsonl import JSONLAdapter
from robot_dataset_tools.io.lerobot import LeRobotAdapter
from robot_dataset_tools.io.rlds import RLDSAdapter
from robot_dataset_tools.io.robodm import RoboDMAdapter
from robot_dataset_tools.models import Episode, Step
from robot_dataset_tools.visualize.trajectories import generate_visualizations

runner = CliRunner()


def test_hdf5_data_layout_flat_layout_scalar_strings_attrs_and_nested_observations(tmp_path):
    data_path = tmp_path / "data_layout.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(data_path, "w") as f:
        f.attrs["config"] = json.dumps({"robot": "arm"})
        ep = f.create_group("data").create_group("episode_a")
        ep.attrs["split"] = "train"
        ep.create_dataset("actions", data=np.asarray([[1.0], [2.0]]))
        ep.create_dataset("language_instruction", data="open drawer", dtype=string_dtype)
        ep.create_group("observations").create_group("images").create_dataset(
            "front", data=np.asarray([[[1, 2]], [[3, 4]]])
        )
    data_eps = list(HDF5Adapter().read_episodes(data_path))
    assert len(data_eps) == 1
    assert data_eps[0].episode_id == "episode_a"
    assert [s.language_instruction for s in data_eps[0].steps] == ["open drawer", "open drawer"]
    assert data_eps[0].steps[0].observation["images"]["front"] == [[1, 2]]

    flat_path = tmp_path / "flat.hdf5"
    with h5py.File(flat_path, "w") as f:
        f.attrs["metadata"] = json.dumps({"layout": "flat"})
        f.create_dataset("actions", data=np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]))
        f.create_dataset("language_instructions", data=np.asarray(["a", "b", "c"], dtype=object), dtype=string_dtype)
        f.create_group("observations").create_dataset("state", data=np.asarray([[0.0], [1.0], [2.0]]))
    flat_eps = list(HDF5Adapter().read_episodes(flat_path))
    assert len(flat_eps) == 1
    assert flat_eps[0].episode_id == "episode_0"
    assert len(flat_eps[0].steps) == 3
    assert flat_eps[0].steps[2].language_instruction == "c"


def test_hdf5_roundtrip_preserves_nested_observation_lengths(tmp_path):
    episodes = [
        Episode(
            "nested",
            {"meta": {"nested": True}},
            [
                Step(index=0, observation={"state": [0, 1], "images": {"front": [[1, 2]]}}, action=[1.0]),
                Step(index=1, observation={"state": [1, 2], "images": {"front": [[3, 4]]}}, action=[2.0]),
            ],
        )
    ]
    out = tmp_path / "nested.h5"
    HDF5Adapter().write_episodes(out, episodes, metadata={"root": {"ok": True}}, overwrite=True)
    read = list(HDF5Adapter().read_episodes(out))
    assert len(read[0].steps) == 2
    assert read[0].steps[1].observation["images"]["front"] == [[3, 4]]


def test_jsonl_single_file_nested_observations_and_strict_json(tmp_path):
    path = tmp_path / "episodes.jsonl"
    episodes = [
        Episode(
            "json",
            {},
            [Step(index=0, observation={"nested": {"x": np.asarray([1, 2])}}, action=[np.nan, np.inf])],
        )
    ]
    JSONLAdapter().write_episodes(path, episodes, overwrite=True)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    read = list(JSONLAdapter().read_episodes(path))
    assert read[0].steps[0].observation["nested"]["x"] == [1, 2]


def test_lerobot_jsonl_action_columns_and_language_fallbacks(tmp_path):
    path = tmp_path / "lerobot_columns"
    (path / "meta").mkdir(parents=True)
    (path / "data").mkdir()
    (path / "meta" / "info.json").write_text("{}", encoding="utf-8")
    rows = [
        {"episode_index": 7, "frame_index": 0, "action.1": 2.0, "action.0": 1.0, "state": [0, 1], "task": "stack"},
        {
            "episode_index": 7,
            "frame_index": 1,
            "action.1": 4.0,
            "action.0": 3.0,
            "observation.state": [1, 2],
            "instruction": "stack",
        },
    ]
    with (path / "data" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    episodes = list(LeRobotAdapter().read_episodes(path))
    assert episodes[0].episode_id == "7"
    assert episodes[0].steps[0].action == [1.0, 2.0]
    assert episodes[0].steps[1].language_instruction == "stack"


def test_robodm_zip_contains_metadata_and_payloads(tmp_path, episodes):
    out = tmp_path / "dataset.robodm.zip"
    RoboDMAdapter().write_episodes(out, episodes, metadata={"name": "zip"}, overwrite=True)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "metadata.json" in names
    assert "episodes/ep_0.json" in names
    assert "arrays/ep_0/actions.npy" in names
    assert len(list(RoboDMAdapter().read_episodes(out))) == len(episodes)


def test_rlds_lite_preserves_step_fields(tmp_path, episodes):
    out = tmp_path / "rlds"
    RLDSAdapter().write_episodes(out, episodes, overwrite=True)
    read = list(RLDSAdapter().read_episodes(out))
    assert read[0].steps[0].reward == episodes[0].steps[0].reward
    assert read[0].steps[0].discount == episodes[0].steps[0].discount
    assert read[0].steps[0].is_first is True
    assert read[0].steps[-1].is_terminal is True


def test_report_action_stats_are_json_safe_for_nan_inf_and_ragged(tmp_path):
    episodes = [
        Episode("a", {}, [Step(index=0, observation={"state": [0]}, action=[1.0, np.nan])]),
        Episode("b", {}, [Step(index=0, observation={"state": [1]}, action=[np.inf])]),
    ]
    report = build_audit_report("memory", "jsonl", episodes)
    assert report["action_stats"]["nan_count"] == 1
    assert report["action_stats"]["inf_count"] == 1
    out = tmp_path / "report.json"
    write_json_report(report, out)
    json.loads(out.read_text(encoding="utf-8"))


def test_conversion_mismatch_reports_all_core_failures(episodes):
    src = deepcopy(episodes)
    dst = deepcopy(episodes[:-1])
    dst[0].steps[0].action = [99.0, 100.0]
    dst[0].steps[1].language_instruction = None
    dst[1].steps.pop()
    report = compare_episode_streams(src, dst)
    codes = {issue["code"] for issue in report["issues"]}
    assert not report["passed"]
    assert "EPISODE_COUNT_MISMATCH" in codes
    assert "ACTION_VALUE_MISMATCH" in codes
    assert "LANGUAGE_MISMATCH" in codes
    assert "EPISODE_LENGTH_MISMATCH" in codes


def test_visualization_no_actions_non_empty_pngs_and_deterministic(tmp_path):
    episodes = [
        Episode("a", {}, [Step(index=0, observation={}, language_instruction=None)]),
        Episode("b", {}, [Step(index=0, observation={}, language_instruction="")]),
        Episode("c", {}, [Step(index=0, observation={}, language_instruction="go")]),
    ]
    dataset = tmp_path / "no_actions"
    JSONLAdapter().write_episodes(dataset, episodes, overwrite=True)
    out1 = tmp_path / "viz1"
    out2 = tmp_path / "viz2"
    result1 = generate_visualizations(dataset, out1, episodes_count=2, seed=5)
    result2 = generate_visualizations(dataset, out2, episodes_count=2, seed=5)
    assert "No actions found" in result1["warnings"][0]
    assert (out1 / "episode_lengths.png").stat().st_size > 0
    assert (out1 / "language_coverage.png").stat().st_size > 0
    assert (out1 / "index.html").exists()
    assert result1["files"] == result2["files"]


def test_benchmark_repeat_json_and_no_psutil(monkeypatch, jsonl_path):
    monkeypatch.setattr("robot_dataset_tools.benchmark.runner.current_rss_bytes", lambda: None)
    result = run_benchmark(jsonl_path, repeat=2)
    assert len(result["per_repeat"]) == 2
    assert result["environment"]["python"]
    json.dumps(result, allow_nan=False)


def test_cli_exit_codes_for_validation_overwrite_optional_and_verify_failure(
    tmp_path, bad_hdf5_path, unknown_path, jsonl_path, monkeypatch
):
    assert runner.invoke(app, ["detect", str(unknown_path)]).exit_code == 2

    strict = runner.invoke(app, ["validate", str(bad_hdf5_path), "--strict"])
    assert strict.exit_code == 1
    non_strict = runner.invoke(app, ["validate", str(bad_hdf5_path)])
    assert non_strict.exit_code == 1

    existing = tmp_path / "existing"
    JSONLAdapter().write_episodes(existing, [], overwrite=True)
    overwrite = runner.invoke(app, ["convert", str(jsonl_path), str(existing), "--dst", "jsonl"])
    assert overwrite.exit_code == 2

    native_like = tmp_path / "native_like"
    native_like.mkdir()
    (native_like / "dataset_info.json").write_text("{}", encoding="utf-8")
    optional = runner.invoke(app, ["validate", str(native_like), "--format", "rlds"])
    assert optional.exit_code == 3

    def failed_convert(*args, **kwargs):
        return {"verified": True, "correctness": {"passed": False, "issues": [{"code": "ACTION_VALUE_MISMATCH"}]}}

    monkeypatch.setattr("robot_dataset_tools.cli.convert_dataset", failed_convert)
    verify = runner.invoke(app, ["convert", str(jsonl_path), str(tmp_path / "out"), "--dst", "jsonl", "--verify"])
    assert verify.exit_code == 1


@pytest.mark.skipif(
    find_spec("pandas") is None or find_spec("pyarrow") is None,
    reason="pandas/pyarrow not installed",
)
def test_lerobot_parquet_read_when_available(tmp_path):
    import pandas as pd

    path = tmp_path / "lerobot_parquet"
    (path / "meta").mkdir(parents=True)
    (path / "data").mkdir()
    (path / "meta" / "info.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "episode_id": "p",
                "frame_index": 0,
                "action": [1.0, 2.0],
                "observation.state": [0.0],
                "language_instruction": "go",
            }
        ]
    ).to_parquet(path / "data" / "episodes.parquet", index=False)
    episodes = list(LeRobotAdapter().read_episodes(path))
    assert episodes[0].steps[0].action == [1.0, 2.0]
