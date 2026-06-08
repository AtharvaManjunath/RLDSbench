import json

import pytest

from robot_dataset_tools.audit.schema import validate_episodes
from robot_dataset_tools.errors import OptionalDependencyError
from robot_dataset_tools.io.jsonl import JSONLAdapter
from robot_dataset_tools.io.rlds import RLDSAdapter
from robot_dataset_tools.models import Episode, Step


def test_empty_dataset(tmp_path):
    path = tmp_path / "empty"
    JSONLAdapter().write_episodes(path, [], overwrite=True)
    episodes = list(JSONLAdapter().read_episodes(path))
    assert episodes == []
    assert validate_episodes(episodes) == []


def test_zero_length_and_inconsistent_actions():
    episodes = [
        Episode("zero", {}, []),
        Episode(
            "bad_shapes",
            {},
            [
                Step(index=0, observation={"state": [0]}, action=[1.0, 2.0]),
                Step(index=1, observation={"state": [1]}, action=[3.0]),
            ],
        ),
    ]
    codes = {issue.code for issue in validate_episodes(episodes)}
    assert "ZERO_LENGTH_EPISODE" in codes
    assert "INCONSISTENT_ACTION_SHAPE" in codes


def test_native_rlds_optional_dependency_message(tmp_path):
    path = tmp_path / "native_like_rlds"
    path.mkdir()
    (path / "dataset_info.json").write_text(json.dumps({"name": "native_like"}), encoding="utf-8")
    with pytest.raises(OptionalDependencyError) as exc:
        list(RLDSAdapter().read_episodes(path))
    assert "RLDS" in str(exc.value) or "TFDS" in str(exc.value)
