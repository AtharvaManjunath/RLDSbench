from robot_dataset_tools.audit.schema import validate_episodes
from robot_dataset_tools.io.hdf5 import HDF5Adapter


def test_hdf5_read_write(hdf5_path):
    episodes = list(HDF5Adapter().read_episodes(hdf5_path))
    assert len(episodes) == 3
    assert [len(ep.steps) for ep in episodes] == [3, 5, 2]
    assert episodes[0].steps[0].language_instruction == "pick object 0"
    assert episodes[0].steps[0].observation["state"] == [0, 0, 0.5]


def test_hdf5_missing_and_nan_detection(bad_hdf5_path):
    issues = validate_episodes(HDF5Adapter().read_episodes(bad_hdf5_path))
    codes = {i.code for i in issues}
    assert "NAN_IN_ACTION" in codes
    assert "ZERO_LENGTH_EPISODE" in codes
    assert "EMPTY_LANGUAGE_INSTRUCTION" in codes
