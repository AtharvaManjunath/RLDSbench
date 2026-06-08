from robot_dataset_tools.io.detect import detect_format


def test_detect_formats(hdf5_path, lerobot_path, rlds_path, robodm_path, jsonl_path, unknown_path):
    assert detect_format(hdf5_path).format == "hdf5"
    assert detect_format(lerobot_path).format == "lerobot"
    assert detect_format(rlds_path).format == "rlds"
    assert detect_format(robodm_path).format == "robodm"
    assert detect_format(jsonl_path).format == "jsonl"
    assert detect_format(unknown_path).format == "unknown"
