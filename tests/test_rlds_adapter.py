from robot_dataset_tools.io.rlds import RLDSAdapter


def test_rlds_lite_read_write(rlds_path, tmp_path, episodes):
    read = list(RLDSAdapter().read_episodes(rlds_path))
    assert len(read) == 3
    out = tmp_path / "out_rlds"
    RLDSAdapter().write_episodes(out, episodes, overwrite=True)
    assert (out / "dataset_info.json").exists()
    assert len(list(RLDSAdapter().read_episodes(out))) == 3
