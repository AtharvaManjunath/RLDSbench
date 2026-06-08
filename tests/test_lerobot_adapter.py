from robot_dataset_tools.io.lerobot import LeRobotAdapter


def test_lerobot_jsonl_read(lerobot_path):
    episodes = list(LeRobotAdapter().read_episodes(lerobot_path))
    assert len(episodes) == 3
    assert episodes[1].steps[0].observation["state"] == [1, 0, 0.5]


def test_lerobot_write(tmp_path, episodes):
    out = tmp_path / "out_lerobot"
    LeRobotAdapter().write_episodes(out, episodes, overwrite=True)
    assert (out / "meta" / "info.json").exists()
    assert list(LeRobotAdapter().read_episodes(out))
