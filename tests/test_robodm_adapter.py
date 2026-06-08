from robot_dataset_tools.io.robodm import RoboDMAdapter


def test_robodm_zip_and_jsonl_read(robodm_path, robodm_jsonl_path):
    assert len(list(RoboDMAdapter().read_episodes(robodm_path))) == 3
    assert len(list(RoboDMAdapter().read_episodes(robodm_jsonl_path))) == 3
