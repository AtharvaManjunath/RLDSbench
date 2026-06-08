from robot_dataset_tools.visualize.trajectories import generate_visualizations


def test_visualization_files_created(hdf5_path, tmp_path):
    out = tmp_path / "viz"
    result = generate_visualizations(hdf5_path, out, episodes_count=2, seed=1)
    assert (out / "episode_lengths.png").exists()
    assert (out / "action_distribution.png").exists()
    assert (out / "language_coverage.png").exists()
    assert (out / "index.html").exists()
    assert result["files"]
