from robot_dataset_tools.benchmark.runner import run_benchmark


def test_benchmark_counts(hdf5_path):
    result = run_benchmark(hdf5_path)
    assert result["per_repeat"][0]["episodes"] == 3
    assert result["per_repeat"][0]["steps"] == 10
    assert result["steps_per_sec"] > 0
