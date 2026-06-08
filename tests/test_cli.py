import json

from typer.testing import CliRunner

from robot_dataset_tools.cli import app

runner = CliRunner()


def test_cli_smoke(hdf5_path, tmp_path):
    result = runner.invoke(app, ["detect", str(hdf5_path)])
    assert result.exit_code == 0
    assert "hdf5" in result.stdout

    report = tmp_path / "report.json"
    result = runner.invoke(app, ["audit", str(hdf5_path), "--out", str(report)])
    assert result.exit_code == 0
    assert json.loads(report.read_text())["episode_count"] == 3

    result = runner.invoke(app, ["validate", str(hdf5_path)])
    assert result.exit_code == 0

    out = tmp_path / "converted.h5"
    result = runner.invoke(app, ["convert", str(hdf5_path), str(out), "--dst", "hdf5", "--verify", "--overwrite"])
    assert result.exit_code == 0

    viz = tmp_path / "viz"
    result = runner.invoke(app, ["visualize", str(hdf5_path), "--out", str(viz)])
    assert result.exit_code == 0

    bench = tmp_path / "bench.json"
    result = runner.invoke(app, ["benchmark", str(hdf5_path), "--json", str(bench)])
    assert result.exit_code == 0
    assert bench.exists()

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0


def test_cli_unknown_format(unknown_path):
    result = runner.invoke(app, ["detect", str(unknown_path)])
    assert result.exit_code == 2
