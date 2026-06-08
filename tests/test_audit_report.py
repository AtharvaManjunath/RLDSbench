import json

from robot_dataset_tools.audit.report import build_audit_report, write_json_report
from robot_dataset_tools.io.hdf5 import HDF5Adapter


def test_audit_report_content(hdf5_path, tmp_path):
    episodes = list(HDF5Adapter().read_episodes(hdf5_path))
    report = build_audit_report(str(hdf5_path), "hdf5", episodes)
    assert report["episode_count"] == 3
    assert report["step_count"] == 10
    assert report["episode_lengths"]["p90"] > 0
    assert report["action_stats"]["dimensionality"] == 2
    assert report["language_instruction_coverage"]["episodes_with_instruction"] == 3
    out = tmp_path / "report.json"
    write_json_report(report, out)
    json.loads(out.read_text())
