"""Create tiny deterministic demo datasets for README examples."""

from __future__ import annotations

import json
from pathlib import Path

from robot_dataset_tools.audit.report import build_audit_report, write_json_report
from robot_dataset_tools.io.hdf5 import HDF5Adapter
from robot_dataset_tools.io.jsonl import JSONLAdapter
from robot_dataset_tools.io.rlds import RLDSAdapter
from robot_dataset_tools.io.robodm import RoboDMAdapter
from robot_dataset_tools.models import Episode, Step

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def make_episodes() -> list[Episode]:
    episodes: list[Episode] = []
    lengths = [4, 6, 5]
    instructions = [
        "pick up the red cube",
        "open the small drawer",
        "place the block in the tray",
    ]
    for episode_index, length in enumerate(lengths):
        steps: list[Step] = []
        for step_index in range(length):
            steps.append(
                Step(
                    index=step_index,
                    observation={
                        "state": [
                            float(episode_index),
                            float(step_index),
                            float(step_index) / max(1, length - 1),
                        ],
                        "proprio": {
                            "gripper": 1.0 if step_index < length - 1 else 0.0,
                            "joint_0": float(episode_index + step_index) / 10.0,
                        },
                    },
                    action=[
                        float(step_index),
                        float(step_index + 1),
                        float(episode_index),
                    ],
                    reward=1.0 if step_index == length - 1 else 0.0,
                    discount=1.0,
                    is_first=step_index == 0,
                    is_last=step_index == length - 1,
                    is_terminal=step_index == length - 1,
                    language_instruction=instructions[episode_index],
                    timestamp=float(step_index) * 0.1,
                )
            )
        episodes.append(
            Episode(
                episode_id=f"demo_ep_{episode_index}",
                metadata={"split": "demo", "task_index": episode_index},
                steps=steps,
            )
        )
    return episodes


def main() -> None:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    episodes = make_episodes()
    metadata = {
        "name": "robotds-demo",
        "description": "Tiny deterministic synthetic robot-learning dataset.",
        "num_episodes": len(episodes),
    }

    hdf5_path = EXAMPLES / "demo_hdf5.h5"
    jsonl_path = EXAMPLES / "demo_jsonl"
    rlds_path = EXAMPLES / "demo_rlds_lite"
    robodm_path = EXAMPLES / "demo_robodm.robodm.zip"

    HDF5Adapter().write_episodes(hdf5_path, episodes, metadata=metadata, overwrite=True)
    JSONLAdapter().write_episodes(jsonl_path, episodes, metadata=metadata, overwrite=True)
    RLDSAdapter().write_episodes(rlds_path, episodes, metadata=metadata, overwrite=True)
    RoboDMAdapter().write_episodes(robodm_path, episodes, metadata=metadata, overwrite=True)

    report = build_audit_report("examples/demo_hdf5.h5", "hdf5", episodes)
    write_json_report(report, EXAMPLES / "sample_audit_report.json")

    manifest = {
        "generated": [
            str(hdf5_path.relative_to(ROOT)),
            str(jsonl_path.relative_to(ROOT)),
            str(rlds_path.relative_to(ROOT)),
            str(robodm_path.relative_to(ROOT)),
            "examples/sample_audit_report.json",
        ]
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
