# robot-dataset-tools

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

A tested CLI for auditing, converting, visualizing, and benchmarking robot-learning datasets across HDF5, LeRobot-style, RLDS-lite, Robo-DM-style, and JSONL formats.

`robot-dataset-tools` is built for local dataset QA: catching schema issues, summarizing actions and episode lengths, checking language-instruction coverage, verifying conversions, and measuring read throughput before a dataset enters a training pipeline.

## Features

- Detect dataset formats automatically with explicit override support.
- Audit dataset health with JSON and HTML reports.
- Validate schema and consistency with structured issue codes.
- Convert through one canonical episode/step representation.
- Verify conversions for episode counts, lengths, actions, and language instructions.
- Generate Matplotlib visualizations for episode lengths, actions, language coverage, and state trajectories.
- Benchmark streaming read throughput and memory.
- Keep TensorFlow/TFDS, Robo-DM, Parquet, and video-heavy dependencies optional.

## Format Support

| Format | Base support | Optional support | Notes |
| --- | --- | --- | --- |
| HDF5 | Read/write practical robot-learning layouts | None | Supports `/episodes/{id}`, `/data/{id}`, and flat single-episode files. |
| JSONL | Read/write canonical interchange format | None | Useful for debugging, fixtures, and portable conversion tests. |
| LeRobot-style | Local JSONL step tables and portable write layout | Parquet with `pandas`/`pyarrow` | Video references can be preserved; video decoding is not required. |
| RLDS / RLDS-lite | Portable RLDS-lite directory | Optional TFDS dependency detection | Base package does not claim arbitrary native TFDS builder parsing. |
| Robo-DM-style portable | `.robodm.jsonl` and `.robodm.zip` | Optional native library hook | Base package does not implement binary Robo-DM/EBML. |

## Quickstart

```bash
git clone <repo-url>
cd robot-dataset-tools

python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"

robotds --help
pytest -q
```

If your shell does not expose `robotds`, use the module form:

```bash
python3 -m robot_dataset_tools.cli --help
```

## Installation

Editable local install:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Optional extras:

```bash
python3 -m pip install -e ".[parquet]"    # LeRobot-style Parquet tables
python3 -m pip install -e ".[benchmark]"  # psutil RSS memory measurements
python3 -m pip install -e ".[rlds]"       # optional TFDS dependency checks
```

This project is not claiming PyPI availability yet. Release/build commands are included for maintainers and local verification.

## CLI Overview

```bash
robotds detect PATH
robotds audit PATH --out report.json --html report.html
robotds validate PATH --strict
robotds convert INPUT OUTPUT --dst jsonl --verify --overwrite
robotds visualize PATH --out reports/viz
robotds benchmark PATH --json reports/benchmark.json
robotds doctor
```

Exit codes:

- `0`: success
- `1`: validation or conversion verification failed
- `2`: usage/input/unknown format error
- `3`: optional dependency missing for the requested operation

## Example Workflow

Generate tiny deterministic demo datasets:

```bash
python3 scripts/create_demo_data.py
```

Then run the same workflow you would use on a real dataset:

```bash
mkdir -p reports outputs

robotds detect examples/demo_hdf5.h5
robotds audit examples/demo_hdf5.h5 --out reports/audit.json --html reports/audit.html
robotds validate examples/demo_hdf5.h5
robotds convert examples/demo_hdf5.h5 outputs/demo_jsonl --dst jsonl --verify --overwrite
robotds visualize examples/demo_hdf5.h5 --out reports/viz
robotds benchmark examples/demo_hdf5.h5 --json reports/benchmark.json
robotds doctor
```

Typical audit output from the demo fixture:

```text
Dataset audit
detected_format         hdf5
episode_count           3
step_count              15
errors                  0
warnings                0
action_dimensionality   3
language_step_coverage  100.00%
```

## Sample Audit Report

The demo generator writes `examples/sample_audit_report.json`. A shortened excerpt:

```json
{
  "dataset_path": "examples/demo_hdf5.h5",
  "detected_format": "hdf5",
  "episode_count": 3,
  "step_count": 15,
  "episode_lengths": {
    "min": 4,
    "max": 6,
    "mean": 5.0,
    "median": 5.0
  },
  "action_stats": {
    "dimensionality": 3,
    "nan_count": 0,
    "inf_count": 0
  },
  "language_instruction_coverage": {
    "episodes_with_instruction": 3,
    "steps_with_instruction": 15,
    "coverage_step_fraction": 1.0
  }
}
```

Reports are strict JSON-safe artifacts. NumPy arrays become lists, NumPy scalars become Python scalars, and non-finite numeric values are represented safely rather than emitted as invalid JSON.

## Visualization Outputs

`robotds visualize` writes an `index.html` plus PNG plots such as:

- `episode_lengths.png`
- `action_distribution.png`
- `language_coverage.png`
- `sampled_episode_<id>_actions.png`
- `sampled_episode_<id>_state_trajectory.png` when state observations exist

The repository does not commit generated PNGs by default; generate them locally with:

```bash
robotds visualize examples/demo_hdf5.h5 --out reports/viz
```

## Conversion Verification

`robotds convert --verify` reads the destination back into the canonical model and compares:

- episode count
- step count
- episode lengths
- action shapes and values with tolerance
- normalized language instructions

Verification failures produce structured issue codes such as `EPISODE_COUNT_MISMATCH`, `EPISODE_LENGTH_MISMATCH`, `ACTION_VALUE_MISMATCH`, and `LANGUAGE_MISMATCH`.

## Benchmarking

`robotds benchmark` streams episodes where possible and reports:

- episodes/sec
- steps/sec
- wall-clock time
- bytes read estimate
- peak RSS delta when `psutil` is installed
- Python/platform/package metadata

Without `psutil`, memory fields degrade gracefully to `null`.

## Canonical Model

All adapters map to:

- `Episode`: `episode_id`, metadata, and ordered `Step` values
- `Step`: index, observation, action, reward, discount, flags, language instruction, and timestamp
- `Issue`: structured validation records with severity, stable code, message, and optional location

The shared model keeps validation, reporting, visualization, conversion, and benchmarking independent of the source format.

## Project Structure

```text
robot_dataset_tools/
  audit/        schema validation, statistics, reports
  benchmark/    streaming benchmark runner
  convert/      conversion engine and correctness checks
  io/           HDF5, JSONL, LeRobot-style, RLDS, Robo-DM adapters
  utils/        array, memory, sampling, hashing helpers
  visualize/    Matplotlib plot generation
tests/          synthetic fixtures and regression tests
scripts/        demo dataset generation
examples/       example documentation and generated sample report
docs/           architecture notes
```

## Design Notes

- The canonical representation is the contract between adapters and tools.
- Optional heavy dependencies fail with clear actionable messages.
- Base tests use tiny synthetic fixtures and require no network access.
- Portable RLDS-lite and Robo-DM-style formats exist for deterministic local development and conversion verification.
- Reports avoid embedding large arrays.

See [docs/architecture.md](docs/architecture.md) for more detail.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ".[dev]"

ruff check .
ruff format --check .
pytest -q
pytest --cov=robot_dataset_tools --cov-report=term-missing
python3 -m build
python3 -m twine check dist/*
```

Convenience targets:

```bash
make install
make demo
make lint
make test
make coverage
make build
```

## Testing

The suite covers format detection, adapters, audit reports, conversion correctness, CLI smoke tests, visualization generation, benchmark output, JSON serialization, and edge cases such as empty datasets, zero-length episodes, inconsistent actions, missing fields, and optional dependency failures.

```bash
pytest -q
```

## Roadmap

- Native RLDS/TFDS adapter support for specific installed builders.
- Deeper LeRobot metadata coverage and optional video frame inspection.
- Richer HTML report templates with linked visualization artifacts.
- More benchmark modes for large datasets and cold/warm cache comparisons.
- Optional static typing checks once the public API stabilizes further.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please keep tests local/offline, fixtures tiny, and optional dependency behavior explicit.

## License

MIT. See [LICENSE](LICENSE).
