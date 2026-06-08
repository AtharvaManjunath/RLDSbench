# Contributing

Thanks for taking a look at `robot-dataset-tools`. The project is intentionally small, local-first, and test-heavy.

## Development Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

Optional extras:

```bash
python3 -m pip install -e ".[parquet]"    # LeRobot-style Parquet support
python3 -m pip install -e ".[benchmark]"  # psutil memory measurements
python3 -m pip install -e ".[rlds]"       # optional TFDS dependency checks
```

## Quality Checks

```bash
ruff check .
ruff format --check .
pytest -q
pytest --cov=robot_dataset_tools --cov-report=term-missing
```

Use `ruff format .` before opening a pull request.

## Demo Data

Generate tiny local fixtures for manual testing:

```bash
python3 scripts/create_demo_data.py
```

The generated datasets are deterministic and small enough for examples. They do not require network access or optional heavyweight dependencies.

## Pull Requests

- Keep changes focused and covered by tests.
- Do not add large binary fixtures or network-dependent tests.
- Be explicit when a feature depends on optional libraries.
- Do not claim native RLDS, Robo-DM, or video support unless it is implemented and tested.
- Preserve clear user-facing errors for unknown formats and missing optional dependencies.

