# Architecture

`robot-dataset-tools` is built around a canonical stream of `Episode` objects. Format adapters translate external dataset layouts into that stream, and the audit, conversion, visualization, and benchmark layers operate on the shared model.

## Canonical Data Model

- `Step`: one timestep with observation, action, reward, discount, flags, language instruction, and timestamp.
- `Episode`: a stable `episode_id`, metadata, and ordered `Step` values.
- `DatasetSummary`: lightweight counts and metadata for UI/reporting.
- `Issue`: structured validation output with severity, stable code, message, and optional location.

The model is intentionally permissive. Missing optional fields are represented as `None` and surfaced as report issues rather than unexpected crashes.

## Adapter Interface

Every adapter implements:

- `can_read(path)`
- `read_episodes(path, limit=None, sample=None)`
- `summarize(path)`
- `write_episodes(path, episodes, metadata=None, overwrite=False)`
- `validate(path)`

Adapters are registered in a small registry used by `--format auto` detection and explicit format selection.

## Portable Compatibility Formats

Full RLDS/TFDS and Robo-DM deployments can require heavyweight or project-specific dependencies. The base package therefore includes:

- RLDS-lite: `dataset_info.json` plus canonical `episodes.jsonl`.
- Robo-DM portable: `.robodm.jsonl` or `.robodm.zip` with metadata and episode payloads.

These formats provide deterministic local conversion tests and useful interchange without pretending to implement every native binary detail.

## Streaming and Materialization

Adapters expose episode iterators so validation and benchmarking can stream through datasets. Some higher-level operations, such as visualizations and full conversion verification, intentionally materialize selected episodes because they need cross-episode statistics or comparison.

## Conversion Verification

The verification layer compares canonical source and destination streams. It checks episode counts, step counts, episode lengths, action shapes, action values, and normalized language instructions. Mismatches are returned as structured issue records.

## JSON-Safe Serialization

Reports and portable formats must be strict JSON. NumPy arrays become lists, NumPy scalars become Python scalars, and non-finite floats become `null` in JSON artifacts. Large arrays are not embedded in reports.

## Optional Dependency Strategy

Heavy or ecosystem-specific dependencies are optional. If an adapter needs an unavailable package, it raises a clear `OptionalDependencyError` rather than exposing a raw import stack trace.

