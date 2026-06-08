"""Typer CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from robot_dataset_tools.audit.report import build_audit_report, write_html_report, write_json_report
from robot_dataset_tools.audit.schema import validate_episodes
from robot_dataset_tools.benchmark.runner import run_benchmark
from robot_dataset_tools.convert.engine import convert_dataset
from robot_dataset_tools.errors import OptionalDependencyError, OverwriteError, RobotDatasetError, UnknownFormatError
from robot_dataset_tools.io.detect import detect_format
from robot_dataset_tools.registry import register_builtin_adapters
from robot_dataset_tools.utils.arrays import json_safe
from robot_dataset_tools.visualize.trajectories import generate_visualizations

app = typer.Typer(help="Audit, convert, visualize, and benchmark robot-learning datasets.")
console = Console()


@app.command()
def detect(path: Path, allow_unknown: bool = typer.Option(False, "--allow-unknown")) -> None:
    """Detect dataset format."""
    result = detect_format(path)
    if result.format == "unknown" and not allow_unknown:
        console.print(f"[red]Unknown format:[/red] {path}")
        raise typer.Exit(2)
    console.print(f"{result.format}\tconfidence={result.confidence:.2f}\t{result.reason}")


@app.command()
def audit(
    path: Path,
    format: str = typer.Option("auto", "--format"),
    out: Path | None = typer.Option(None, "--out"),
    html: Path | None = typer.Option(None, "--html"),
    sample_episodes: int | None = typer.Option(None, "--sample-episodes"),
    max_episodes: int | None = typer.Option(None, "--max-episodes"),
    strict: bool = typer.Option(False, "--strict"),
    quiet: bool = typer.Option(False, "--quiet"),
    benchmark: bool = typer.Option(False, "--benchmark"),
) -> None:
    """Generate a dataset health report."""
    try:
        adapter = register_builtin_adapters().resolve(path, format)
        episodes = list(adapter.read_episodes(path, limit=max_episodes or sample_episodes))
        bench = run_benchmark(path, adapter.format_name, max_episodes=max_episodes, repeat=1) if benchmark else None
        report = build_audit_report(str(path), adapter.format_name, episodes, strict=strict, benchmark=bench)
        if out:
            write_json_report(report, out)
        if html:
            write_html_report(report, html)
        if not quiet:
            _print_audit_summary(report)
        if strict and report["errors"]:
            raise typer.Exit(1)
    except OptionalDependencyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from None
    except (RobotDatasetError, OSError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None


@app.command()
def validate(
    path: Path,
    format: str = typer.Option("auto", "--format"),
    strict: bool = typer.Option(False, "--strict"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Validate schema and consistency."""
    try:
        adapter = register_builtin_adapters().resolve(path, format)
        issues = validate_episodes(adapter.read_episodes(path), strict=strict)
        if json_output:
            console.print(
                json.dumps(json_safe([issue.to_dict() for issue in issues]), allow_nan=False, indent=2, sort_keys=True)
            )
        else:
            _print_issues(issues)
        has_errors = any(issue.severity == "error" for issue in issues)
        has_warnings = any(issue.severity == "warning" for issue in issues)
        if has_errors or (strict and has_warnings):
            raise typer.Exit(1)
    except OptionalDependencyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from None
    except (RobotDatasetError, OSError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None


@app.command()
def convert(
    input: Path,
    output: Path,
    src: str = typer.Option("auto", "--src"),
    dst: str = typer.Option(..., "--dst"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    max_episodes: int | None = typer.Option(None, "--max-episodes"),
    verify: bool = typer.Option(False, "--verify"),
    verify_samples: int | None = typer.Option(None, "--verify-samples"),
    lossy_ok: bool = typer.Option(False, "--lossy-ok"),
    report: Path | None = typer.Option(None, "--report"),
) -> None:
    """Convert between dataset formats."""
    try:
        result = convert_dataset(
            input, output, src, dst, overwrite, max_episodes, verify, verify_samples, lossy_ok, report
        )
        console.print(json.dumps(json_safe(result), allow_nan=False, indent=2, sort_keys=True))
        if verify and not result.get("correctness", {}).get("passed", False):
            raise typer.Exit(1)
    except OptionalDependencyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from None
    except (OverwriteError, UnknownFormatError, OSError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None


@app.command()
def visualize(
    path: Path,
    format: str = typer.Option("auto", "--format"),
    out: Path = typer.Option(..., "--out"),
    episodes: int = typer.Option(3, "--episodes"),
    seed: int = typer.Option(0, "--seed"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    show_actions: bool = typer.Option(False, "--show-actions"),
    show_states: bool = typer.Option(False, "--show-states"),
) -> None:
    """Generate trajectory and dataset plots."""
    try:
        result = generate_visualizations(path, out, format, episodes, seed, max_steps, show_actions, show_states)
        console.print(f"Wrote {len(result['files'])} visualization files to {result['out_dir']}")
    except OptionalDependencyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from None
    except (RobotDatasetError, OSError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None


@app.command()
def benchmark(
    path: Path,
    format: str = typer.Option("auto", "--format"),
    max_episodes: int | None = typer.Option(None, "--max-episodes"),
    warmup_episodes: int = typer.Option(0, "--warmup-episodes"),
    repeat: int = typer.Option(1, "--repeat"),
    json_path: Path | None = typer.Option(None, "--json"),
) -> None:
    """Measure streaming read throughput and memory."""
    try:
        result = run_benchmark(path, format, max_episodes, warmup_episodes, repeat)
        if json_path:
            json_path.write_text(
                json.dumps(json_safe(result), allow_nan=False, indent=2, sort_keys=True), encoding="utf-8"
            )
        console.print(json.dumps(json_safe(result), allow_nan=False, indent=2, sort_keys=True))
    except OptionalDependencyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from None
    except (RobotDatasetError, OSError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None


@app.command()
def doctor() -> None:
    """Check installed optional dependencies."""
    deps = {
        "pandas": "LeRobot Parquet read/write",
        "pyarrow": "Parquet backend",
        "tensorflow_datasets": "Native RLDS/TFDS reading",
        "tensorflow": "Native TensorFlow dataset support",
        "psutil": "RSS memory measurement",
        "robodm": "Native Robo-DM reading",
    }
    table = Table(title="robotds doctor")
    table.add_column("Dependency")
    table.add_column("Available")
    table.add_column("Feature")
    for dep, feature in deps.items():
        try:
            __import__(dep)
            available = "yes"
        except ImportError:
            available = "no"
        table.add_row(dep, available, feature)
    console.print(table)


def _print_audit_summary(report: dict) -> None:
    table = Table(title="Dataset audit")
    table.add_column("Metric")
    table.add_column("Value")
    for key in ["detected_format", "episode_count", "step_count"]:
        table.add_row(key, str(report[key]))
    table.add_row("errors", str(len(report.get("errors", []))))
    table.add_row("warnings", str(len(report.get("warnings", []))))
    table.add_row("action_dimensionality", str(report["action_stats"].get("dimensionality", 0)))
    table.add_row("language_step_coverage", f"{report['language_instruction_coverage']['coverage_step_fraction']:.2%}")
    console.print(table)


def _print_issues(issues) -> None:
    if not issues:
        console.print("[green]No issues found.[/green]")
        return
    table = Table(title="Validation issues")
    for col in ["severity", "code", "episode", "step", "field", "message"]:
        table.add_column(col)
    for issue in issues:
        table.add_row(
            issue.severity,
            issue.code,
            issue.episode_id or "",
            "" if issue.step_index is None else str(issue.step_index),
            issue.field or "",
            issue.message,
        )
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
