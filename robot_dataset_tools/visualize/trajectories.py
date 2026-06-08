"""Matplotlib visualizations for canonical episodes."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robot_dataset_tools.models import Episode
from robot_dataset_tools.registry import register_builtin_adapters
from robot_dataset_tools.utils.arrays import sanitize_id
from robot_dataset_tools.utils.sampling import sample_items


def generate_visualizations(
    path: str | Path,
    out_dir: str | Path,
    format_name: str = "auto",
    episodes_count: int = 3,
    seed: int = 0,
    max_steps: int | None = None,
    show_actions: bool = False,
    show_states: bool = False,
) -> dict[str, Any]:
    adapter = register_builtin_adapters().resolve(path, format_name)
    episodes = list(adapter.read_episodes(path))
    sampled = sample_items(episodes, min(episodes_count, len(episodes)), seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    warnings: list[str] = []
    _plot_lengths(episodes, out / "episode_lengths.png")
    files.append("episode_lengths.png")
    if any(step.action is not None for ep in episodes for step in ep.steps):
        _plot_action_distribution(episodes, out / "action_distribution.png")
        files.append("action_distribution.png")
    else:
        warnings.append("No actions found; skipped action distribution.")
    _plot_language(episodes, out / "language_coverage.png")
    files.append("language_coverage.png")
    for ep in sampled:
        sid = sanitize_id(ep.episode_id)
        if show_actions or any(step.action is not None for step in ep.steps):
            target = out / f"sampled_episode_{sid}_actions.png"
            if _plot_episode_actions(ep, target, max_steps):
                files.append(target.name)
        if show_states or any("state" in step.observation for step in ep.steps):
            target = out / f"sampled_episode_{sid}_state_trajectory.png"
            if _plot_state(ep, target, max_steps):
                files.append(target.name)
    _write_index(out, files, warnings)
    return {"out_dir": str(out), "files": files, "warnings": warnings}


def _plot_lengths(episodes: list[Episode], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist([len(ep.steps) for ep in episodes], bins=max(1, min(10, len(episodes))))
    plt.xlabel("Episode length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _action_matrix(episodes: list[Episode]) -> np.ndarray | None:
    rows = []
    for ep in episodes:
        for step in ep.steps:
            arr = step.action_array()
            if arr is not None:
                rows.append(arr.reshape(-1))
    if not rows:
        return None
    dim = max(r.size for r in rows)
    matrix = np.full((len(rows), dim), np.nan)
    for i, row in enumerate(rows):
        matrix[i, : row.size] = row
    return matrix


def _plot_action_distribution(episodes: list[Episode], path: Path) -> None:
    matrix = _action_matrix(episodes)
    if matrix is None:
        return
    plt.figure(figsize=(7, 4))
    plt.boxplot([matrix[:, i][~np.isnan(matrix[:, i])] for i in range(matrix.shape[1])])
    plt.xlabel("Action dimension")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_language(episodes: list[Episode], path: Path) -> None:
    with_lang = sum(1 for ep in episodes if any((s.language_instruction or "").strip() for s in ep.steps))
    without = max(0, len(episodes) - with_lang)
    plt.figure(figsize=(5, 4))
    plt.bar(["with", "without"], [with_lang, without], color=["#2f855a", "#718096"])
    plt.ylabel("Episodes")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_episode_actions(episode: Episode, path: Path, max_steps: int | None) -> bool:
    rows = []
    for step in episode.steps[:max_steps]:
        arr = step.action_array()
        if arr is not None:
            rows.append(arr.reshape(-1))
    if not rows:
        return False
    dim = max(r.size for r in rows)
    matrix = np.full((len(rows), dim), np.nan)
    for i, row in enumerate(rows):
        matrix[i, : row.size] = row
    plt.figure(figsize=(8, 4))
    for d in range(matrix.shape[1]):
        plt.plot(matrix[:, d], label=f"a{d}")
    plt.xlabel("Step")
    plt.ylabel("Action")
    if matrix.shape[1] <= 8:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


def _plot_state(episode: Episode, path: Path, max_steps: int | None) -> bool:
    states = []
    for step in episode.steps[:max_steps]:
        state = step.observation.get("state")
        if state is not None:
            states.append(np.asarray(state, dtype=float).reshape(-1))
    if not states:
        return False
    dim = max(s.size for s in states)
    matrix = np.full((len(states), dim), np.nan)
    for i, row in enumerate(states):
        matrix[i, : row.size] = row
    plt.figure(figsize=(7, 5))
    if matrix.shape[1] >= 2:
        plt.plot(matrix[:, 0], matrix[:, 1], marker="o")
        plt.xlabel("state[0]")
        plt.ylabel("state[1]")
    else:
        plt.plot(matrix[:, 0])
        plt.xlabel("Step")
        plt.ylabel("state[0]")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


def _write_index(out: Path, files: list[str], warnings: list[str]) -> None:
    links = "\n".join(f"<li><a href='{html.escape(name)}'>{html.escape(name)}</a></li>" for name in files)
    warn = "".join(f"<p>{html.escape(w)}</p>" for w in warnings)
    (out / "index.html").write_text(
        f"<html><body><h1>Robot Dataset Visualizations</h1>{warn}<ul>{links}</ul></body></html>", encoding="utf-8"
    )
