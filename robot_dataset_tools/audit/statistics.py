"""Statistics over canonical episodes."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from robot_dataset_tools.models import Episode, flatten_keys
from robot_dataset_tools.utils.arrays import json_safe


def compute_statistics(episodes: list[Episode]) -> dict[str, Any]:
    lengths = np.asarray([len(ep.steps) for ep in episodes], dtype=float)
    actions: list[np.ndarray] = []
    action_shapes: Counter[str] = Counter()
    obs_counts: Counter[str] = Counter()
    lang_episode_ids = set()
    lang_steps = 0
    empty_lang = 0
    unique_lang: set[str] = set()
    total_steps = 0
    for ep in episodes:
        ep_has_lang = False
        for step in ep.steps:
            total_steps += 1
            for key in flatten_keys(step.observation):
                obs_counts[key] += 1
            arr = step.action_array()
            if arr is not None:
                flat = arr.reshape(-1)
                actions.append(flat)
                action_shapes[str(tuple(arr.shape))] += 1
            if step.language_instruction is not None:
                lang_steps += 1
                normalized = " ".join(str(step.language_instruction).split())
                if normalized:
                    unique_lang.add(normalized)
                    ep_has_lang = True
                else:
                    empty_lang += 1
        if ep_has_lang:
            lang_episode_ids.add(ep.episode_id)
    action_stats: dict[str, Any] = {"available_steps": len(actions), "shapes": dict(action_shapes)}
    if actions:
        max_dim = max(a.size for a in actions)
        matrix = np.full((len(actions), max_dim), np.nan)
        for i, arr in enumerate(actions):
            matrix[i, : arr.size] = arr
        raw_values = np.concatenate(actions) if actions else np.asarray([], dtype=float)
        action_stats.update(
            {
                "dimensionality": int(max_dim),
                "min": _per_dimension(matrix, "min"),
                "max": _per_dimension(matrix, "max"),
                "mean": _per_dimension(matrix, "mean"),
                "std": _per_dimension(matrix, "std"),
                "nan_count": int(np.isnan(raw_values).sum()),
                "inf_count": int(np.isinf(raw_values).sum()),
            }
        )
    else:
        action_stats.update({"dimensionality": 0, "nan_count": 0, "inf_count": 0})
    return json_safe(
        {
            "episode_lengths": _length_stats(lengths),
            "actions": action_stats,
            "language": {
                "episodes_with_instruction": len(lang_episode_ids),
                "steps_with_instruction": lang_steps,
                "empty_instructions": empty_lang,
                "unique_instruction_count": len(unique_lang),
                "coverage_episode_fraction": len(lang_episode_ids) / len(episodes) if episodes else 0.0,
                "coverage_step_fraction": lang_steps / total_steps if total_steps else 0.0,
            },
            "observations": {
                "keys": sorted(obs_counts),
                "coverage": {k: v / total_steps if total_steps else 0.0 for k, v in sorted(obs_counts.items())},
            },
        }
    )


def _length_stats(lengths: np.ndarray) -> dict[str, Any]:
    if lengths.size == 0:
        return {k: 0 for k in ["min", "max", "mean", "median", "p50", "p90", "p95", "p99"]}
    return {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
    }


def _per_dimension(matrix: np.ndarray, stat: str) -> list[float | None]:
    values: list[float | None] = []
    for dim in range(matrix.shape[1]):
        column = matrix[:, dim]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            values.append(None)
        elif stat == "min":
            values.append(float(np.min(finite)))
        elif stat == "max":
            values.append(float(np.max(finite)))
        elif stat == "mean":
            values.append(float(np.mean(finite)))
        elif stat == "std":
            values.append(float(np.std(finite)))
        else:
            raise ValueError(f"Unknown statistic: {stat}")
    return values
