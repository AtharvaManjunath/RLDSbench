"""Schema validation for canonical episodes."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from robot_dataset_tools.models import Episode, Issue


def validate_episodes(episodes: Iterable[Episode], strict: bool = False) -> list[Issue]:
    issues: list[Issue] = []
    for episode in episodes:
        if not episode.episode_id:
            issues.append(Issue("error", "MISSING_EPISODE_ID", "Episode is missing an id."))
        if not episode.steps:
            issues.append(
                Issue("error", "ZERO_LENGTH_EPISODE", "Episode contains no steps.", episode_id=episode.episode_id)
            )
            continue
        last_index = None
        last_timestamp = None
        action_shape = None
        for pos, step in enumerate(episode.steps):
            if step.index is None:
                issues.append(
                    Issue("warning", "MISSING_STEP_INDEX", "Step index was missing.", episode.episode_id, pos, "index")
                )
            elif last_index is not None and step.index <= last_index:
                issues.append(
                    Issue(
                        "error",
                        "NON_MONOTONIC_STEP_INDEX",
                        "Step indices are not strictly increasing.",
                        episode.episode_id,
                        step.index,
                        "index",
                    )
                )
            last_index = step.index
            if step.action is None:
                issues.append(
                    Issue(
                        "error" if strict else "warning",
                        "MISSING_ACTION",
                        "Step is missing an action.",
                        episode.episode_id,
                        step.index,
                        "action",
                    )
                )
            else:
                arr = step.action_array()
                if arr is None:
                    issues.append(
                        Issue(
                            "error",
                            "INVALID_ACTION",
                            "Action is not numeric.",
                            episode.episode_id,
                            step.index,
                            "action",
                        )
                    )
                else:
                    if action_shape is None:
                        action_shape = arr.shape
                    elif arr.shape != action_shape:
                        issues.append(
                            Issue(
                                "error",
                                "INCONSISTENT_ACTION_SHAPE",
                                f"Action shape {arr.shape} differs from {action_shape}.",
                                episode.episode_id,
                                step.index,
                                "action",
                            )
                        )
                    if np.isnan(arr).any():
                        issues.append(
                            Issue(
                                "error",
                                "NAN_IN_ACTION",
                                "Action contains NaN.",
                                episode.episode_id,
                                step.index,
                                "action",
                            )
                        )
                    if np.isinf(arr).any():
                        issues.append(
                            Issue(
                                "error",
                                "INF_IN_ACTION",
                                "Action contains Inf.",
                                episode.episode_id,
                                step.index,
                                "action",
                            )
                        )
            if not step.observation:
                issues.append(
                    Issue(
                        "error" if strict else "warning",
                        "MISSING_OBSERVATION",
                        "Step is missing observations.",
                        episode.episode_id,
                        step.index,
                        "observation",
                    )
                )
            if step.timestamp is not None:
                ts = float(step.timestamp)
                if last_timestamp is not None and ts < last_timestamp:
                    issues.append(
                        Issue(
                            "error",
                            "NON_MONOTONIC_TIMESTAMPS",
                            "Timestamps decrease within episode.",
                            episode.episode_id,
                            step.index,
                            "timestamp",
                        )
                    )
                last_timestamp = ts
            if step.language_instruction is not None:
                if not isinstance(step.language_instruction, str):
                    issues.append(
                        Issue(
                            "error",
                            "INVALID_LANGUAGE_INSTRUCTION",
                            "Language instruction must be a string.",
                            episode.episode_id,
                            step.index,
                            "language_instruction",
                        )
                    )
                elif not step.language_instruction.strip():
                    issues.append(
                        Issue(
                            "warning",
                            "EMPTY_LANGUAGE_INSTRUCTION",
                            "Language instruction is empty.",
                            episode.episode_id,
                            step.index,
                            "language_instruction",
                        )
                    )
        if episode.steps[0].is_first is False:
            issues.append(
                Issue(
                    "warning",
                    "FIRST_FLAG_FALSE",
                    "First step has is_first=False.",
                    episode.episode_id,
                    episode.steps[0].index,
                    "is_first",
                )
            )
        if episode.steps[-1].is_last is False:
            issues.append(
                Issue(
                    "warning",
                    "LAST_FLAG_FALSE",
                    "Final step has is_last=False.",
                    episode.episode_id,
                    episode.steps[-1].index,
                    "is_last",
                )
            )
    return issues
