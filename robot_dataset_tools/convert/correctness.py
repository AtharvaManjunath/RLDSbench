"""Conversion correctness checks."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from robot_dataset_tools.models import Episode, Issue


def compare_episode_streams(
    source: Iterable[Episode],
    destination: Iterable[Episode],
    atol: float = 1e-6,
    lossy_ok: bool = False,
) -> dict:
    src = list(source)
    dst = list(destination)
    issues: list[Issue] = []
    if len(src) != len(dst):
        issues.append(
            Issue("error", "EPISODE_COUNT_MISMATCH", f"Source has {len(src)} episodes, destination has {len(dst)}.")
        )
    for a, b in zip(src, dst, strict=False):
        if a.episode_id != b.episode_id:
            issues.append(
                Issue(
                    "warning",
                    "EPISODE_ID_MISMATCH",
                    f"Episode id differs: {a.episode_id} != {b.episode_id}.",
                    a.episode_id,
                )
            )
        if len(a.steps) != len(b.steps):
            issues.append(
                Issue(
                    "error",
                    "EPISODE_LENGTH_MISMATCH",
                    f"Episode length differs: {len(a.steps)} != {len(b.steps)}.",
                    a.episode_id,
                )
            )
            continue
        for sa, sb in zip(a.steps, b.steps, strict=False):
            aa, ab = sa.action_array(), sb.action_array()
            if aa is None and ab is not None or aa is not None and ab is None:
                issues.append(
                    Issue(
                        "error",
                        "ACTION_DROPPED",
                        "Action presence changed during conversion.",
                        a.episode_id,
                        sa.index,
                        "action",
                    )
                )
            elif aa is not None and ab is not None:
                if aa.shape != ab.shape:
                    issues.append(
                        Issue(
                            "error",
                            "ACTION_SHAPE_MISMATCH",
                            f"Action shape differs: {aa.shape} != {ab.shape}.",
                            a.episode_id,
                            sa.index,
                            "action",
                        )
                    )
                elif not np.allclose(aa, ab, atol=atol, equal_nan=True):
                    issues.append(
                        Issue(
                            "error",
                            "ACTION_VALUE_MISMATCH",
                            "Action values differ beyond tolerance.",
                            a.episode_id,
                            sa.index,
                            "action",
                        )
                    )
            if _norm(sa.language_instruction) != _norm(sb.language_instruction):
                issues.append(
                    Issue(
                        "error",
                        "LANGUAGE_MISMATCH",
                        "Language instruction changed.",
                        a.episode_id,
                        sa.index,
                        "language_instruction",
                    )
                )
    return {
        "passed": not any(issue.severity == "error" for issue in issues),
        "source_episode_count": len(src),
        "destination_episode_count": len(dst),
        "source_step_count": sum(len(ep.steps) for ep in src),
        "destination_step_count": sum(len(ep.steps) for ep in dst),
        "issues": [issue.to_dict() for issue in issues],
        "lossy_ok": lossy_ok,
    }


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", " ", value).strip()
