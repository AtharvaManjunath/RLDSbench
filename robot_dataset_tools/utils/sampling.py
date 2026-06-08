"""Deterministic sampling helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def sample_items(items: Sequence[T], count: int, seed: int = 0) -> list[T]:
    if count >= len(items):
        return list(items)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(items)), count))
    return [items[i] for i in indices]
