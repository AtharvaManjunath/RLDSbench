"""Stable hashing helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from robot_dataset_tools.utils.arrays import json_safe


def stable_hash(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
