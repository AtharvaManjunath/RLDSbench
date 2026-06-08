"""Project exceptions."""

from __future__ import annotations


class RobotDatasetError(Exception):
    """Base exception for expected user-facing failures."""


class UnknownFormatError(RobotDatasetError):
    """Raised when no adapter can read a path."""


class OptionalDependencyError(RobotDatasetError):
    """Raised when a requested feature requires an optional dependency."""

    def __init__(self, dependency: str, feature: str) -> None:
        super().__init__(
            f"Optional dependency '{dependency}' is required for {feature}. "
            f"Install the relevant extra or choose a portable-compatible format."
        )
        self.dependency = dependency
        self.feature = feature


class OverwriteError(RobotDatasetError):
    """Raised when an output path exists and overwrite is false."""
