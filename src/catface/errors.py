"""Exception hierarchy for CatFace Search.

A single root exception makes it possible for the CLI to translate *expected*
failures (missing dataset, checksum mismatch, incompatible checkpoint) into a
clean diagnostic exit code, while unexpected exceptions still surface as
tracebacks for debugging.
"""

from __future__ import annotations


class CatFaceError(Exception):
    """Base class for every error raised deliberately by this project."""


class ConfigError(CatFaceError):
    """Configuration is missing, malformed, or internally inconsistent."""


class DataError(CatFaceError):
    """Dataset is missing, malformed, or fails integrity validation."""


class ArtifactError(CatFaceError):
    """A checkpoint / index artifact is missing, corrupt, or incompatible."""


class ModelError(CatFaceError):
    """A backbone or head could not be constructed as requested."""


class BenchmarkError(CatFaceError):
    """An evaluation protocol is not satisfiable with the supplied data."""


__all__ = [
    "ArtifactError",
    "BenchmarkError",
    "CatFaceError",
    "ConfigError",
    "DataError",
    "ModelError",
]
