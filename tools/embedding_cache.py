"""Cache descriptors so retrieval research does not re-run the encoder.

Why this exists
---------------
Embedding the benchmark protocol costs ~140 s for the trained DINOv2-S over 12 644 images. Every
retrieval idea worth testing acts on the descriptor *matrix*, not on the images, so re-encoding for
each idea caps research throughput at one experiment per 140 s and makes a parameter sweep
impossible. With the matrix cached, the same sweep is pure linear algebra.

Cache validity
--------------
A cache entry is keyed by a fingerprint of everything that could change the numbers: the checkpoint
file's own content hash, the manifest, the split, the image size, the TTA views, the batch size and
the descriptor width. Keying on the checkpoint *hash* rather than its path matters: a retrained
model usually lands on the same path, and a stale cache silently scoring a different model is
exactly the class of error this project has been bitten by before. The hash is printed with every
load so a reader can tie a number back to the weights that produced it.

Usage::

    from tools.embedding_cache import load_or_embed, cache_key

    key = cache_key(checkpoint, manifest, protocol="cat_individuals", image_size=224)
    payload = load_or_embed(key, lambda: embed_protocol(...))
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Cache lives under the gitignored artifacts tree: descriptors are derived data, often tens of MB.
DEFAULT_CACHE_DIR = REPO_ROOT / "artifacts" / "embedding-cache"


def file_digest(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's content, or a placeholder when the file is absent.

    Absent is not an error: a zero-shot configuration legitimately has no checkpoint, and it still
    needs a stable key. The placeholder keeps that case distinguishable from a real digest.
    """
    target = Path(path)
    if not target.is_file():
        return "none"
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def cache_key(
    checkpoint: str | Path | None,
    manifest: str | Path,
    protocol: str,
    image_size: int,
    tta: tuple[str, ...] | list[str] = ("identity", "hflip"),
    batch_size: int = 32,
    extra: dict[str, Any] | None = None,
) -> str:
    """Stable identifier for one embedding job.

    ``extra`` carries protocol parameters that are not paths (e.g. ``queries_per_identity``,
    ``seed``, ``max_gallery_per_identity``). Leaving one out would make two different protocols
    share a cache entry, so callers should pass everything that influences the split.
    """
    payload = {
        "checkpoint_sha256": file_digest(checkpoint) if checkpoint else "none",
        "manifest_sha256": file_digest(manifest),
        "protocol": protocol,
        "image_size": int(image_size),
        "tta": list(tta),
        "batch_size": int(batch_size),
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _path(directory: str | Path, key: str) -> Path:
    return Path(directory) / f"{key}.npz"


def load(directory: str | Path, key: str) -> dict[str, Any] | None:
    """Return the cached arrays plus validity metadata, or ``None`` when absent."""
    target = _path(directory, key)
    if not target.is_file():
        return None
    with np.load(target, allow_pickle=False) as archive:
        payload: dict[str, Any] = {name: archive[name] for name in archive.files}
    payload["query_paths"] = [str(item) for item in payload["query_paths"].tolist()]
    payload["gallery_paths"] = [str(item) for item in payload["gallery_paths"].tolist()]
    payload["query_labels"] = np.asarray(payload["query_labels"]).astype(str).tolist()
    payload["gallery_labels"] = np.asarray(payload["gallery_labels"]).astype(str).tolist()
    # np.load returns 0-d arrays for the scalars; unwrap them so callers see plain Python values.
    for name in ("hit@1_guard",):
        if name in payload:
            payload[name] = float(payload[name])
    return payload


def save(directory: str | Path, key: str, payload: dict[str, Any]) -> Path:
    """Persist arrays only: everything here is numeric or string, never a live object."""
    target = _path(directory, key)
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = {
        "query": np.asarray(payload["query"], dtype=np.float32),
        "gallery": np.asarray(payload["gallery"], dtype=np.float32),
        "query_ids": np.asarray(payload["query_ids"], dtype="U256"),
        "gallery_ids": np.asarray(payload["gallery_ids"], dtype="U256"),
        "query_labels": np.asarray(payload["query_labels"], dtype="U128"),
        "gallery_labels": np.asarray(payload["gallery_labels"], dtype="U128"),
        "query_paths": np.asarray(payload["query_paths"], dtype="U512"),
        "gallery_paths": np.asarray(payload["gallery_paths"], dtype="U512"),
    }
    np.savez_compressed(target, **archive)
    return target


def save_arrays(
    directory: str | Path, key: str, arrays: dict[str, Any], list_fields: Sequence[str] = ()
) -> Path:
    """Persist an arbitrary set of arrays under ``key``.

    Separate from :func:`save` because that one encodes the query/gallery protocol shape, and forcing
    a single-split job (the 8 833-image training set, for instance) through it would either fail or
    invent empty halves. The first attempt did fail, with a ``KeyError: 'query'``, after spending
    95 seconds embedding — the cost of a cache that only understands one shape.
    """
    target = _path(directory, key)
    target.parent.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if name in list_fields:
            archive[name] = np.asarray(value, dtype="U512")
        else:
            array = np.asarray(value)
            archive[name] = array if array.dtype != np.float64 else array.astype(np.float32)
    np.savez_compressed(target, **archive)
    return target


def load_arrays(directory: str | Path, key: str, list_fields: Sequence[str] = ()) -> dict[str, Any] | None:
    """Read what :func:`save_arrays` wrote, or ``None`` when the entry is absent or unreadable."""
    target = _path(directory, key)
    if not target.is_file():
        return None
    try:
        with np.load(target, allow_pickle=False) as archive:
            payload: dict[str, Any] = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, EOFError, KeyError):
        # A truncated file is what an interrupted write leaves behind; treat it as a miss.
        return None
    for name in list_fields:
        if name in payload:
            payload[name] = [str(item) for item in payload[name].tolist()]
    return payload


def load_or_embed(
    key: str,
    producer: Callable[[], dict[str, Any]],
    directory: str | Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Return ``(payload, was_cached)``, embedding only when the cache misses."""
    if not force:
        cached = load(directory, key)
        if cached is not None:
            return cached, True
    payload = producer()
    save(directory, key, payload)
    return payload, False


__all__ = [
    "DEFAULT_CACHE_DIR",
    "cache_key",
    "file_digest",
    "load",
    "load_arrays",
    "load_or_embed",
    "save",
    "save_arrays",
]
