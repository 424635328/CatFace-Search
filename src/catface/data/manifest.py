"""Dataset manifests: one immutable table describing every usable face crop.

A manifest is the contract between data preparation and every later stage. It is
written as JSON Lines so it diffs cleanly, streams without loading everything into
memory, and survives partial writes. Each row records not just *what* an image is
but *how* it was produced (source crop, detector, alignment, quality flags) — that
provenance is what allows a benchmark failure to be attributed to the data rather
than to the model.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DataError

MANIFEST_VERSION = 2


@dataclass
class FaceRecord:
    """One cropped face image plus its provenance and identity label."""

    image_id: str
    """Stable, unique key (``<source>:<basename>``). Survives path moves."""
    path: str
    """Absolute path to the cropped square face image."""
    source: str
    """Dataset the crop came from, e.g. ``oiid_cat``, ``calfw``."""
    identity: str | None
    """Identity label; ``None`` for unlabeled corpora (used for pre-training only)."""
    breed: str | None = None
    source_image: str | None = None
    """Original pre-crop image (traceability back to the raw dataset)."""
    bbox_xyxy: tuple[int, int, int, int] | None = None
    """Detector/annotation box in *source image* coordinates."""
    detector: str = "unknown"
    alignment: str = "none"
    quality: dict[str, float] = field(default_factory=dict)
    """Numeric quality signals (sharpness, luminance, face fraction)."""
    flags: tuple[str, ...] = ()
    """Non-fatal warnings, e.g. ``blurry``, ``too_small``, ``duplicate``."""
    sha1: str | None = None
    """Hash of the crop bytes; used for leakage-proof split assignment."""
    width: int = 0
    height: int = 0

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox_xyxy"] = list(self.bbox_xyxy) if self.bbox_xyxy else None
        payload["flags"] = list(self.flags)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> FaceRecord:
        data = dict(payload)
        bbox = data.get("bbox_xyxy")
        data["bbox_xyxy"] = tuple(bbox) if bbox else None
        data["flags"] = tuple(data.get("flags", ()))
        # ``dataclasses.fields`` rather than ``__slots__``: the latter only exists because
        # ``slots=True`` was requested, which is a Python 3.10+ option this package cannot use
        # while it supports 3.9. Deriving the field set from the dataclass itself is both more
        # portable and the actual source of truth.
        known = {f.name for f in dataclass_fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise DataError(f"Manifest row has unknown field(s): {unknown}")
        return cls(**data)

    @property
    def is_labeled(self) -> bool:
        return bool(self.identity)


@dataclass
class ManifestStats:
    """Aggregate description of a manifest, logged and stored for review."""

    total: int = 0
    labeled: int = 0
    unlabeled: int = 0
    identities: int = 0
    images_per_identity: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)
    flag_counts: dict[str, int] = field(default_factory=dict)
    singleton_identities: int = 0

    @property
    def usable_for_metric_learning(self) -> int:
        """Identities with at least two images — the minimum for positive pairs."""
        return sum(1 for count in self.images_per_identity.values() if count >= 2)

    def summary(self) -> dict[str, Any]:
        counts = sorted(self.images_per_identity.values(), reverse=True)
        return {
            "total": self.total,
            "labeled": self.labeled,
            "unlabeled": self.unlabeled,
            "identities": self.identities,
            "identities_with_ge2": self.usable_for_metric_learning,
            "singleton_identities": self.singleton_identities,
            "median_images_per_identity": float(np.median(counts)) if counts else 0.0,
            "max_images_per_identity": counts[0] if counts else 0,
            "sources": dict(sorted(self.sources.items())),
            "flags": dict(sorted(self.flag_counts.items())),
        }


def sha1_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hash a file's bytes; used for duplicate detection across corpora."""
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class Manifest:
    """In-memory collection of :class:`FaceRecord` with JSONL persistence."""

    def __init__(self, records: Sequence[FaceRecord] = ()) -> None:
        self.records: list[FaceRecord] = list(records)
        self._by_id: dict[str, FaceRecord] = {r.image_id: r for r in self.records}
        if len(self._by_id) != len(self.records):
            duplicates = [k for k, v in Counter(r.image_id for r in self.records).items() if v > 1]
            raise DataError(f"Duplicate image_id values in manifest: {duplicates[:5]}")

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[FaceRecord]:
        return iter(self.records)

    def __getitem__(self, index: int) -> FaceRecord:
        return self.records[index]

    def by_id(self, image_id: str) -> FaceRecord:
        try:
            return self._by_id[image_id]
        except KeyError as exc:
            raise DataError(f"Unknown image_id: {image_id!r}") from exc

    # -- filtering ----------------------------------------------------------
    def filter(
        self,
        sources: Iterable[str] | None = None,
        min_identity_images: int = 1,
        require_label: bool = False,
        exclude_flags: Iterable[str] = ("duplicate",),
        identities: Iterable[str] | None = None,
    ) -> Manifest:
        """Return a new manifest containing only rows matching every criterion."""
        source_set = set(sources) if sources else None
        identity_set = set(identities) if identities else None
        banned = set(exclude_flags)

        candidate = [
            r
            for r in self.records
            if (source_set is None or r.source in source_set)
            and (identity_set is None or (r.identity in identity_set))
            and not (banned & set(r.flags))
            and (not require_label or r.is_labeled)
        ]

        if min_identity_images > 1:
            counts = Counter(r.identity for r in candidate if r.identity)
            candidate = [r for r in candidate if r.identity and counts[r.identity] >= min_identity_images]

        return Manifest(candidate)

    # -- statistics ---------------------------------------------------------
    def stats(self) -> ManifestStats:
        counts = Counter(r.identity for r in self.records if r.identity)
        flags = Counter(flag for r in self.records for flag in r.flags)
        sources = Counter(r.source for r in self.records)
        return ManifestStats(
            total=len(self.records),
            labeled=sum(1 for r in self.records if r.is_labeled),
            unlabeled=sum(1 for r in self.records if not r.is_labeled),
            identities=len(counts),
            images_per_identity=dict(counts),
            sources=dict(sources),
            flag_counts=dict(flags),
            singleton_identities=sum(1 for c in counts.values() if c == 1),
        )

    def identity_counts(self) -> dict[str, int]:
        return dict(Counter(r.identity for r in self.records if r.identity))

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write JSONL. Existing files are replaced atomically."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"_manifest_version": MANIFEST_VERSION}) + "\n")
            for record in self.records:
                handle.write(json.dumps(record.as_json(), ensure_ascii=False) + "\n")
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        """Read JSONL, validating the version header."""
        source = Path(path)
        if not source.is_file():
            raise DataError(f"Manifest not found: {source}")
        records: list[FaceRecord] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if "_manifest_version" in payload:
                    version = payload["_manifest_version"]
                    if version != MANIFEST_VERSION:
                        raise DataError(
                            f"Manifest {source} has version {version}, this build expects {MANIFEST_VERSION}"
                        )
                    continue
                try:
                    records.append(FaceRecord.from_json(payload))
                except Exception as exc:
                    raise DataError(f"{source}:{line_number} is invalid: {exc}") from exc
        return cls(records)

    def extend(self, other: Manifest) -> Manifest:
        """Concatenate two manifests, rejecting id collisions."""
        return Manifest(self.records + other.records)


def assign_identity_splits(
    manifest: Manifest,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 1337,
    min_images_per_identity: int = 2,
) -> dict[str, str]:
    """Assign every *identity* to exactly one of train/val/test.

    Splitting by identity (never by image) is the single most important
    correctness property of an identity-retrieval benchmark: if two photos of the
    same cat land on opposite sides of the split, the test score measures
    memorisation instead of generalisation.

    Identities with fewer than ``min_images_per_identity`` usable images go to
    ``test`` when ``min_images_per_identity == 2`` (they can still be queried, they
    simply cannot contribute positive training pairs) — callers usually filter
    first with :meth:`Manifest.filter`.

    Returns:
        Mapping ``identity -> split``.
    """
    counts = Counter(r.identity for r in manifest if r.identity)
    identities = sorted(i for i, c in counts.items() if c >= min_images_per_identity)
    if not identities:
        raise DataError("No identity has enough images to build an identity-level split")

    rng = np.random.default_rng(seed)
    shuffled = list(identities)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, round(n * train_ratio))
    n_val = round(n * val_ratio)
    # Guarantee at least one identity in each of val/test when the corpus allows.
    if n >= 3:
        n_train = min(n_train, n - 2)
        n_val = max(1, min(n_val, n - n_train - 1))

    assignment: dict[str, str] = {}
    for index, identity in enumerate(shuffled):
        if index < n_train:
            assignment[identity] = "train"
        elif index < n_train + n_val:
            assignment[identity] = "val"
        else:
            assignment[identity] = "test"
    return assignment


def assign_image_splits_hash(records: Sequence[FaceRecord], seed: int = 1337) -> dict[str, str]:
    """Deterministic per-image split for corpora that carry no identity labels.

    Used only for self-supervised or pair-based training, where identity grouping is
    unavailable. Hashes the content digest so re-running the pipeline on the same
    files reproduces the same split.
    """
    assignment: dict[str, str] = {}
    for record in records:
        key = f"{record.sha1 or record.image_id}:{seed}".encode()
        bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) % 100
        assignment[record.image_id] = "train" if bucket < 90 else ("val" if bucket < 95 else "test")
    return assignment


def write_split_files(assignment: Mapping[str, str], directory: str | Path) -> dict[str, Path]:
    """Persist split assignments as ``train.txt`` / ``val.txt`` / ``test.txt``."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[str]] = {}
    for key, split in assignment.items():
        grouped.setdefault(split, []).append(key)
    written: dict[str, Path] = {}
    for split, keys in grouped.items():
        path = target / f"{split}.txt"
        path.write_text("\n".join(sorted(keys)) + "\n", encoding="utf-8")
        written[split] = path
    return written


__all__ = [
    "MANIFEST_VERSION",
    "FaceRecord",
    "Manifest",
    "ManifestStats",
    "assign_identity_splits",
    "assign_image_splits_hash",
    "sha1_file",
    "write_split_files",
]
