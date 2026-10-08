"""Evaluation protocols.

A retrieval benchmark is only as trustworthy as its protocol, so the protocol is
explicit code rather than an implicit convention:

* **Query/gallery separation** — every identity contributes query images and gallery
  images, and a query never counts itself as a match.
* **Identity-disjointness** — train/val/test share no individual animals. Enforced at
  split time (``assign_identity_splits``) and re-verified here with an assertion, so a
  leaky split cannot silently produce a 0.99 score.
* **Gallery ratio** — reported, because a gallery of 1 image per identity is a much
  easier problem than 10.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..data.manifest import FaceRecord
from ..errors import BenchmarkError
from ..logging_utils import get_logger

LOGGER = get_logger("eval.protocol")


@dataclass
class Split:
    """A query/gallery partition of one manifest."""

    name: str
    query_records: list[FaceRecord]
    gallery_records: list[FaceRecord]
    query_labels: np.ndarray
    gallery_labels: np.ndarray
    """Identity arrays aligned with the record lists."""
    self_occurrences: np.ndarray | None = None
    """``(n_query, n_gallery)`` mask of (query, gallery) pairs with identical image id."""
    identity_overlap_with: dict[str, set[str]] = field(default_factory=dict)
    """Per source split, the identities that also appear in another split (debug aid)."""

    @property
    def num_queries(self) -> int:
        return len(self.query_records)

    @property
    def num_gallery(self) -> int:
        return len(self.gallery_records)

    def describe(self) -> dict[str, object]:
        unique_query_identities = len(set(self.query_labels.tolist()))
        unique_gallery_identities = len(set(self.gallery_labels.tolist()))
        shared = len(set(self.query_labels.tolist()) & set(self.gallery_labels.tolist()))
        per_identity = defaultdict(int)
        for label in self.gallery_labels:
            per_identity[label] += 1
        counts = np.array(sorted(per_identity.values()), dtype=np.float64)
        return {
            "name": self.name,
            "queries": self.num_queries,
            "gallery": self.num_gallery,
            "query_identities": unique_query_identities,
            "gallery_identities": unique_gallery_identities,
            "identities_in_both": shared,
            "gallery_images_per_identity_mean": float(counts.mean()) if counts.size else 0.0,
            "gallery_images_per_identity_median": float(np.median(counts)) if counts.size else 0.0,
            "gallery_images_per_identity_min": int(counts.min()) if counts.size else 0,
            "gallery_images_per_identity_max": int(counts.max()) if counts.size else 0,
        }


def build_identity_split(
    records: Sequence[FaceRecord],
    queries_per_identity: int = 1,
    max_gallery_per_identity: int | None = None,
    name: str = "identity",
    seed: int = 1337,
    require_min_identity_images: int = 2,
) -> Split:
    """Partition records so every identity with enough images is queryable.

    For each identity, ``queries_per_identity`` images become queries and the rest
    become gallery references. Identities with fewer than
    ``require_min_identity_images`` images cannot produce a positive pair and are
    dropped, with the number reported in the log.

    Args:
        records: Face records, all from one split (e.g. all ``test`` identities).
        queries_per_identity: Query images taken from each identity.
        max_gallery_per_identity: Optional cap, to stop a prolific identity from
            dominating the gallery.
        name: Label used in reports.
        seed: Sampling seed; fixes *which* image of an identity is the query.
        require_min_identity_images: Minimum images for an identity to participate.

    Raises:
        BenchmarkError: If no identity qualifies.
    """
    grouped: dict[str, list[FaceRecord]] = defaultdict(list)
    for record in records:
        if record.identity:
            grouped[record.identity].append(record)
    if not grouped:
        raise BenchmarkError("No labelled records supplied to build_identity_split")

    rng = np.random.default_rng(seed)
    query_records: list[FaceRecord] = []
    gallery_records: list[FaceRecord] = []
    skipped = 0

    for identity in sorted(grouped):
        members = sorted(grouped[identity], key=lambda r: r.image_id)
        if len(members) < require_min_identity_images:
            skipped += 1
            continue
        order = rng.permutation(len(members))
        shuffled = [members[i] for i in order]
        n_query = min(queries_per_identity, len(shuffled) - 1)
        identity_queries = shuffled[:n_query]
        identity_gallery = shuffled[n_query:]
        if max_gallery_per_identity is not None and len(identity_gallery) > max_gallery_per_identity:
            identity_gallery = identity_gallery[:max_gallery_per_identity]
        query_records.extend(identity_queries)
        gallery_records.extend(identity_gallery)

    if not query_records or not gallery_records:
        raise BenchmarkError(
            "Split construction produced no query/gallery pairs; every identity has "
            "too few images"
        )
    if skipped:
        LOGGER.warning(
            "Dropped %d identities with fewer than %d images (no positive pair possible)",
            skipped, require_min_identity_images,
        )

    query_labels = np.array([r.identity for r in query_records])
    gallery_labels = np.array([r.identity for r in gallery_records])
    self_mask = np.array(
        [[q.image_id == g.image_id for g in gallery_records] for q in query_records],
        dtype=bool,
    )

    shared = set(query_labels.tolist()) & set(gallery_labels.tolist())
    if not shared:
        raise BenchmarkError(
            "Query and gallery share no identities; no query can be evaluated"
        )

    return Split(
        name=name,
        query_records=query_records,
        gallery_records=gallery_records,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        self_occurrences=self_mask,
    )


def assert_identity_disjoint(
    splits: Mapping[str, Iterable[FaceRecord]],
    tolerance: int = 0,
) -> None:
    """Fail loudly if identities leak across splits.

    ``tolerance`` allows a small number of shared identities for corpora where the
    same animal genuinely appears in two source datasets; callers that need this
    must opt in so the leak is visible in review rather than accidental.
    """
    seen: dict[str, str] = {}
    overlaps: list[tuple[str, str, str]] = []
    for split_name, records in splits.items():
        for record in records:
            if not record.identity:
                continue
            previous = seen.get(record.identity)
            if previous is not None and previous != split_name:
                overlaps.append((record.identity, previous, split_name))
            seen[record.identity] = split_name

    if len(overlaps) > tolerance:
        sample = overlaps[:10]
        raise BenchmarkError(
            f"Identity leakage detected across splits ({len(overlaps)} identity/ies), "
            f"e.g. {sample}. Split by identity, not by image."
        )


def build_pair_split(
    records: Sequence[FaceRecord],
    pair_labels: Sequence[int],
    pair_paths_a: Sequence[str],
    pair_paths_b: Sequence[str],
    name: str = "verification",
) -> tuple[list[FaceRecord], np.ndarray]:
    """Validate a labelled verification-pair protocol.

    Only checks shape agreement and class balance; scoring happens in the runner so
    that a single embedding pass covers both protocols.
    """
    if not (len(pair_labels) == len(pair_paths_a) == len(pair_paths_b)):
        raise BenchmarkError(
            f"Pair arrays disagree: labels={len(pair_labels)}, a={len(pair_paths_a)}, "
            f"b={len(pair_paths_b)}"
        )
    labels = np.asarray(pair_labels, dtype=np.int64)
    if labels.size == 0:
        raise BenchmarkError("Empty verification protocol")
    if labels.min() == labels.max():
        raise BenchmarkError("Verification protocol has only one class")
    return list(records), labels


__all__ = [
    "Split",
    "assert_identity_disjoint",
    "build_identity_split",
    "build_pair_split",
]
