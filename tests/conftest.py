"""Shared pytest fixtures.

Tests must not depend on downloaded corpora or on a GPU: the suite has to run in CI on a
CPU-only runner with no network. Anything requiring real data is built here from
synthetic images.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def synthetic_face_dir(tmp_path_factory) -> Path:
    """A tiny on-disk corpus: 4 identities x 3 near-identical but distinct images.

    Each identity has a distinctive colour/texture signature so that a correct
    descriptor space must place same-identity images closer together than different
    ones. That property is what lets the metric tests assert *behaviour*, not just
    that a number came out.
    """
    import cv2

    root = tmp_path_factory.mktemp("faces")
    rng = np.random.default_rng(20240501)
    palette = {
        "cat_a": (40, 40, 220),
        "cat_b": (40, 200, 40),
        "cat_c": (210, 60, 30),
        "cat_d": (200, 200, 60),
    }
    for identity, base_colour in palette.items():
        identity_dir = root / identity
        identity_dir.mkdir()
        for index in range(3):
            image = np.full((128, 128, 3), base_colour, dtype=np.uint8)
            noise = rng.normal(0, 6 + index * 3, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.circle(image, (64, 64), 30 + index * 4, (255, 255, 255), 3)
            cv2.rectangle(image, (20, 20), (50 + index * 5, 50), (10, 10, 10), -1)
            cv2.imwrite(str(identity_dir / f"{identity}_{index:02d}.jpg"), image)
    return root


@pytest.fixture
def identity_records(synthetic_face_dir):
    """``FaceRecord`` objects matching the synthetic corpus on disk."""
    from catface.data.manifest import FaceRecord, sha1_file

    records = []
    for path in sorted(synthetic_face_dir.rglob("*.jpg")):
        identity = path.parent.name
        records.append(
            FaceRecord(
                image_id=f"synthetic:{identity}:{path.stem}",
                path=str(path),
                source="synthetic",
                identity=identity,
                sha1=sha1_file(path),
                width=128,
                height=128,
            )
        )
    return records


@pytest.fixture(scope="session")
def deterministic_embeddings():
    """Two cluster-separated descriptor sets with known ground truth.

    Cluster 0 is tight, cluster 1 is loose, and cluster 2 sits between them. A correct
    metric implementation must rank cluster-0 queries far better than cluster-2 ones —
    which is exactly what the ordering tests assert.
    """
    rng = np.random.default_rng(7)
    dim = 32
    identities = ["a", "b", "c"]
    spreads = {"a": 0.02, "b": 0.25, "c": 0.10}
    per_identity = 4

    gallery_vectors, gallery_labels = [], []
    query_vectors, query_labels = [], []
    for index, identity in enumerate(identities):
        centre = np.zeros(dim, dtype=np.float32)
        centre[index] = 1.0
        centre /= np.linalg.norm(centre)
        members = centre + rng.normal(0, spreads[identity], (per_identity, dim)).astype(np.float32)
        members /= np.linalg.norm(members, axis=1, keepdims=True)
        query_vectors.append(members[0])
        query_labels.append(identity)
        gallery_vectors.append(members[1:])
        gallery_labels.extend([identity] * (per_identity - 1))

    return {
        "query": np.vstack(query_vectors),
        "query_labels": np.array(query_labels),
        "gallery": np.vstack(gallery_vectors),
        "gallery_labels": np.array(gallery_labels),
    }


@pytest.fixture
def tiny_manifest_file(tmp_path, identity_records):
    """A manifest persisted to disk, for round-trip tests."""
    from catface.data.manifest import Manifest

    path = tmp_path / "manifest.jsonl"
    Manifest(identity_records).save(path)
    return path
