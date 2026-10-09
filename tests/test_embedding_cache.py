"""The general array cache: arbitrary array sets, not just a query/gallery protocol.

``tools/embedding_cache.save`` encodes one shape — query plus gallery. A single-split job (the 8 833
image training set) does not fit it, and forcing it through failed with ``KeyError: 'query'`` after
95 seconds of embedding. ``save_arrays``/``load_arrays`` exist for that case, so they are tested for
the two properties that matter: a round trip preserves the values exactly, and a truncated file is
reported as a miss rather than raising, because an interrupted write genuinely leaves one behind.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache = _load("embedding_cache")


class TestArrayCache:
    def test_round_trip_preserves_values(self, tmp_path):
        vectors = np.random.default_rng(0).random((7, 5), dtype=np.float32)
        cache.save_arrays(tmp_path, "key", {"vectors": vectors, "ids": ["a", "b", "c"]}, list_fields=("ids",))
        restored = cache.load_arrays(tmp_path, "key", list_fields=("ids",))
        assert restored is not None
        assert np.array_equal(vectors, restored["vectors"]), "the cache must not alter the values"
        assert restored["ids"] == ["a", "b", "c"]

    def test_a_missing_entry_is_a_miss(self, tmp_path):
        assert cache.load_arrays(tmp_path, "absent") is None

    def test_a_truncated_entry_is_a_miss_not_a_crash(self, tmp_path):
        """An interrupted write leaves a zero-byte file; that must degrade to "no cache"."""
        cache.save_arrays(tmp_path, "key", {"vectors": np.zeros((2, 2), dtype=np.float32)})
        (tmp_path / "key.npz").write_bytes(b"")
        assert cache.load_arrays(tmp_path, "key") is None

    def test_float64_input_is_stored_as_float32(self, tmp_path):
        """Descriptors are float32 throughout; storing float64 would double the file for nothing."""
        cache.save_arrays(tmp_path, "key", {"vectors": np.zeros((2, 2), dtype=np.float64)})
        restored = cache.load_arrays(tmp_path, "key")
        assert restored is not None
        assert restored["vectors"].dtype == np.float32

    def test_single_split_usage_does_not_require_a_gallery(self, tmp_path):
        """The shape that broke: one split, no query/gallery pair."""
        cache.save_arrays(
            tmp_path,
            "train",
            {
                "vectors": np.ones((4, 3), dtype=np.float32),
                "ids": [f"img-{i}" for i in range(4)],
                "labels": ["cat-a", "cat-a", "cat-b", "cat-b"],
            },
            list_fields=("ids", "labels"),
        )
        restored = cache.load_arrays(tmp_path, "train", list_fields=("ids", "labels"))
        assert restored is not None
        assert restored["vectors"].shape == (4, 3)
        assert restored["labels"] == ["cat-a", "cat-a", "cat-b", "cat-b"]

    def test_a_cache_without_list_fields_round_trips(self, tmp_path):
        cache.save_arrays(tmp_path, "plain", {"matrix": np.arange(6).reshape(2, 3).astype(np.float32)})
        restored = cache.load_arrays(tmp_path, "plain")
        assert restored is not None
        assert restored["matrix"].tolist() == [[0, 1, 2], [3, 4, 5]]


class TestCacheKeyStability:
    def test_the_same_inputs_produce_the_same_key(self, tmp_path):
        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"weights")
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}", encoding="utf-8")
        first = cache.cache_key(checkpoint, manifest, protocol="p", image_size=224, tta=("identity",))
        second = cache.cache_key(checkpoint, manifest, protocol="p", image_size=224, tta=("identity",))
        assert first == second

    def test_changed_checkpoint_content_changes_the_key(self, tmp_path):
        """Retraining usually writes the same path, so the key must follow the bytes."""
        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"first")
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}", encoding="utf-8")
        before = cache.cache_key(checkpoint, manifest, protocol="p", image_size=224)
        checkpoint.write_bytes(b"second")
        after = cache.cache_key(checkpoint, manifest, protocol="p", image_size=224)
        assert before != after, "a retrained checkpoint must not reuse cached descriptors"

    def test_resolution_is_part_of_the_key(self, tmp_path):
        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"weights")
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}", encoding="utf-8")
        assert cache.cache_key(checkpoint, manifest, protocol="p", image_size=224) != (
            cache.cache_key(checkpoint, manifest, protocol="p", image_size=280)
        )


@pytest.mark.parametrize("missing", ["checkpoint"])
def test_absent_checkpoint_still_yields_a_key(tmp_path, missing):
    """A zero-shot configuration has no checkpoint and still needs a stable key."""
    del missing
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}", encoding="utf-8")
    key = cache.cache_key(None, manifest, protocol="p", image_size=224)
    assert isinstance(key, str) and len(key) == 16
