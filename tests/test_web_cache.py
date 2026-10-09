"""Gallery descriptor cache used at service startup.

Embedding 12 644 gallery images takes ~170 s on the reference machine, and that dominated every
start. Caching the descriptors makes it ~12 s. The optimisation is only safe if a *stale* entry can
never be used, so the tests here focus on rejection as much as on the hit: a wrong descriptor served
under a known label corrupts every answer while looking perfectly healthy, which is the worst failure
mode a retrieval service has.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from catface.data.manifest import Manifest
from catface.errors import DataError
from catface.web.service import SearchService

WIDTH = 4


class FakeEmbedder:
    """Returns a fixed descriptor per call, so the vectors written are predictable."""

    class _Config:
        backbone = "fake"
        tta: ClassVar[list[str]] = ["identity"]

    class _Head:
        embedding_dim = WIDTH

    def __init__(self) -> None:
        self.config = self._Config()
        self.head = self._Head()
        self.calls = 0


def _manifest(tmp_path: Path, count: int = 3) -> Path:
    path = tmp_path / "manifest.jsonl"
    lines = []
    for index in range(count):
        (tmp_path / f"{index}.jpg").write_bytes(b"fake")
        lines.append(
            json.dumps(
                {
                    "image_id": f"img-{index}",
                    "identity": f"cat-{index}",
                    "path": str(tmp_path / f"{index}.jpg"),
                    "source": "test",
                    "detector": "whole_image",
                }
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _service(tmp_path: Path, manifest: Path, cache_dir: Path) -> SearchService:
    checkpoint = tmp_path / "best.pt"
    if not checkpoint.is_file():
        checkpoint.write_bytes(b"not a real checkpoint")
    return SearchService(
        checkpoint=checkpoint,
        manifest=manifest,
        device="cpu",
        descriptor_cache_dir=cache_dir,
    )


def _install_fake_embedder(service: SearchService, vector: list[float], monkeypatch) -> None:
    """Replace both the checkpoint load and the encoder, so no torch work happens."""
    import catface.web.service as module

    embedder = FakeEmbedder()
    monkeypatch.setattr(module.Embedder, "load", staticmethod(lambda *a, **k: embedder))
    array = np.asarray([[vector] * 1], dtype=np.float32)
    array = np.tile(np.asarray(vector, dtype=np.float32), (1, 1))

    class Result:
        vectors = array

    def fake_embed_records(_embedder, paths, **kwargs):
        Result.vectors = np.tile(np.asarray(vector, dtype=np.float32), (len(paths), 1))
        embedder.calls += 1
        return Result

    monkeypatch.setattr(module, "embed_records", fake_embed_records)


class TestCacheWriteAndRead:
    def test_first_load_writes_a_cache_and_second_load_reads_it(self, tmp_path, monkeypatch):
        manifest = _manifest(tmp_path)
        cache_dir = tmp_path / "cache"
        service = _service(tmp_path, manifest, cache_dir)
        _install_fake_embedder(service, [1.0, 2.0, 3.0, 4.0], monkeypatch)

        service.load()
        assert service.ready
        written = list(cache_dir.glob("*.npz"))
        assert len(written) == 1, "the first load must leave a cache behind"
        first = service._vectors.copy()

        # A second service over the same inputs must come from disk and produce identical vectors.
        import catface.web.service as module

        monkeypatch.setattr(
            module,
            "embed_records",
            lambda *a, **k: pytest.fail("the second load must not re-embed"),
        )
        second_service = _service(tmp_path, manifest, cache_dir)
        monkeypatch.setattr(module.Embedder, "load", staticmethod(lambda *a, **k: FakeEmbedder()))
        second_service.load()
        assert second_service.ready
        assert np.array_equal(first, second_service._vectors), (
            "the cached descriptors must equal the ones that were written"
        )

    def test_cache_is_not_written_when_the_model_cannot_load(self, tmp_path, monkeypatch):
        """A failed load must not leave a partial cache that a later run would trust."""
        manifest = _manifest(tmp_path)
        cache_dir = tmp_path / "cache"
        service = _service(tmp_path, manifest, cache_dir)
        import catface.web.service as module

        def boom(*args, **kwargs):
            raise DataError("checkpoint is not loadable")

        monkeypatch.setattr(module.Embedder, "load", staticmethod(boom))
        with pytest.raises(DataError):
            service.load()
        assert not list(cache_dir.glob("*.npz")), "no cache may be written from a failed load"


class TestStaleCacheIsRejected:
    """The controls that matter: a cache that does not match must never be used."""

    def _prime(self, tmp_path, monkeypatch, vector: list[float]):
        manifest = _manifest(tmp_path)
        cache_dir = tmp_path / "cache"
        service = _service(tmp_path, manifest, cache_dir)
        _install_fake_embedder(service, vector, monkeypatch)
        service.load()
        entry = next(cache_dir.glob("*.npz"))
        return manifest, cache_dir, service, entry

    def test_a_different_checkpoint_gets_a_different_cache_key(self, tmp_path, monkeypatch):
        """Retraining usually writes the same path, so the key must hash the content."""
        manifest, cache_dir, _service_unused, entry = self._prime(tmp_path, monkeypatch, [1.0, 0.0, 0.0, 0.0])
        assert entry.is_file()

        # Same path, different bytes: this is what a retrained checkpoint looks like.
        (tmp_path / "best.pt").write_bytes(b"a different set of weights entirely")
        changed = _service(tmp_path, manifest, cache_dir)
        import catface.web.service as module

        monkeypatch.setattr(module.Embedder, "load", staticmethod(lambda *a, **k: FakeEmbedder()))
        assert changed._cache_path() != entry, (
            "a checkpoint with different content must not reuse the cached descriptors"
        )

    def test_a_manifest_mismatch_is_rejected_even_if_the_file_exists(self, tmp_path, monkeypatch):
        """Defence in depth: a corrupted or hand-placed cache must be detected on its contents."""
        manifest, cache_dir, _service_unused, entry = self._prime(tmp_path, monkeypatch, [1.0, 0.0, 0.0, 0.0])

        # Rewrite the cache with the right shape but the wrong labels, keeping the file name.
        with np.load(entry, allow_pickle=False) as archive:
            vectors = archive["vectors"]
            ids = archive["ids"].tolist()
        np.savez_compressed(
            entry,
            vectors=vectors,
            ids=np.asarray(ids, dtype="U256"),
            labels=np.asarray(["wrong-label"] * len(ids), dtype="U128"),
        )

        rebuilt = _service(tmp_path, manifest, cache_dir)
        import catface.web.service as module

        monkeypatch.setattr(module.Embedder, "load", staticmethod(lambda *a, **k: FakeEmbedder()))
        monkeypatch.setattr(
            module,
            "embed_records",
            lambda _e, paths, **k: type(
                "R",
                (),
                {"vectors": np.tile(np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(paths), 1))},
            )(),
        )
        rebuilt._embedder = FakeEmbedder()
        rebuilt._records = list(Manifest.load(manifest))
        assert rebuilt._load_cached_descriptors() is None, (
            "a cache whose labels disagree with the manifest must be refused, not used"
        )

    def test_an_unreadable_cache_falls_back_to_embedding(self, tmp_path, monkeypatch):
        """A truncated file (an interrupted write) must degrade to the slow path, not crash."""
        manifest, cache_dir, _service_unused, entry = self._prime(tmp_path, monkeypatch, [1.0, 0.0, 0.0, 0.0])
        entry.write_bytes(b"")  # exactly what an interrupted save leaves behind

        rebuilt = _service(tmp_path, manifest, cache_dir)
        rebuilt._embedder = FakeEmbedder()
        rebuilt._records = list(Manifest.load(manifest))
        assert rebuilt._load_cached_descriptors() is None
