"""Index and retrieval-integration tests.

The index is the component a production caller touches directly, so it is tested for the
properties that matter operationally: ids stay aligned with vectors, scores are true
cosine similarities, the NumPy and FAISS backends agree, and an index survives a
round-trip through disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from catface.errors import ArtifactError
from catface.index.faiss_index import INDEX_FORMAT_VERSION, VectorIndex


def unit_vectors(count: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class TestConstruction:
    def test_add_requires_aligned_ids(self):
        index = VectorIndex(dim=8)
        with pytest.raises(ArtifactError, match="aligned"):
            index.add(unit_vectors(3, 8), ["a", "b"])

    def test_add_rejects_a_wrong_width(self):
        index = VectorIndex(dim=8)
        with pytest.raises(ArtifactError, match="does not match index dim"):
            index.add(unit_vectors(2, 4), ["a", "b"])

    def test_add_rejects_a_one_dimensional_input(self):
        index = VectorIndex(dim=8)
        with pytest.raises(ArtifactError, match="2-D"):
            index.add(np.zeros(8, dtype=np.float32), ["a"])

    def test_build_rejects_an_empty_index(self):
        with pytest.raises(ArtifactError, match="no vectors"):
            VectorIndex(dim=8, kind="flat_ip").build()

    def test_size_tracks_the_number_of_vectors(self):
        index = VectorIndex(dim=8).add(unit_vectors(5, 8), list("abcde"))
        assert len(index) == 5
        assert index.size == 5

    def test_unknown_metric_is_rejected(self):
        with pytest.raises(ArtifactError, match="metric"):
            VectorIndex(dim=8, metric="cosine-similarity")


class TestSearch:
    def test_exact_match_is_returned_first(self):
        vectors = unit_vectors(20, 32)
        ids = [f"v{i}" for i in range(20)]
        index = VectorIndex(dim=32).add(vectors, ids).build()

        result = index.search(vectors[7][None, :], top_k=3)
        assert result.ids[0][0] == "v7"
        assert result.scores[0][0] == pytest.approx(1.0, abs=1e-5)

    def test_scores_descend(self):
        index = VectorIndex(dim=32).add(unit_vectors(30, 32), [f"v{i}" for i in range(30)]).build()
        result = index.search(unit_vectors(1, 32, seed=9), top_k=10)
        assert np.all(np.diff(result.scores[0]) <= 1e-6)

    def test_scores_match_manual_cosine(self):
        vectors = unit_vectors(10, 16)
        index = VectorIndex(dim=16).add(vectors, [f"v{i}" for i in range(10)]).build()
        query = unit_vectors(1, 16, seed=4)
        result = index.search(query, top_k=4)
        expected = (query @ vectors.T)[0]
        for identifier, score in zip(result.ids[0], result.scores[0]):
            position = int(identifier[1:])
            assert score == pytest.approx(expected[position], abs=1e-5)

    def test_numpy_backend_matches_the_flat_backend(self):
        vectors = unit_vectors(40, 24)
        ids = [f"v{i}" for i in range(40)]
        query = unit_vectors(3, 24, seed=3)

        faiss_like = VectorIndex(dim=24, kind="flat_ip").add(vectors, ids).build()
        numpy_like = VectorIndex(dim=24, kind="numpy").add(vectors, ids).build()
        assert np.allclose(faiss_like.score_all(query), numpy_like.score_all(query), atol=1e-5)

        first = faiss_like.search(query, top_k=5)
        second = numpy_like.search(query, top_k=5)
        assert first.ids == second.ids
        assert np.allclose(first.scores, second.scores, atol=1e-5)

    def test_top_k_larger_than_the_index_is_clamped(self):
        index = VectorIndex(dim=8).add(unit_vectors(3, 8), list("abc")).build()
        result = index.search(unit_vectors(1, 8), top_k=50)
        assert len(result.ids[0]) == 3

    def test_search_on_an_empty_index_is_reported(self):
        with pytest.raises(ArtifactError, match="empty"):
            VectorIndex(dim=8).search(unit_vectors(1, 8))

    def test_query_width_mismatch_is_reported(self):
        index = VectorIndex(dim=8).add(unit_vectors(3, 8), list("abc")).build()
        with pytest.raises(ArtifactError, match="query width"):
            index.search(unit_vectors(1, 4))

    def test_scores_are_order_independent_of_add_order(self):
        vectors = unit_vectors(25, 16)
        ids = [f"v{i}" for i in range(25)]
        query = unit_vectors(1, 16, seed=11)

        forward = VectorIndex(dim=16).add(vectors, ids).build().search(query, top_k=5)
        order = list(range(25))[::-1]
        backward = VectorIndex(dim=16).add(vectors[order], [ids[i] for i in order]).build()
        reversed_result = backward.search(query, top_k=5)
        assert forward.ids == reversed_result.ids


class TestPersistence:
    def test_round_trip_preserves_ids_scores_and_metadata(self, tmp_path):
        vectors = unit_vectors(12, 32)
        ids = [f"cat{i:02d}" for i in range(12)]
        index = VectorIndex(dim=32, kind="flat_ip", metric="cosine").add(vectors, ids).build()
        query = unit_vectors(2, 32, seed=5)
        before = index.search(query, top_k=4)

        index.save(tmp_path, embeddings=vectors)
        restored = VectorIndex.load(tmp_path)
        after = restored.search(query, top_k=4)

        assert restored.size == 12
        assert restored.ids == ids
        assert before.ids == after.ids
        assert np.allclose(before.scores, after.scores, atol=1e-5)
        assert restored.metric == "cosine"

    def test_existing_metadata_is_required(self, tmp_path):
        with pytest.raises(ArtifactError, match=r"index_meta\.json"):
            VectorIndex.load(tmp_path / "nothing")

    def test_future_index_version_is_rejected(self, tmp_path):
        import json

        index = VectorIndex(dim=8).add(unit_vectors(2, 8), ["a", "b"]).build()
        index.save(tmp_path)
        meta_path = tmp_path / "index_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["version"] = INDEX_FORMAT_VERSION + 1
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ArtifactError, match="version"):
            VectorIndex.load(tmp_path)

    def test_raw_embeddings_are_written_when_asked(self, tmp_path):
        vectors = unit_vectors(4, 8)
        VectorIndex(dim=8).add(vectors, list("abcd")).build().save(tmp_path, embeddings=vectors)
        assert (tmp_path / "raw_embeddings.npy").is_file()


class TestRetrievalIntegration:
    """End-to-end: recover a known identity structure from descriptors through the index."""

    def test_index_recovers_the_correct_identity(self):
        from catface.eval.metrics import evaluate_retrieval

        rng = np.random.default_rng(21)
        dim = 64
        identities = 10
        per_identity = 5
        centres = unit_vectors(identities, dim, seed=99)

        gallery_vectors, gallery_labels = [], []
        query_vectors, query_labels = [], []
        for index, centre in enumerate(centres):
            members = centre + rng.normal(0, 0.05, (per_identity, dim)).astype(np.float32)
            members /= np.linalg.norm(members, axis=1, keepdims=True)
            query_vectors.append(members[0])
            query_labels.append(f"cat{index}")
            gallery_vectors.append(members[1:])
            gallery_labels.extend([f"cat{index}"] * (per_identity - 1))

        query = np.vstack(query_vectors)
        gallery = np.vstack(gallery_vectors)
        ids = [f"g{i}" for i in range(len(gallery))]

        vector_index = VectorIndex(dim=dim).add(gallery, ids).build()
        similarity = vector_index.score_all(query)

        metrics = evaluate_retrieval(
            similarity, np.array(query_labels), np.array(gallery_labels), recall_ks=(1, 5)
        )
        # Well-separated clusters must be perfectly retrievable; anything less means the
        # index is corrupting the scores.
        assert metrics.hit_at[1] == pytest.approx(1.0)
        assert metrics.hit_at[5] == pytest.approx(1.0)
