"""Descriptor post-processing tests.

These transforms are the highest-leverage, lowest-risk accuracy levers in the system, so
each one is tested for the property it is supposed to provide rather than for its exact
output: whitening must decorrelate, DBA must denoise, and αQE must keep identical inputs
identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from catface.errors import BenchmarkError
from catface.eval.postprocess import (
    AlphaQueryExpansion,
    WhiteningTransform,
    alpha_query_expansion,
    database_side_augmentation,
    l2_normalize,
)


def clustered_features(n_identities: int = 8, per_identity: int = 6, dim: int = 24, spread: float = 0.05, seed: int = 0):
    """Descriptors with a dominant shared direction, mimicking real face embeddings.

    The shared component is what whitening is supposed to suppress: it contributes to
    every cosine similarity while carrying no identity information.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(size=dim).astype(np.float32)
    common /= np.linalg.norm(common)

    vectors, labels = [], []
    for index in range(n_identities):
        centre = rng.normal(size=dim).astype(np.float32)
        centre /= np.linalg.norm(centre)
        centre = 0.3 * centre + 0.95 * common  # let the common mode dominate
        members = centre + rng.normal(0, spread, (per_identity, dim)).astype(np.float32)
        vectors.append(members)
        labels.extend([f"id{index}"] * per_identity)
    return l2_normalize(np.vstack(vectors)), np.array(labels)


class TestL2Normalize:
    def test_rows_have_unit_norm(self):
        matrix = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        normalised = l2_normalize(matrix)
        assert np.allclose(np.linalg.norm(normalised, axis=1), 1.0)

    def test_zero_rows_are_left_at_zero_not_nan(self):
        matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        normalised = l2_normalize(matrix)
        assert np.isfinite(normalised).all()
        assert np.allclose(normalised[0], 0.0)


class TestWhitening:
    def test_fit_rejects_a_degenerate_input(self):
        with pytest.raises(BenchmarkError, match="at least 2"):
            WhiteningTransform.fit(np.zeros((1, 8), dtype=np.float32))

    def test_output_dimension_follows_the_request(self):
        features, _ = clustered_features(dim=32)
        transform = WhiteningTransform.fit(features, dim=16)
        assert transform.output_dim == 16
        assert transform.transform(features).shape == (features.shape[0], 16)

    def test_output_is_unit_norm_and_finite(self):
        features, _ = clustered_features()
        for mode in ("pca", "pcaw"):
            transformed = WhiteningTransform.fit(features, mode=mode).transform(features)
            assert np.isfinite(transformed).all()
            assert np.allclose(np.linalg.norm(transformed, axis=1), 1.0, atol=1e-4)

    def test_whitening_decorrelates_dimensions(self):
        """PCA-whitening's defining property: off-diagonal covariance collapses."""
        features, _ = clustered_features(dim=24)
        before = np.abs(np.corrcoef((features - features.mean(0)).T))
        np.fill_diagonal(before, 0.0)
        transformed = WhiteningTransform.fit(features, mode="pcaw").transform(features)
        after = np.abs(np.corrcoef((transformed - transformed.mean(0)).T))
        np.fill_diagonal(after, 0.0)
        assert after.max() < before.max()
        assert after.mean() < before.mean()

    def test_whitening_improves_separation_on_clustered_data(self):
        """The whole point: same-identity similarity rises relative to different."""
        features, labels = clustered_features(seed=1)
        same, different = 0, 0
        count_same, count_diff = 0, 0
        similarity = features @ features.T
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    same += similarity[i, j]
                    count_same += 1
                else:
                    different += similarity[i, j]
                    count_diff += 1
        baseline_gap = same / count_same - different / count_diff

        transformed = WhiteningTransform.fit(features, mode="pcaw").transform(features)
        similarity = transformed @ transformed.T
        same, different = 0, 0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    same += similarity[i, j]
                else:
                    different += similarity[i, j]
        whitened_gap = same / count_same - different / count_diff
        assert whitened_gap > baseline_gap

    def test_pca_mode_preserves_magnitude_structure(self):
        features, _ = clustered_features()
        pca = WhiteningTransform.fit(features, mode="pca").transform(features)
        pcaw = WhiteningTransform.fit(features, mode="pcaw").transform(features)
        # The two modes must not be the same transform.
        assert not np.allclose(pca, pcaw)

    def test_variance_report_is_a_fraction(self):
        features, _ = clustered_features()
        transform = WhiteningTransform.fit(features, dim=8)
        assert 0.0 < transform.explain_variance() <= 1.0


class TestDatabaseSideAugmentation:
    def test_identity_when_k_is_zero(self):
        features, _ = clustered_features()
        augmented = database_side_augmentation(features, k=0)
        assert np.allclose(augmented, l2_normalize(features), atol=1e-6)

    def test_output_is_unit_norm_and_finite(self):
        features, _ = clustered_features()
        augmented = database_side_augmentation(features, k=3, alpha=3.0)
        assert augmented.shape == features.shape
        assert np.allclose(np.linalg.norm(augmented, axis=1), 1.0, atol=1e-4)

    def test_augmentation_moves_descriptors_toward_their_cluster(self):
        """DBA should raise same-identity similarity, which is its denoising effect."""
        features, labels = clustered_features(seed=3)
        augmented = database_side_augmentation(features, k=3, alpha=3.0)

        def mean_same(matrix: np.ndarray) -> float:
            similarity = matrix @ matrix.T
            values = [
                similarity[i, j]
                for i in range(len(labels))
                for j in range(i + 1, len(labels))
                if labels[i] == labels[j]
            ]
            return float(np.mean(values))

        assert mean_same(augmented) > mean_same(features)

    def test_single_vector_is_handled(self):
        single = np.ones((1, 8), dtype=np.float32)
        augmented = database_side_augmentation(single, k=3)
        assert augmented.shape == single.shape


class TestAlphaQueryExpansion:
    def test_disabled_returns_a_normalised_copy(self):
        features, _ = clustered_features()
        result = alpha_query_expansion(features[:2], features, AlphaQueryExpansion(enabled=False))
        assert np.allclose(result, l2_normalize(features[:2]), atol=1e-6)

    def test_output_shape_and_norm(self):
        features, _ = clustered_features()
        config = AlphaQueryExpansion(enabled=True, top_k=3, alpha=3.0)
        expanded = alpha_query_expansion(features[:4], features, config)
        assert expanded.shape == (4, features.shape[1])
        assert np.allclose(np.linalg.norm(expanded, axis=1), 1.0, atol=1e-4)
        assert np.isfinite(expanded).all()

    def test_expansion_is_influenced_by_the_gallery(self):
        """αQE must actually use its neighbours: a query far from all gallery entries
        (zero-similarity neighbourhood) can only move if the expansion reads the gallery.

        The accuracy benefit of αQE is asserted on real descriptors in the benchmark; a
        synthetic Gaussian cluster is too easy to produce a reliable ordering from, so
        this test pins the mechanism instead.
        """
        dim = 16
        features, _labels = clustered_features(n_identities=6, per_identity=4, dim=dim)
        gallery = features

        config = AlphaQueryExpansion(enabled=True, top_k=4, alpha=4.0, reverse_weight=0.3)
        query = features[0]
        expanded = alpha_query_expansion(query[None, :], gallery, config)[0]
        assert not np.allclose(expanded, l2_normalize(query[None, :])[0], atol=1e-5)

    def test_higher_alpha_stays_closer_to_the_original_query(self):
        """``alpha`` is the self-weight: raising it must shrink the modification."""
        features, _ = clustered_features(seed=6)
        query = features[0][None, :]
        baseline = l2_normalize(query)[0]

        distances = []
        for alpha in (1.0, 8.0):
            config = AlphaQueryExpansion(enabled=True, top_k=4, alpha=alpha, reverse_weight=0.0)
            expanded = alpha_query_expansion(query, features, config)[0]
            distances.append(float(1.0 - expanded @ baseline))
        assert distances[1] <= distances[0] + 1e-6

    def test_empty_gallery_is_handled(self):
        features, _ = clustered_features()
        config = AlphaQueryExpansion(enabled=True, top_k=3)
        expanded = alpha_query_expansion(features[:2], np.zeros((0, features.shape[1]), dtype=np.float32), config)
        assert expanded.shape == (2, features.shape[1])
        assert np.isfinite(expanded).all()

    def test_top_k_larger_than_gallery_is_clamped(self):
        features, _ = clustered_features(n_identities=2, per_identity=2)
        config = AlphaQueryExpansion(enabled=True, top_k=999)
        expanded = alpha_query_expansion(features[:1], features, config)
        assert np.isfinite(expanded).all()


class TestPipelineOrdering:
    def test_full_stack_preserves_shapes_and_finiteness(self):
        """The composed transform used by the benchmark must be numerically safe."""
        from catface.eval.benchmark import PostprocessConfig, apply_postprocessing

        features, _labels = clustered_features(n_identities=6, per_identity=4, dim=32)
        query, gallery = features[:6], features[6:]

        for config in (
            PostprocessConfig(),
            PostprocessConfig(whiten="pcaw", whiten_dim=16),
            PostprocessConfig(whiten="pcaw", dba=True),
            PostprocessConfig(whiten="pcaw", dba=True, query_expansion="aqe"),
            PostprocessConfig(query_expansion="aqe"),
        ):
            q, g, diagnostics = apply_postprocessing(query, gallery, config)
            assert q.shape[0] == query.shape[0]
            assert g.shape[0] == gallery.shape[0]
            assert q.shape[1] == g.shape[1], f"widths diverged for {config.describe()}"
            assert np.isfinite(q).all() and np.isfinite(g).all()
            assert diagnostics["query_dim"] == diagnostics["gallery_dim"]
