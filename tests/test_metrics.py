"""Metric correctness tests.

These are the most important tests in the repository: if ``evaluate_retrieval`` or
``evaluate_verification`` is wrong, every conclusion in the benchmark report is wrong in
a way no amount of model work can fix. Each metric is therefore checked against a
hand-computed value on a tiny example, not merely against its own output shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from catface.errors import BenchmarkError
from catface.eval.metrics import (
    bootstrap_recall_ci,
    cosine_similarity_matrix,
    evaluate_retrieval,
    evaluate_verification,
    paired_bootstrap_delta,
    pairwise_cosine,
)


class TestRetrievalHandComputed:
    """Hand-computed expectations. Gallery layout is [a, a, b, b] in every case.

    The similarity matrices are written so that the ranking is readable at a glance:
    for the "perfect" case each query puts its own identity in the top two ranks.
    """

    def test_perfect_ranking(self):
        similarity = np.array(
            [[0.9, 0.8, 0.1, 0.0],   # ranks: a, a, b, b
             [0.1, 0.0, 0.9, 0.8]],  # ranks: b, b, a, a
            dtype=np.float32,
        )
        labels_q = np.array(["a", "b"])
        labels_g = np.array(["a", "a", "b", "b"])
        metrics = evaluate_retrieval(similarity, labels_q, labels_g, recall_ks=(1, 2, 4))

        assert metrics.hit_at[1] == pytest.approx(1.0)
        assert metrics.hit_at[2] == pytest.approx(1.0)
        # Full recall at k=1 is only half: one of the two references was retrieved.
        assert metrics.recall_at[1] == pytest.approx(0.5)
        assert metrics.recall_at[2] == pytest.approx(1.0)
        assert metrics.mean_reciprocal_rank == pytest.approx(1.0)
        assert metrics.mean_rank_of_first_match == pytest.approx(1.0)
        assert metrics.median_rank_of_first_match == pytest.approx(1.0)
        # AP@1 = 1/1 for both queries; AP@2 = (1 + 2/2)/2 = 1.0.
        assert metrics.map_at[1] == pytest.approx(1.0)
        assert metrics.map_at[2] == pytest.approx(1.0)
        # Both relevant items sit at the top, so INP is 1.
        assert metrics.mean_inverse_negative_penalty == pytest.approx(1.0)

    def test_half_the_queries_find_their_match_first(self):
        similarity = np.array(
            [[0.9, 0.8, 0.1, 0.0],     # query a: a is first -> hit
             [0.9, 0.8, 0.7, 0.6]],    # query b: b is fourth -> miss at k=1
            dtype=np.float32,
        )
        labels_q = np.array(["a", "b"])
        labels_g = np.array(["a", "a", "b", "b"])
        metrics = evaluate_retrieval(similarity, labels_q, labels_g, recall_ks=(1, 4))
        assert metrics.hit_at[1] == pytest.approx(0.5)
        assert metrics.hit_at[4] == pytest.approx(1.0)
        assert metrics.recall_at[1] == pytest.approx(0.25)
        assert metrics.recall_at[4] == pytest.approx(1.0)
        # Mean reciprocal rank: query a -> 1, query b -> 1/3.
        assert metrics.mean_reciprocal_rank == pytest.approx((1.0 + 1 / 3) / 2)

    def test_hit_rate_is_never_below_full_recall(self):
        """Hitting the target implies recovering at least one relevant item."""
        similarity = np.array([[0.9, 0.8, 0.1, 0.0]], dtype=np.float32)
        labels_q = np.array(["a"])
        labels_g = np.array(["a", "a", "b", "b"])
        metrics = evaluate_retrieval(similarity, labels_q, labels_g, recall_ks=(1, 2, 4))
        for k in (1, 2, 4):
            assert metrics.hit_at[k] >= metrics.recall_at[k] - 1e-9

    def test_unordered_match_gives_expected_ranks(self):
        # Gallery [c, a, b]; query a finds its match at rank 2, query b at rank 2.
        similarity = np.array(
            [[0.9, 0.8, 0.1],
             [0.9, 0.1, 0.8]],
            dtype=np.float32,
        )
        labels_q = np.array(["a", "b"])
        labels_g = np.array(["c", "a", "b"])
        metrics = evaluate_retrieval(similarity, labels_q, labels_g, recall_ks=(1, 2, 3))
        assert metrics.hit_at[1] == pytest.approx(0.0)
        assert metrics.hit_at[2] == pytest.approx(1.0)
        assert metrics.hit_at[3] == pytest.approx(1.0)
        assert metrics.mean_rank_of_first_match == pytest.approx(2.0)

    def test_map_at_k_matches_hand_computation(self):
        # Single query, gallery [x, a, a, y]: relevant items land at ranks 2 and 3.
        similarity = np.array([[0.9, 0.8, 0.7, 0.1]], dtype=np.float32)
        labels_q = np.array(["a"])
        labels_g = np.array(["x", "a", "a", "y"])
        metrics = evaluate_retrieval(similarity, labels_q, labels_g, recall_ks=(2, 4))
        # AP@2 = precision at the single relevant rank 2 divided by min(hits, k) = 0.5/2.
        assert metrics.map_at[2] == pytest.approx(0.25)
        # AP@4 = (1/2 + 2/3) / 2 = 0.58333..., which matches sklearn's
        # average_precision_score on this ranking.
        assert metrics.map_at[4] == pytest.approx((0.5 + 2 / 3) / 2)

    def test_mrr_uses_rank_of_first_match(self):
        similarity = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
        labels_q = np.array(["a"])
        labels_g = np.array(["x", "y", "a"])
        metrics = evaluate_retrieval(similarity, labels_q, labels_g, recall_ks=(1,))
        assert metrics.mean_reciprocal_rank == pytest.approx(1 / 3)
        assert metrics.mean_rank_of_first_match == pytest.approx(3.0)

    def test_micro_average_across_queries(self):
        # Gallery [a, b, c]: query a finds a at rank 1, query b finds b at rank 2.
        similarity = np.array(
            [[0.9, 0.1, 0.0], [0.9, 0.8, 0.1]], dtype=np.float32
        )
        labels_q = np.array(["a", "b"])
        labels_g = np.array(["a", "b", "c"])
        metrics = evaluate_retrieval(similarity, labels_q, labels_g, recall_ks=(1, 2))
        assert metrics.hit_at[1] == pytest.approx(0.5)
        assert metrics.hit_at[2] == pytest.approx(1.0)
        assert metrics.num_queries == 2

    def test_minp_penalises_a_late_last_hit(self):
        """mINP uses the *last* relevant rank, so it separates models mAP saturates on."""
        # Gallery [a, a, b, b].
        tight = np.array([[0.9, 0.8, 0.1, 0.05]], dtype=np.float32)   # both a's at ranks 1-2
        spread = np.array([[0.9, 0.1, 0.8, 0.05]], dtype=np.float32)  # a's at ranks 1 and 3
        labels_q = np.array(["a"])
        labels_g = np.array(["a", "a", "b", "b"])
        tight_metrics = evaluate_retrieval(tight, labels_q, labels_g, recall_ks=(4,))
        spread_metrics = evaluate_retrieval(spread, labels_q, labels_g, recall_ks=(4,))
        # tight: last relevant rank 2, 2 relevant -> 2/2 = 1.0
        assert tight_metrics.mean_inverse_negative_penalty == pytest.approx(1.0)
        # spread: last relevant rank 3, 2 relevant -> 2/3
        assert spread_metrics.mean_inverse_negative_penalty == pytest.approx(2 / 3)
        assert tight_metrics.mean_inverse_negative_penalty > spread_metrics.mean_inverse_negative_penalty
        # mAP@4 is 1.0 for both, which is exactly why mINP is also reported.
        assert tight_metrics.map_at[4] == pytest.approx(1.0)
        assert spread_metrics.map_at[4] == pytest.approx(5 / 6)


class TestSelfExclusion:
    def test_self_match_is_not_counted_as_a_hit(self):
        """Without exclusion, a query that is in the gallery scores a trivial 1.0."""
        similarity = np.array([[1.0, 0.2, 0.1]], dtype=np.float32)
        labels_q = np.array(["a"])
        labels_g = np.array(["a", "b", "c"])
        exclude = np.array([[True, False, False]])

        with_self = evaluate_retrieval(similarity, labels_q, labels_g, recall_ks=(1,))
        assert with_self.hit_at[1] == pytest.approx(1.0)
        # After excluding the exact same image, no other positive remains, so this
        # query is unusable — the function must say so rather than reporting 0.0.
        with pytest.raises(BenchmarkError):
            evaluate_retrieval(
                similarity, labels_q, labels_g, recall_ks=(1,), exclude_self=exclude
            )

    def test_exclusion_keeps_other_positives(self):
        similarity = np.array([[1.0, 0.9, 0.1]], dtype=np.float32)
        labels_q = np.array(["a"])
        labels_g = np.array(["a", "a", "b"])
        exclude = np.array([[True, False, False]])
        metrics = evaluate_retrieval(
            similarity, labels_q, labels_g, recall_ks=(1,), exclude_self=exclude
        )
        assert metrics.hit_at[1] == pytest.approx(1.0)
        # The remaining positive now sits at rank 1 because the self-match is masked out.
        assert metrics.mean_rank_of_first_match == pytest.approx(1.0)
        assert metrics.num_queries == 1


class TestRetrievalErrors:
    def test_shape_mismatch_between_similarity_and_labels(self):
        with pytest.raises(BenchmarkError, match="query labels"):
            evaluate_retrieval(np.zeros((2, 3)), np.array(["a"]), np.array(["a", "b", "c"]))
        with pytest.raises(BenchmarkError, match="gallery labels"):
            evaluate_retrieval(np.zeros((2, 3)), np.array(["a", "b"]), np.array(["a", "b"]))

    def test_one_dimensional_similarity_is_rejected(self):
        with pytest.raises(BenchmarkError, match="2-D"):
            evaluate_retrieval(np.zeros(3), np.array(["a"]), np.array(["a", "b", "c"]))

    def test_query_without_any_relevant_item_is_rejected(self):
        similarity = np.array([[0.9, 0.1]], dtype=np.float32)
        with pytest.raises(BenchmarkError, match="No query has a relevant"):
            evaluate_retrieval(similarity, np.array(["z"]), np.array(["a", "b"]))


class TestBootstrap:
    def test_ci_is_ordered_and_contains_the_point_estimate(self):
        rng = np.random.default_rng(3)
        embeddings = rng.normal(size=(40, 16)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.array([f"id{i // 4}" for i in range(40)])
        similarity = embeddings @ embeddings.T

        lower, upper = bootstrap_recall_ci(
            similarity, labels, labels, k=1, samples=200, seed=1
        )
        assert lower <= upper
        metrics = evaluate_retrieval(similarity, labels, labels, recall_ks=(1,))
        assert lower - 1e-6 <= metrics.hit_at[1] <= upper + 1e-6

    def test_identical_models_have_zero_delta(self):
        rng = np.random.default_rng(5)
        embeddings = rng.normal(size=(24, 8)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.array([f"id{i // 3}" for i in range(24)])
        similarity = embeddings @ embeddings.T

        stats = paired_bootstrap_delta(
            similarity, similarity, labels, labels, k=1, samples=200, seed=2
        )
        assert stats["observed_delta"] == pytest.approx(0.0)
        assert stats["ci95"][0] <= 0.0 <= stats["ci95"][1]


class TestVerification:
    def test_perfectly_separable_scores(self):
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        labels = np.array([1, 1, 1, 0, 0, 0])
        metrics = evaluate_verification(scores, labels, far_targets=(0.0, 0.5))
        assert metrics.roc_auc == pytest.approx(1.0)
        assert metrics.eer == pytest.approx(0.0, abs=0.06)
        assert metrics.tar_at_far[0.0] == pytest.approx(1.0)

    def test_auc_equals_half_for_identical_distributions(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 1, 0, 0])
        metrics = evaluate_verification(scores, labels)
        assert metrics.roc_auc == pytest.approx(0.5)

    def test_auc_matches_sklearn_reference(self):
        """Cross-checked against sklearn.metrics.roc_auc_score for the same input."""
        scores = np.array([0.9, 0.4, 0.3, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0, 0])
        metrics = evaluate_verification(scores, labels)
        # 0.9 clears all 3 negatives, 0.4 and 0.3 clear 2 each -> 7/9.
        assert metrics.roc_auc == pytest.approx(7 / 9)
        sklearn = pytest.importorskip("sklearn.metrics")
        assert metrics.roc_auc == pytest.approx(
            sklearn.roc_auc_score(labels, scores), abs=1e-9
        )

    def test_auc_matches_sklearn_on_ties(self):
        """Ties must be scored by average rank, not by array order."""
        sklearn = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(17)
        scores = rng.integers(0, 3, size=400).astype(np.float64)  # many ties
        labels = rng.integers(0, 2, size=400)
        metrics = evaluate_verification(scores, labels)
        assert metrics.roc_auc == pytest.approx(
            sklearn.roc_auc_score(labels, scores), abs=1e-9
        )

    def test_tar_at_far_is_monotonic_in_far(self):
        rng = np.random.default_rng(11)
        positives = rng.normal(0.7, 0.2, 500)
        negatives = rng.normal(0.3, 0.2, 500)
        scores = np.concatenate([positives, negatives])
        labels = np.concatenate([np.ones(500), np.zeros(500)])
        metrics = evaluate_verification(scores, labels, far_targets=(1e-3, 1e-2, 1e-1))
        assert metrics.tar_at_far[1e-3] <= metrics.tar_at_far[1e-2] <= metrics.tar_at_far[1e-1]

    def test_single_class_is_rejected(self):
        with pytest.raises(BenchmarkError, match="both same- and different"):
            evaluate_verification(np.array([0.5, 0.6]), np.array([1, 1]))

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(BenchmarkError, match="scores for"):
            evaluate_verification(np.array([0.5, 0.6]), np.array([1, 0, 1]))


class TestSimilarityHelpers:
    def test_cosine_matrix_matches_manual_computation(self):
        a = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        similarity = cosine_similarity_matrix(a, b)
        assert similarity[0, 0] == pytest.approx(1.0)
        assert similarity[0, 1] == pytest.approx(1 / np.sqrt(2), abs=1e-6)
        assert similarity[1, 0] == pytest.approx(0.0, abs=1e-6)

    def test_pairwise_cosine_matches_rowwise(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert pairwise_cosine(a[None, :], b[None, :])[0] == pytest.approx(0.0)

    def test_dimension_mismatch_detected(self):
        with pytest.raises(BenchmarkError, match="dimension mismatch"):
            cosine_similarity_matrix(np.zeros((1, 4)), np.zeros((1, 5)))
        with pytest.raises(BenchmarkError, match="equal shapes"):
            pairwise_cosine(np.zeros((1, 4)), np.zeros((1, 5)))

    def test_zero_rows_do_not_produce_nan(self):
        similarity = cosine_similarity_matrix(np.zeros((1, 3)), np.zeros((1, 3)))
        assert np.isfinite(similarity).all()
