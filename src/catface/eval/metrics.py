"""Retrieval and verification metrics.

Two metric families are needed because they answer different product questions:

* **Retrieval** (``recall@k``, ``mAP``, ``mINP``) — "given this photo, can I find the
  other photos of the same cat?" This is what the search engine actually does.
* **Verification** (``ROC-AUC``, ``EER``, ``TAR@FAR``) — "are these two photos the same
  cat?" This is what a threshold has to decide, and it is the metric that decides
  whether an end-user-facing confidence score is meaningful.

Reporting only accuracy-style numbers hides the failure mode that matters: at a
fixed 1 % false-accept budget, how many genuine matches are we missing?
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import BenchmarkError


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalMetrics:
    """Rank-based retrieval quality over a set of queries.

    Two families of numbers are reported because they answer different questions and
    are easy to confuse:

    ``hit_at`` (also called *CMC* or top-k accuracy, written ``R@k`` in this project's
    reports) — the share of queries for which **at least one** same-identity image
    appears in the top ``k``. This is the standard re-identification metric and the one
    that answers "did the search find the cat at all?".

    ``recall_at`` — the share of **all** same-identity gallery images that appear in the
    top ``k``. With several references per cat this is much lower than ``hit_at`` and is
    the useful number when a user wants *every* photo of that cat.

    Reporting one while claiming the other is the single most common way a retrieval
    evaluation is accidentally overstated.
    """

    num_queries: int
    hit_at: dict[int, float]
    recall_at: dict[int, float]
    map_at: dict[int, float]
    mean_inverse_negative_penalty: float
    mean_reciprocal_rank: float
    mean_rank_of_first_match: float
    median_rank_of_first_match: float

    def to_dict(self) -> dict[str, float]:
        payload: dict[str, float] = {
            "num_queries": float(self.num_queries),
            "mINP": self.mean_inverse_negative_penalty,
            "mRR": self.mean_reciprocal_rank,
            "mean_first_match_rank": self.mean_rank_of_first_match,
            "median_first_match_rank": self.median_rank_of_first_match,
        }
        # ``hit@k`` (top-k accuracy) — named to match the Markdown table and the JSON, so a
        # single metric cannot appear under two different names in the same report.
        for k, value in self.hit_at.items():
            payload[f"hit@{k}"] = value
        for k, value in self.recall_at.items():
            payload[f"fullRecall@{k}"] = value
        for k, value in self.map_at.items():
            payload[f"mAP@{k}"] = value
        return payload


def _relevance_matrix(
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    exclude_self: np.ndarray | None,
) -> np.ndarray:
    relevant = query_labels[:, None] == gallery_labels[None, :]
    if exclude_self is not None:
        relevant &= ~exclude_self
    return relevant


def evaluate_retrieval(
    similarities: np.ndarray,
    query_labels: Sequence[str] | np.ndarray,
    gallery_labels: Sequence[str] | np.ndarray,
    recall_ks: Sequence[int] = (1, 5, 10),
    exclude_self: np.ndarray | None = None,
) -> RetrievalMetrics:
    """Compute rank metrics for a query/gallery similarity matrix.

    Args:
        similarities: ``(n_queries, n_gallery)`` higher-is-better scores.
        query_labels: Identity label per query row.
        gallery_labels: Identity label per gallery column.
        recall_ks: Cut-offs for hit rate, full-recall and mAP.
        exclude_self: Optional boolean mask marking (query, gallery) pairs that are
            the *same photo*. Excluding them prevents a self-match from being counted
            as a successful retrieval, which would inflate recall whenever the query
            image is also in the gallery.

    Returns:
        Aggregated :class:`RetrievalMetrics`.

    Raises:
        BenchmarkError: If the shapes disagree or no query has a relevant item.
    """
    sims = np.asarray(similarities, dtype=np.float64)
    q_labels = np.asarray(query_labels)
    g_labels = np.asarray(gallery_labels)

    if sims.ndim != 2:
        raise BenchmarkError(f"similarities must be 2-D, got shape {sims.shape}")
    if sims.shape[0] != q_labels.shape[0]:
        raise BenchmarkError(f"{sims.shape[0]} similarity rows but {q_labels.shape[0]} query labels")
    if sims.shape[1] != g_labels.shape[0]:
        raise BenchmarkError(f"{sims.shape[1]} similarity columns but {g_labels.shape[0]} gallery labels")

    relevant = _relevance_matrix(q_labels, g_labels, exclude_self)
    hits_per_query = relevant.sum(axis=1)
    usable = hits_per_query > 0
    if not usable.any():
        raise BenchmarkError("No query has a relevant gallery item; the protocol is not evaluable")

    sims = sims[usable]
    relevant = relevant[usable]
    hits_per_query = hits_per_query[usable]

    # Excluded pairs (the query's own gallery entry) must be removed from the *ranking*,
    # not merely marked irrelevant. Leaving them in place lets them occupy the top-k
    # slots, which understates R@k for every query whose own photo is in the gallery —
    # exactly the situation a production index is in.
    if exclude_self is not None:
        excluded = np.asarray(exclude_self, dtype=bool)[usable]
        if excluded.any():
            sims = np.where(excluded, -np.inf, sims)

    # Descending similarity; stable sort keeps gallery order for ties so results are
    # reproducible rather than dependent on the sort implementation.
    order = np.argsort(-sims, axis=1, kind="stable")
    ranked_relevant = np.take_along_axis(relevant, order, axis=1).astype(np.float64)

    cumulative = np.cumsum(ranked_relevant, axis=1)
    positions = np.arange(1, ranked_relevant.shape[1] + 1, dtype=np.float64)

    hit_at: dict[int, float] = {}
    recall_at: dict[int, float] = {}
    map_at: dict[int, float] = {}
    max_k = ranked_relevant.shape[1]
    for k in recall_ks:
        if k > max_k:
            continue
        hits = cumulative[:, k - 1]
        # ``hit_at`` counts queries with at least one same-identity neighbour in the
        # top k; ``recall_at`` counts how much of each identity's gallery was recovered.
        hit_at[k] = float((hits > 0).mean())
        recall_at[k] = float((hits / hits_per_query).mean())
        precision_at_k = cumulative[:, :k] / positions[:k]
        average_precision = (precision_at_k * ranked_relevant[:, :k]).sum(axis=1) / np.minimum(
            hits_per_query, k
        )
        map_at[k] = float(average_precision.mean())

    first_hit = np.argmax(ranked_relevant > 0, axis=1) + 1
    # INP: recall normalised by the rank of the *hardest* relevant item
    # (Ye et al., "Deep Learning for Person Re-identification: A Survey and Outlook").
    # Unlike mAP it does not saturate once k covers every relevant item, so it still
    # separates models that mAP cannot distinguish.
    #
    # Counting from the end is what makes it the *last* relevant rank: the index of the
    # first True in the reversed mask is how many positions away that rank is from the
    # end of the ranking, so the rank itself is N - that_index.
    any_relevant = ranked_relevant > 0
    distance_from_end = np.argmax(any_relevant[:, ::-1], axis=1)
    last_hit_rank = ranked_relevant.shape[1] - distance_from_end
    inverse_negative_penalty = (hits_per_query / last_hit_rank).mean()

    return RetrievalMetrics(
        num_queries=int(sims.shape[0]),
        hit_at=hit_at,
        recall_at=recall_at,
        map_at=map_at,
        mean_inverse_negative_penalty=float(inverse_negative_penalty),
        mean_reciprocal_rank=float((1.0 / first_hit).mean()),
        mean_rank_of_first_match=float(first_hit.mean()),
        median_rank_of_first_match=float(np.median(first_hit)),
    )


def bootstrap_recall_ci(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int = 1,
    samples: int = 500,
    confidence: float = 0.95,
    seed: int = 1337,
    exclude_self: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the top-k hit rate.

    A single-point number invites over-reading of small differences. The interval makes
    it visible when two configurations are statistically indistinguishable on the
    available query set.
    """
    sims = np.asarray(similarities, dtype=np.float64)
    q_labels = np.asarray(query_labels)
    g_labels = np.asarray(gallery_labels)
    relevant = _relevance_matrix(q_labels, g_labels, exclude_self)
    usable = relevant.sum(axis=1) > 0
    sims, relevant = sims[usable], relevant[usable]
    if sims.shape[0] == 0:
        raise BenchmarkError("Cannot bootstrap: no usable queries")

    order = np.argsort(-sims, axis=1, kind="stable")
    ranked = np.take_along_axis(relevant, order, axis=1)
    k = min(k, ranked.shape[1])
    per_query_hit = ranked[:, :k].any(axis=1).astype(np.float64)

    rng = np.random.default_rng(seed)
    n = per_query_hit.shape[0]
    draws = rng.integers(0, n, size=(samples, n))
    estimates = per_query_hit[draws].mean(axis=1)
    lower = float(np.quantile(estimates, (1 - confidence) / 2))
    upper = float(np.quantile(estimates, 1 - (1 - confidence) / 2))
    return lower, upper


def paired_bootstrap_delta(
    similarities_a: np.ndarray,
    similarities_b: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int = 1,
    samples: int = 500,
    seed: int = 1337,
    exclude_self: np.ndarray | None = None,
) -> dict[str, Any]:
    """Test whether model A beats model B on the *same* queries.

    Paired resampling is the correct test here: both models see identical queries, so
    the shared difficulty is cancelled out and much smaller differences become
    detectable. Returns the observed delta, a confidence interval, and the fraction of
    resamples in which A did not beat B (a one-sided p-value).
    """
    sims_a = np.asarray(similarities_a, dtype=np.float64)
    sims_b = np.asarray(similarities_b, dtype=np.float64)
    q_labels = np.asarray(query_labels)
    g_labels = np.asarray(gallery_labels)

    hits = []
    for sims in (sims_a, sims_b):
        relevant = _relevance_matrix(q_labels, g_labels, exclude_self)
        order = np.argsort(-sims, axis=1, kind="stable")
        ranked = np.take_along_axis(relevant, order, axis=1)
        hits.append(ranked[:, : min(k, ranked.shape[1])].any(axis=1).astype(np.float64))
    usable = _relevance_matrix(q_labels, g_labels, exclude_self).sum(axis=1) > 0
    hit_a, hit_b = hits[0][usable], hits[1][usable]

    observed = float(hit_a.mean() - hit_b.mean())
    rng = np.random.default_rng(seed)
    n = hit_a.shape[0]
    draws = rng.integers(0, n, size=(samples, n))
    deltas = hit_a[draws].mean(axis=1) - hit_b[draws].mean(axis=1)
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return {
        "observed_delta": observed,
        "ci95": [float(lower), float(upper)],
        "p_a_not_better": float((deltas <= 0).mean()),
        "samples": samples,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerificationMetrics:
    """Threshold-independent quality of a same/different score."""

    num_pairs: int
    num_positives: int
    roc_auc: float
    eer: float
    eer_threshold: float
    tar_at_far: dict[float, float]
    accuracy_at_eer: float
    mean_positive_score: float
    mean_negative_score: float

    @property
    def separation(self) -> float:
        """Difference of class means; a quick sanity check on score calibration."""
        return self.mean_positive_score - self.mean_negative_score

    def to_dict(self) -> dict[str, float]:
        payload: dict[str, float] = {
            "num_pairs": float(self.num_pairs),
            "num_positives": float(self.num_positives),
            "roc_auc": self.roc_auc,
            "eer": self.eer,
            "eer_threshold": self.eer_threshold,
            "accuracy_at_eer": self.accuracy_at_eer,
            "mean_positive_score": self.mean_positive_score,
            "mean_negative_score": self.mean_negative_score,
            "score_separation": self.separation,
        }
        for far, tar in self.tar_at_far.items():
            payload[f"TAR@FAR={far:g}"] = tar
        return payload


def evaluate_verification(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    far_targets: Sequence[float] = (1e-3, 1e-2, 1e-1),
) -> VerificationMetrics:
    """Compute ROC-AUC, EER and TAR at target false-accept rates.

    Args:
        scores: Similarity score per pair (higher = more likely the same individual).
        labels: ``1`` for same-identity pairs, ``0`` otherwise.
        far_targets: False-accept rates at which to report true-accept rate.

    Raises:
        BenchmarkError: If only one class is present, which makes AUC undefined.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if scores.shape != labels.shape:
        raise BenchmarkError(f"{scores.shape[0]} scores for {labels.shape[0]} labels")
    positives = labels == 1
    negatives = ~positives
    if positives.sum() == 0 or negatives.sum() == 0:
        raise BenchmarkError("Verification needs both same- and different-identity pairs")

    # ROC via rank statistics (Mann-Whitney U): exact, and avoids the tie-handling
    # ambiguity of a trapezoidal integration over a coarse threshold grid.
    order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.float64)
    # Average ranks across ties so AUC is unbiased when scores collide.
    sorted_scores = scores[order]
    start = 0
    while start < sorted_scores.shape[0]:
        end = start + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            average = (start + 1 + end) / 2.0
            ranks[order[start:end]] = average
        start = end

    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    auc = (ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    thresholds = np.unique(scores)[::-1]
    # Decision rule: accept when score >= threshold.
    decisions = scores[None, :] >= thresholds[:, None]
    true_positive = (decisions & positives[None, :]).sum(axis=1)
    false_positive = (decisions & negatives[None, :]).sum(axis=1)
    tar = true_positive / n_pos
    far = false_positive / n_neg

    # EER: interpolate where TAR crosses (1 - FAR).
    target = 1.0 - far
    crossing = np.where(np.diff(np.sign(tar - target)) != 0)[0]
    if crossing.size:
        index = int(crossing[0])
        # Linear interpolation between the bracketing thresholds.
        x0, x1 = tar[index] - target[index], tar[index + 1] - target[index + 1]
        span = x0 - x1
        weight = 0.0 if abs(span) < 1e-12 else x0 / span
        eer = float(1.0 - (target[index] + weight * (target[index + 1] - target[index])))
        eer_threshold = float(thresholds[index] + weight * (thresholds[index + 1] - thresholds[index]))
    else:
        # No crossing: the curves are dominated, EER is bounded by the best achievable.
        gap = np.abs(tar - target)
        index = int(np.argmin(gap))
        eer = float(1.0 - target[index])
        eer_threshold = float(thresholds[index])

    tar_at_far: dict[float, float] = {}
    for far_target in far_targets:
        eligible = np.where(far <= far_target)[0]
        tar_at_far[float(far_target)] = float(tar[eligible].max()) if eligible.size else 0.0

    accuracy_at_eer = float(((scores >= eer_threshold) == positives).mean())

    return VerificationMetrics(
        num_pairs=int(scores.shape[0]),
        num_positives=n_pos,
        roc_auc=float(auc),
        eer=eer,
        eer_threshold=eer_threshold,
        tar_at_far=tar_at_far,
        accuracy_at_eer=accuracy_at_eer,
        mean_positive_score=float(scores[positives].mean()),
        mean_negative_score=float(scores[negatives].mean()),
    )


def cosine_similarity_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Cosine similarity between every query and gallery row.

    Vectors are expected to be unit-norm already; the explicit renormalisation makes
    the function safe to call on un-normalised descriptors too.
    """
    q = np.asarray(query, dtype=np.float32)
    g = np.asarray(gallery, dtype=np.float32)
    if q.ndim != 2 or g.ndim != 2:
        raise BenchmarkError("cosine_similarity_matrix expects 2-D arrays")
    if q.shape[1] != g.shape[1]:
        raise BenchmarkError(f"dimension mismatch: {q.shape[1]} vs {g.shape[1]}")
    q_norm = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    g_norm = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    return (q_norm @ g_norm.T).astype(np.float32)


def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise cosine similarity between aligned rows of ``a`` and ``b``."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise BenchmarkError(f"pairwise_cosine expects equal shapes, got {a.shape} and {b.shape}")
    numerator = (a * b).sum(axis=1)
    denominator = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return (numerator / denominator).astype(np.float32)


__all__ = [
    "RetrievalMetrics",
    "VerificationMetrics",
    "bootstrap_recall_ci",
    "cosine_similarity_matrix",
    "evaluate_retrieval",
    "evaluate_verification",
    "paired_bootstrap_delta",
    "pairwise_cosine",
]
