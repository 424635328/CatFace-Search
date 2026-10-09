"""The random-ranking null used to interpret retrieval errors.

The aggregate metric cannot say how close a failure was. That is what the hypergeometric null is
for, and it is the only number in the research report that quantifies "the descriptor nearly got
it" — so it is verified against exhaustive enumeration rather than trusted.

The production implementation uses a running-product form of the survival function because the
binomial form materialises integers with thousands of digits and timed out. Two forms of the same
formula is exactly the situation where a silent algebra slip survives review, hence this test.
"""

from __future__ import annotations

import importlib.util
import itertools
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


error_analysis = _load("analyse_retrieval_error")


def brute_force_bucket_probabilities(gallery: int, relevant: int, edges: list[int]) -> list[float]:
    """Enumerate every permutation and record where the best relevant item lands."""
    counts = [0] * (len(edges) + 1)
    total = 0
    for permutation in itertools.permutations(range(gallery)):
        total += 1
        best = next(index + 1 for index, item in enumerate(permutation) if item < relevant)
        for bucket, edge in enumerate(edges):
            if best <= edge:
                counts[bucket] += 1
                break
        else:
            counts[-1] += 1
    return [count / total for count in counts]


def _one_query(gallery: int, relevant: int) -> tuple[np.ndarray, np.ndarray]:
    similarity = np.zeros((1, gallery), dtype=np.float32)
    mask = np.zeros((1, gallery), dtype=bool)
    mask[0, :relevant] = True
    return similarity, mask


class TestRandomRankingNull:
    def test_running_product_matches_exhaustive_enumeration(self):
        edges = [1, 2, 5]
        similarity, relevant = _one_query(6, 2)
        fast = error_analysis.random_ranking_expectation(similarity, relevant, edges)
        brute = brute_force_bucket_probabilities(6, 2, edges)
        assert np.allclose(fast, brute, atol=1e-9), (
            f"the fast survival form disagrees with brute force: {fast} vs {brute}"
        )
        assert abs(sum(brute) - 1.0) < 1e-9, "the brute-force distribution must be normalised"

    def test_probabilities_are_normalised_and_monotone_in_relevance(self):
        """A gallery where everything is relevant must always put the best match at rank 1."""
        similarity, relevant = _one_query(8, 8)
        expected = error_analysis.random_ranking_expectation(similarity, relevant, [1, 2, 5])
        assert expected[0] == 1.0
        assert sum(expected) == 1.0

    def test_more_relevant_items_means_a_better_expected_rank(self):
        """Sanity: the null must be sensitive to how many true matches exist."""
        fewer = error_analysis.random_ranking_expectation(*_one_query(100, 2), [1, 2, 5])
        more = error_analysis.random_ranking_expectation(*_one_query(100, 20), [1, 2, 5])
        assert more[0] > fewer[0], "more true matches must raise P(best rank == 1)"
        assert sum(fewer) == pytest.approx(1.0)
        assert sum(more) == pytest.approx(1.0)

    def test_a_query_with_no_relevant_item_is_skipped(self):
        """An unanswerable query would otherwise add probability to every bucket."""
        similarity, relevant = _one_query(10, 0)
        expected = error_analysis.random_ranking_expectation(similarity, relevant, [1, 2, 5])
        assert expected[0] == 0.0, "no relevant item means rank 1 is impossible"
        assert all(value == 0.0 for value in expected)
