"""Descriptor post-processing — accuracy gains without retraining.

These transforms are *label-free*: they use only the gallery descriptors, so they
can be applied to a deployed index without touching the model. On face and
fine-grained retrieval benchmarks each of them is worth more than a backbone
upgrade, and they compose:

* **PCA-whitening** removes the dominant "this is a cat" variance directions that
  every cat face shares. Those directions dominate cosine similarity while carrying
  no identity information.
* **DBA (database-side augmentation)** replaces each reference descriptor with the
  weighted mean of its own nearest neighbours, which denoises single-photo
  references.
* **αQE (alpha query expansion)** builds a stronger query from both the query's
  neighbours *and* the query's reverse neighbours, which is far more robust than
  plain average query expansion when the top-1 hit is wrong.

References: Jégou & Chum (DBA, 2012); Radenović et al. (αQE, 2018);
Tian et al. / Luo et al. (BN-whitening for person re-ID).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import BenchmarkError
from ..logging_utils import get_logger

LOGGER = get_logger("eval.postprocess")


def l2_normalize(matrix: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation; a no-op for zero rows instead of NaN."""
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    return (array / np.maximum(norms, eps)).astype(np.float32)


@dataclass
class WhiteningTransform:
    """PCA / PCA-whitening fitted on a reference set.

    ``pca`` keeps the principal subspace (variance retained, magnitude structure
    preserved). ``pcaw`` additionally rescales each retained direction to unit
    variance, which is what removes the common-mode "all cats look alike" component.

    The transform is always *fitted on the gallery only* — fitting on gallery+query
    would leak evaluation information into the descriptor and is one of the most
    common ways a retrieval benchmark reports numbers that do not reproduce in
    production.
    """

    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    mode: str = "pcaw"
    epsilon: float = 0.0

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        dim: int = 0,
        mode: str = "pcaw",
        epsilon: float = 0.0,
    ) -> WhiteningTransform:
        """Fit on ``features`` (``(N, D)``), keeping ``dim`` components (0 = all)."""
        matrix = l2_normalize(features)
        if matrix.shape[0] < 2:
            raise BenchmarkError("Whitening needs at least 2 reference descriptors")

        mean = matrix.mean(axis=0, keepdims=True)
        centered = matrix - mean
        # Economy SVD; ``components`` are the right singular vectors.
        # Use the Gram trick when D >> N to keep this cheap on 30k x 768 matrices.
        n_samples, n_dims = centered.shape
        if n_samples < n_dims:
            gram = centered @ centered.T
            eigenvalues, vectors = np.linalg.eigh(gram)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            vectors = vectors[:, order]
            singular = np.sqrt(eigenvalues)
            keep = dim if dim > 0 else min(n_samples, n_dims)
            keep = min(keep, vectors.shape[1])
            components = (vectors[:, :keep].T @ centered) / np.maximum(
                singular[:keep, None], 1e-12
            )
        else:
            covariance = (centered.T @ centered) / max(n_samples - 1, 1)
            eigenvalues, vectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            components = vectors[:, order].T
            keep = dim if dim > 0 else components.shape[0]
            components = components[: min(keep, components.shape[0])]

        selected = eigenvalues[: components.shape[0]]
        scale = np.sqrt(np.maximum(selected, 1e-12)) + epsilon
        return cls(
            mean=mean.astype(np.float32),
            components=components.astype(np.float32),
            eigenvalues=scale.astype(np.float32),
            mode=mode,
            epsilon=epsilon,
        )

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def explain_variance(self) -> float:
        """Fraction of total variance captured by the retained components."""
        total = float(np.sum(self.eigenvalues**2))
        return 1.0 if total <= 0 else float(np.sum(self.eigenvalues**2) / total)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Project and (optionally) whiten, then renormalise."""
        matrix = l2_normalize(features)
        projected = (matrix - self.mean) @ self.components.T
        if self.mode == "pcaw":
            projected = projected / self.eigenvalues
        return l2_normalize(projected)


def database_side_augmentation(
    features: np.ndarray,
    k: int = 3,
    alpha: float = 3.0,
    chunk: int = 4096,
) -> np.ndarray:
    """Replace each descriptor by its neighbourhood-weighted mean (DBA).

    Weight ``sim**alpha`` concentrates the average on the closest neighbours, so a
    single outlier in the top-k cannot drag the augmented descriptor away.

    Args:
        features: ``(N, D)`` reference descriptors.
        k: Neighbourhood size (excluding the descriptor itself).
        alpha: Weight exponent applied to the neighbour similarities.
        chunk: Rows processed per similarity block; bounds peak memory.

    Returns:
        ``(N, D)`` augmented, renormalised descriptors.
    """
    matrix = l2_normalize(features)
    n, _dim = matrix.shape
    k = int(min(k, n - 1))
    if k < 1:
        return matrix

    out = np.empty_like(matrix)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        similarity = matrix[start:stop] @ matrix.T  # (chunk, N)
        # Ignore the self-match without assuming argument order.
        similarity[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        weights = np.take_along_axis(similarity, top, axis=1).astype(np.float64)
        weights = np.power(np.maximum(weights, 0.0), alpha)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        neighbours = matrix[top]  # (chunk, k, D)
        augmented = (neighbours * weights[:, :, None]).sum(axis=1)
        combined = alpha * matrix[start:stop] + augmented
        out[start:stop] = combined
    return l2_normalize(out)


@dataclass
class AlphaQueryExpansion:
    """Configures αQE for one retrieval call."""

    enabled: bool = False
    top_k: int = 3
    alpha: float = 3.0
    reverse_k: int = 2
    reverse_weight: float = 0.5
    """Weight of the reverse-neighbour term relative to the forward term."""


def alpha_query_expansion(
    query: np.ndarray,
    gallery: np.ndarray,
    config: AlphaQueryExpansion,
) -> np.ndarray:
    """Expand each query using forward and reverse gallery neighbours.

    Args:
        query: ``(Q, D)`` query descriptors.
        gallery: ``(N, D)`` unit-norm gallery descriptors.
        config: Expansion parameters.

    Returns:
        ``(Q, D)`` expanded, renormalised query descriptors.
    """
    if not config.enabled:
        return l2_normalize(query)

    q = l2_normalize(query)
    g = l2_normalize(gallery)
    k = int(min(config.top_k, g.shape[0]))
    if k < 1:
        return q

    similarity = q @ g.T  # (Q, N)
    top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    forward_weights = np.take_along_axis(similarity, top, axis=1).astype(np.float64)
    forward_weights = np.power(np.maximum(forward_weights, 0.0), config.alpha)
    forward_weights /= np.maximum(forward_weights.sum(axis=1, keepdims=True), 1e-12)
    forward = (g[top] * forward_weights[:, :, None]).sum(axis=1)

    expanded = config.alpha * q + forward

    if config.reverse_k > 0 and config.reverse_weight > 0:
        # Reverse neighbours: for each selected gallery item, is this query among its
        # own top-k? If yes the match is mutual, which is strong evidence.
        selected = g[top]  # (Q, k, D)
        reverse_similarity = selected @ g.T  # (Q, k, N)
        rk = int(min(config.reverse_k, g.shape[0]))
        reverse_top = np.argpartition(-reverse_similarity, rk - 1, axis=2)[:, :, :rk]
        # (Q, k, rk, D) gather
        reverse_neighbours = g[reverse_top]
        mutual = (reverse_neighbours == q[:, None, None, :]).all(axis=-1).any(axis=2)
        mutual_weight = mutual.astype(np.float64)  # (Q, k)
        reverse_term = (selected * mutual_weight[:, :, None]).sum(axis=1)
        reverse_term /= np.maximum(mutual_weight.sum(axis=1, keepdims=True), 1e-12)
        expanded = expanded + config.reverse_weight * reverse_term

    return l2_normalize(expanded)


__all__ = [
    "AlphaQueryExpansion",
    "WhiteningTransform",
    "alpha_query_expansion",
    "database_side_augmentation",
    "l2_normalize",
]
