"""Retrieval-algorithm research harness: same descriptors, several scorers, one table.

The scientific question
-----------------------
The trained DINOv2-S reaches ``hit@1 = 0.9682`` with a direct cosine ranking of a 12 141-image
gallery. The remaining 16 errors split into 5 label-collision defects and 11 genuine confusions.
The descriptors are frozen here on purpose: this harness answers "how much retrieval quality is
still available *inside* this embedding space", which is a different question from "would a better
backbone help" and much cheaper to answer.

Why graph methods are the right next step
-----------------------------------------
The protocol has roughly 24 gallery images per identity against 503 distinct identities, and the
similarity matrix is asymmetric in role but not in meaning: every gallery image is another sample
of *some* identity, and for an identity present in the gallery its own samples are mutual
neighbours. That is precisely the structure that neighbourhood-graph methods exploit, and it is
information a per-query sort throws away. Two classical formulations are implemented:

**k-reciprocal Jaccard re-ranking** (Zhong et al., CVPR 2017). For probe ``p`` with k-nearest
neighbour set ``N(p)``, the k-reciprocal set is

    R(p, k) = { g in N(p,k) : p in N(g,k) }

and it is expanded to ``R*(p,k) = R(p,k) ∪ R(q, k/2)`` for ``q`` in ``R(p,k)`` whose cardinality is
``2/3 |R(p,k)|``, which recovers true matches lost to a hard distractor. The re-ranked distance is
the Jaccard distance between those sets,

    d_J(p, g) = 1 - |R*(p,k) ∩ R*(g,k)| / |R*(p,k) ∪ R*(g,k)|.

The formulation is arithmetic on sets, so it is testable; its weakness is also structural: the
Jaccard term saturates once every sample of an identity is inside ``R*``, so it cannot separate two
identities that share most of their neighbourhoods.

**Diffusion / manifold ranking.** Let ``S`` be the row-normalised similarity graph over
queries ∪ gallery. Propagating an indicator vector ``y`` (1 at the probe, 0 elsewhere) by

    f <- alpha * S f + (1 - alpha) y

converges to ``f* = (1 - alpha)(I - alpha S)^-1 y``, the manifold-ranking solution. Equivalently it
is a random walk with restart of probability ``alpha``, and ``f*_j`` is the expected discounted
occupancy of node ``j``: similarity is accumulated along *paths*, so two images linked through a
chain of shared neighbours still reinforce each other even when their direct cosine is low. This is
the global counterpart to k-reciprocal's local set arithmetic, and it is why it can act on the
11 genuine errors, whose direct similarity is 0.44-0.67 and whose margin is negative.

``alpha`` is not a free knob to tune on the test set: for ``S`` row-normalised its spectral radius is
1, and the series converges iff ``alpha < 1``. Reported values of alpha are chosen on validation
identities, per this project's existing protocol.

Usage::

    python -m tools.retrieval_research --checkpoint artifacts/train/dinov2s-arcface/best.pt
    python -m tools.retrieval_research --checkpoint <ckpt> --methods baseline,kreciprocal,diffusion
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catface.config import load_config
from catface.data.manifest import Manifest
from catface.errors import CatFaceError
from catface.logging_utils import configure_utf8_console, get_logger
from catface.models.embedder import Embedder, embed_records
from catface.pipeline import build_within_split
from tools.embedding_cache import DEFAULT_CACHE_DIR, cache_key, load_or_embed

LOGGER = get_logger("tools.retrieval_research")


# --------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------
def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, eps)


def retrieval_metrics(similarity: np.ndarray, query_labels: list[str],
                      gallery_labels: list[str]) -> dict[str, float]:
    """CMC hit@k, mINP, mAP over all relevant gallery items, mRR.

    ``hit@k`` is the share of queries whose top-k contains at least one same-identity image.
    ``fullR@k`` (what share of that identity's gallery appears in the top-k) is not computed here;
    the two differ by a factor of ~25 on this corpus and conflating them is the usual way retrieval
    numbers get overstated.
    """
    labels_query = np.asarray(query_labels)
    labels_gallery = np.asarray(gallery_labels)
    relevant = labels_query[:, None] == labels_gallery[None, :]
    hits_per_query = relevant.sum(axis=1)

    order = np.argsort(-similarity, axis=1, kind="stable")
    ranked_relevant = np.take_along_axis(relevant, order, axis=1)

    out: dict[str, float] = {}
    n = similarity.shape[0]
    for k in (1, 5, 10):
        kk = min(k, ranked_relevant.shape[1])
        out[f"hit@{k}"] = float(ranked_relevant[:, :kk].any(axis=1).mean())

    ranks = np.argmax(ranked_relevant, axis=1) + 1
    out["mRR"] = float((1.0 / ranks).mean())

    precision_at = np.cumsum(ranked_relevant, axis=1) / np.arange(1, ranked_relevant.shape[1] + 1)
    average_precision = (precision_at * ranked_relevant).sum(axis=1) / np.maximum(hits_per_query, 1)
    out["mAP"] = float(average_precision.mean())

    # mINP: mean over queries of (hits / rank of the last relevant item). Unlike mAP it keeps
    # discriminating after recall saturates, which is the regime this system is in.
    last_hit = ranked_relevant.shape[1] - np.argmax(ranked_relevant[:, ::-1], axis=1)
    out["mINP"] = float((hits_per_query / last_hit).mean())
    out["queries"] = float(n)
    return out


# --------------------------------------------------------------------------------------------
# algorithms
# --------------------------------------------------------------------------------------------
def score_baseline(probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Direct cosine similarity under a linear scan."""
    return (probe @ gallery.T).astype(np.float32)


def _knn(similarity: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    """Indices of the top-``k`` columns per row, with the diagonal removed when it is the self-match.

    A probe and a gallery image are distinct sets here, so ``exclude_self`` is only meaningful for
    the gallery-to-gallery neighbour lists.
    """
    scores = similarity.copy()
    if exclude_self:
        scores[np.arange(scores.shape[0]), np.arange(scores.shape[0])] = -np.inf
    k = int(min(k, scores.shape[1] - 1 if exclude_self else scores.shape[1]))
    if k < 1:
        return np.empty((scores.shape[0], 0), dtype=np.int64)
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    ordered = np.take_along_axis(scores, top, axis=1)
    return np.take_along_axis(top, np.argsort(-ordered, axis=1, kind="stable"), axis=1)


def score_kreciprocal(query: np.ndarray, gallery: np.ndarray, k1: int = 20,
                      k2: int = 6, lambda_value: float = 0.3) -> np.ndarray:
    """k-reciprocal Jaccard re-ranking (Zhong et al., CVPR 2017).

    The probe and the gallery are treated as one symmetric graph for neighbourhood purposes:
    ``N(p)`` is computed over the gallery, ``N(g)`` over the gallery plus the probe itself, so a
    probe is findable inside a gallery image's neighbour list and mutual-neighbour sets are
    well defined. That is the standard setting when the probe is not part of the index.

    Cost: O((Q+R) * k1 * (R + k2)), dominated by the pairwise Jaccard expansion. It is computed once
    per candidate parameter set, so the sweep is affordable but not free.
    """
    query = l2_normalize(query)
    gallery = l2_normalize(gallery)
    n_g = gallery.shape[0]

    similarity_qg = (query @ gallery.T).astype(np.float32)
    similarity_gg = (gallery @ gallery.T).astype(np.float32)

    # Neighbour lists over the gallery, for probes and for gallery images alike.
    n_qg = _knn(similarity_qg, k1, exclude_self=False)
    n_gg = _knn(similarity_gg, k1, exclude_self=True)
    n_gg_half = _knn(similarity_gg, k2, exclude_self=True)

    # Materialise memberships once. Rebuilding a set per (row, item) pair inside the loop below is
    # O(k1^2) set constructions per row and dominated the whole re-ranking in a first attempt.
    n_gg_sets = [{int(x) for x in row} for row in n_gg]

    def reciprocal_sets(neighbours: np.ndarray) -> list[set[int]]:
        """R*(p, k1) as a list of gallery-index sets."""
        result: list[set[int]] = []
        for row in range(neighbours.shape[0]):
            base = [int(x) for x in neighbours[row]]
            reciprocal = {item for item in base if row in n_gg_sets[item]}
            expanded = set(reciprocal)
            for item in reciprocal:
                # The 2/3 rule: a neighbour whose own reciprocal set is unusually small is likely a
                # true match demoted by a hard distractor, so its half-k neighbours are folded in.
                if len(n_gg_sets[item]) > 2.0 / 3.0 * len(reciprocal):
                    expanded.update(int(x) for x in n_gg_half[item])
            result.append(expanded)
        return result

    probe_sets = reciprocal_sets(n_qg)
    gallery_sets = reciprocal_sets(n_gg)

    # Convert to boolean membership matrices so the Jaccard numerator becomes a matmul.
    def to_matrix(sets: list[set[int]]) -> np.ndarray:
        matrix = np.zeros((len(sets), n_g), dtype=np.float32)
        for row, members in enumerate(sets):
            if members:
                matrix[row, list(members)] = 1.0
        return matrix

    probe_membership = to_matrix(probe_sets)
    gallery_membership = to_matrix(gallery_sets)
    gallery_sizes = gallery_membership.sum(axis=1)

    intersection = probe_membership @ gallery_membership.T
    union = (probe_membership.sum(axis=1)[:, None] + gallery_sizes[None, :] - intersection)
    jaccard = intersection / np.maximum(union, 1e-12)

    # The final score blends the original cosine with the Jaccard distance, which keeps direct
    # appearance evidence in play instead of letting the set structure override it.
    return (similarity_qg + lambda_value * jaccard).astype(np.float32)


def score_diffusion(query: np.ndarray, gallery: np.ndarray, alpha: float = 0.9,
                    steps: int = 20, k_graph: int = 12, tol: float = 1e-6) -> np.ndarray:
    """Manifold ranking by random walk with restart over the gallery similarity graph.

    Solves ``f* = alpha S f* + (1 - alpha) y`` by iteration; the closed form
    ``(1 - alpha)(I - alpha S)^-1 y`` is the same thing but inverting a 12 644 x 12 644 matrix is
    pointless when the fixed-point iteration converges geometrically at rate ``alpha``.

    ``k_graph`` sparsifies ``S`` by keeping each node's strongest edges, which
    (a) removes the long tail of near-zero similarities that otherwise lets probability mass leak
    across unrelated identities, and (b) turns dense propagation into a sparse matvec.

    Asymmetry is deliberate: ``S`` is row-normalised, so a popular node does not accumulate rank
    merely by being close to many others.
    """
    query = l2_normalize(query)
    gallery = l2_normalize(gallery)
    n_g = gallery.shape[0]

    similarity_gg = (gallery @ gallery.T).astype(np.float32)
    np.fill_diagonal(similarity_gg, 0.0)

    if k_graph and k_graph < n_g:
        keep = _knn(similarity_gg, k_graph, exclude_self=False)
        sparse = np.zeros_like(similarity_gg)
        rows = np.repeat(np.arange(n_g), keep.shape[1])
        sparse[rows, keep.ravel()] = np.maximum(similarity_gg[rows, keep.ravel()], 0.0)
        similarity_gg = sparse
    else:
        np.maximum(similarity_gg, 0.0, out=similarity_gg)

    row_sum = similarity_gg.sum(axis=1, keepdims=True)
    transition = similarity_gg / np.maximum(row_sum, 1e-12)

    similarity_qg = (query @ gallery.T).astype(np.float32)
    # Restart distribution: the probe's own affinity to each gallery node, sparsified the same way
    # so a probe is not restarted into nodes it has no real relation with.
    restart = np.maximum(similarity_qg, 0.0)
    restart /= np.maximum(restart.sum(axis=1, keepdims=True), 1e-12)

    field = restart.copy()
    for _ in range(int(steps)):
        updated = alpha * (field @ transition) + (1.0 - alpha) * restart
        delta = float(np.abs(updated - field).max())
        field = updated
        if delta < tol:
            break

    # Combine the propagated field with the direct cosine: diffusion is a ranking of *reachability*
    # and is noisier for probes whose identity is sparsely represented, so it is used as a re-ranker
    # rather than as the sole score.
    return (similarity_qg + field).astype(np.float32)


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------
METHODS = ("baseline", "kreciprocal", "diffusion")


def build_protocol(config_path: str, manifest_override: str | None,
                   queries_per_identity: int, seed: int):
    config = load_config(config_path)
    manifest_path = Path(manifest_override) if manifest_override else (
        Path(config.data.manifest) / "cat_individuals_manifest.jsonl")
    if not manifest_path.is_file():
        raise CatFaceError(f"manifest missing at {manifest_path}")
    records = list(Manifest.load(manifest_path))
    split = build_within_split(records, queries_per_identity=queries_per_identity,
                              name="cat_individuals", seed=seed)
    return records, split


def embed_split(embedder: Embedder, records, split, image_size: int,
                views: Sequence[str] | None = None) -> dict:
    query = embed_records(embedder, [r.path for r in split.query_records],
                          image_size=image_size, batch_size=32, views=views)
    gallery = embed_records(embedder, [r.path for r in split.gallery_records],
                            image_size=image_size, batch_size=32, views=views)
    return {
        "query": np.asarray(query.vectors, dtype=np.float32),
        "gallery": np.asarray(gallery.vectors, dtype=np.float32),
        "query_ids": [r.image_id for r in split.query_records],
        "gallery_ids": [r.image_id for r in split.gallery_records],
        "query_labels": [r.identity for r in split.query_records],
        "gallery_labels": [r.identity for r in split.gallery_records],
        "query_paths": [r.path for r in split.query_records],
        "gallery_paths": [r.path for r in split.gallery_records],
    }


def main(argv: list[str] | None = None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Compare retrieval scorers on frozen descriptors")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--queries-per-identity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--methods", default="baseline,kreciprocal,diffusion")
    parser.add_argument("--views", default="identity,hflip",
                        help="TTA views, comma-separated; identity,hflip,scale_112 is available")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-embed", action="store_true",
                        help="ignore the cache and re-encode")
    parser.add_argument("--k1", type=int, default=20, help="k-reciprocal primary neighbourhood")
    parser.add_argument("--k2", type=int, default=6, help="k-reciprocal expansion neighbourhood")
    parser.add_argument("--alpha", type=float, default=0.9, help="diffusion restart complement")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--k-graph", type=int, default=12)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    records, split = build_protocol(args.config, args.manifest,
                                    args.queries_per_identity, args.seed)

    views = tuple(name.strip() for name in args.views.split(",") if name.strip())
    manifest_for_key = Path(args.manifest) if args.manifest else (
        Path(load_config(args.config).data.manifest) / "cat_individuals_manifest.jsonl")
    key = cache_key(
        args.checkpoint, manifest_for_key,
        protocol="cat_individuals", image_size=args.image_size, tta=views,
        extra={"queries_per_identity": args.queries_per_identity, "seed": args.seed},
    )

    def producer() -> dict:
        embedder = Embedder.load(args.checkpoint, device=args.device or "cuda")
        LOGGER.info("loaded %s (backbone=%s)", args.checkpoint, embedder.config.backbone)
        return embed_split(embedder, records, split, args.image_size, views)

    payload, cached = load_or_embed(key, producer, directory=args.cache_dir,
                                    force=args.force_embed)
    LOGGER.info("descriptors %s (cache key %s, dim %d, %d queries / %d gallery)",
                "loaded from cache" if cached else "freshly embedded",
                key, payload["query"].shape[1], payload["query"].shape[0],
                payload["gallery"].shape[0])

    wanted = [name.strip() for name in args.methods.split(",") if name.strip()]
    results: list[dict] = []
    for name in wanted:
        if name not in METHODS:
            raise CatFaceError(f"unknown method {name!r}; choose from {METHODS}")
        started = time.perf_counter()
        if name == "baseline":
            similarity = score_baseline(payload["query"], payload["gallery"])
        elif name == "kreciprocal":
            similarity = score_kreciprocal(payload["query"], payload["gallery"],
                                           k1=args.k1, k2=args.k2)
        else:
            similarity = score_diffusion(payload["query"], payload["gallery"],
                                         alpha=args.alpha, steps=args.steps,
                                         k_graph=args.k_graph)
        elapsed = time.perf_counter() - started
        metrics = retrieval_metrics(similarity, payload["query_labels"], payload["gallery_labels"])
        results.append({"method": name, "seconds": round(elapsed, 2), **metrics})
        LOGGER.info("%-13s hit@1=%.4f hit@5=%.4f mINP=%.4f mAP=%.4f mRR=%.4f  (%.1fs)",
                    name, metrics["hit@1"], metrics["hit@5"], metrics["mINP"],
                    metrics["mAP"], metrics["mRR"], elapsed)

    base = next((row for row in results if row["method"] == "baseline"), None)
    print()
    header = f"{'method':<14}{'hit@1':>8}{'hit@5':>8}{'mINP':>8}{'mAP':>8}{'mRR':>8}{'sec':>7}"
    print(header)
    print("-" * len(header))
    for row in results:
        delta = ""
        if base and row["method"] != "baseline":
            delta = f"   (Δhit@1 {row['hit@1'] - base['hit@1']:+.4f})"
        print(f"{row['method']:<14}{row['hit@1']:>8.4f}{row['hit@5']:>8.4f}"
              f"{row['mINP']:>8.4f}{row['mAP']:>8.4f}{row['mRR']:>8.4f}{row['seconds']:>7.1f}{delta}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "checkpoint": args.checkpoint, "cache_key": key, "cached": cached,
            "protocol": {"queries": int(payload["query"].shape[0]),
                         "gallery": int(payload["gallery"].shape[0]),
                         "dim": int(payload["query"].shape[1])},
            "params": {"k1": args.k1, "k2": args.k2, "alpha": args.alpha,
                       "steps": args.steps, "k_graph": args.k_graph},
            "results": results,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
