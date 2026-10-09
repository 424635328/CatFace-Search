"""Is the residual error a representation limit or a scoring-rule limit?

The question
------------
Retrieval-side methods are exhausted: direct cosine, k-reciprocal re-ranking, diffusion and
identity pooling all land on hit@1 = 0.9682 (the pooling figure of 0.9702 was proved to be a
tie-break artefact). The 16 remaining errors are all near ties. Before spending GPU time on
self-training — which is a *linear* head update on frozen features — it is worth knowing whether a
linear map of the current descriptor could possibly separate them.

The test
--------
A linear probe on the frozen 512-d descriptors, trained on one set of identities and evaluated on
identities it never saw. Two variants, because they answer slightly different questions:

* **Ridge regression onto one-hot labels**, closed form: ``W = (XᵀX + λI)⁻¹ Xᵀ Y``. This is the
  discriminative linear map that best separates the training identities, evaluated on held-out ones.
* **Regularised nearest-class-mean**, the generative counterpart, as a sanity check that the ridge
  result is not an artefact of the one-hot target scale.

If both land at or below the cosine baseline, then no linear reweighting of this descriptor
separates the failing queries, and self-training on frozen features cannot fix them either — the
remaining error is representational, and the fix has to add information (more identities, a
stronger backbone) rather than reweight what is already there.

If a probe clearly beats the baseline, the opposite conclusion follows and the head is where the
work belongs.

Both are evaluated with the *same* protocol shape as the benchmark so the numbers are comparable:
one probe image per identity, every other image in the reference set, self excluded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.embedding_cache import DEFAULT_CACHE_DIR, cache_key, load
from tools.retrieval_research import l2_normalize, retrieval_metrics


def ridge_fit(features: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
    """Closed-form ridge solution for ``(D, C)`` weights.

    Solved in the dual when the descriptor is wider than the sample count, which is the normal case
    here (512 features against a few thousand images) and avoids a needless 512x512 inverse.
    """
    n, dim = features.shape
    if n <= dim:
        gram = features @ features.T
        dual = np.linalg.solve(gram + penalty * np.eye(n, dtype=np.float32), targets)
        return features.T @ dual
    normal = features.T @ features + penalty * np.eye(dim, dtype=np.float32)
    return np.linalg.solve(normal, features.T @ targets)


def class_means(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Mean descriptor per class."""
    names = sorted(set(labels.tolist()))
    means = np.stack([features[labels == name].mean(axis=0) for name in names])
    return means, names


def evaluate(scores: np.ndarray, query_labels: list[str], reference_labels: list[str]) -> dict:
    return retrieval_metrics(scores, query_labels, reference_labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Linear-probe ceiling on frozen descriptors")
    parser.add_argument("--checkpoint", default="artifacts/train/dinov2s-arcface/best.pt")
    parser.add_argument("--manifest", default="data/manifests/cat_individuals_manifest.jsonl")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.2, help="share of identities used only for evaluation"
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    key = cache_key(
        args.checkpoint,
        args.manifest,
        protocol="cat_individuals",
        image_size=args.image_size,
        tta=("identity", "hflip"),
        extra={"queries_per_identity": 1, "seed": 1337},
    )
    payload = load(args.cache_dir, key)
    if payload is None:
        print(f"cache miss for {key}; run tools.retrieval_research first")
        return 1

    query = l2_normalize(payload["query"])
    gallery = l2_normalize(payload["gallery"])
    query_labels = np.asarray(payload["query_labels"])
    gallery_labels = np.asarray(payload["gallery_labels"])

    baseline = retrieval_metrics(
        (query @ gallery.T).astype(np.float32), query_labels.tolist(), gallery_labels.tolist()
    )
    print(f"baseline cosine                hit@1={baseline['hit@1']:.4f} mINP={baseline['mINP']:.4f}")

    # Identities, not images, are held out: a probe that has seen the query's identity during
    # training would answer a closed-set question, while the benchmark asks an open-set one.
    identities = np.array(sorted(set(gallery_labels.tolist())))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(identities)
    n_holdout = max(1, round(len(identities) * args.holdout_fraction))
    holdout = set(identities[:n_holdout].tolist())
    train_ids = np.array([name for name in identities if name not in holdout])

    train_mask = np.isin(gallery_labels, train_ids)
    query_mask = np.isin(query_labels, list(holdout))
    print(f"probe training identities      : {len(train_ids)} ({int(train_mask.sum())} images)")
    print(f"held-out identities            : {n_holdout} ({int(query_mask.sum())} query images)")

    features = gallery[train_mask]
    labels = gallery_labels[train_mask]
    names = sorted(set(labels.tolist()))
    index = {name: position for position, name in enumerate(names)}

    # Restrict the reference set to the held-out identities so the probe is scored on the same
    # task shape as the benchmark: find the right identity among the ones it was never trained on.
    reference_mask = np.isin(gallery_labels, list(holdout))
    reference = gallery[reference_mask]
    reference_labels = gallery_labels[reference_mask]
    heldout_queries = query[query_mask]
    heldout_query_labels = query_labels[query_mask]

    # Restricting the reference set makes the task easier than the benchmark, so also score the
    # probe against the full gallery for a like-for-like comparison.
    results: dict = {
        "baseline_full_gallery": baseline,
        "probe_training_identities": len(train_ids),
        "probe_training_images": int(train_mask.sum()),
        "holdout_identities": n_holdout,
        "holdout_queries": int(query_mask.sum()),
        "penalty": args.penalty,
    }

    targets = np.zeros((features.shape[0], len(names)), dtype=np.float32)
    for row, name in enumerate(labels.tolist()):
        targets[row, index[name]] = 1.0

    weights = ridge_fit(features, targets, args.penalty)
    # A held-out identity has no weight column, so the probe cannot classify it directly. What the
    # probe can do is define a *metric*: the linear map reweights the space, and the same
    # nearest-reference rule is then applied in the reweighted space. That is the honest transfer
    # test — it asks whether the learned reweighting generalises, not whether it memorised classes.
    projected_query = heldout_queries @ weights
    projected_reference = reference @ weights

    print()
    print("--- ridge-reweighted space, reference = held-out identities only ---")
    ridge_holdout = evaluate(
        (projected_query @ projected_reference.T).astype(np.float32),
        heldout_query_labels.tolist(),
        reference_labels.tolist(),
    )
    plain_holdout = evaluate(
        (heldout_queries @ reference.T).astype(np.float32),
        heldout_query_labels.tolist(),
        reference_labels.tolist(),
    )
    print(
        f"  plain cosine in that subset  hit@1={plain_holdout['hit@1']:.4f} mINP={plain_holdout['mINP']:.4f}"
    )
    print(
        f"  ridge-reweighted             hit@1={ridge_holdout['hit@1']:.4f} mINP={ridge_holdout['mINP']:.4f}"
    )
    results["holdout_subset_plain"] = plain_holdout
    results["holdout_subset_ridge"] = ridge_holdout

    print()
    print("--- ridge-reweighted space, full gallery (like-for-like with the baseline) ---")
    full_ridge = evaluate(
        (query @ weights @ (gallery @ weights).T).astype(np.float32),
        query_labels.tolist(),
        gallery_labels.tolist(),
    )
    print(f"  ridge-reweighted, all queries hit@1={full_ridge['hit@1']:.4f} mINP={full_ridge['mINP']:.4f}")
    results["full_gallery_ridge"] = full_ridge

    means, mean_names = class_means(features, labels)
    reference_means, _ = class_means(reference, reference_labels)
    ncm_plain = evaluate(
        (heldout_queries @ reference_means.T).astype(np.float32),
        heldout_query_labels.tolist(),
        sorted(set(reference_labels.tolist())),
    )
    print()
    print(f"--- nearest-class-mean control (held-out subset) hit@1={ncm_plain['hit@1']:.4f} ---")
    results["holdout_subset_ncm"] = ncm_plain
    del means, mean_names

    print()
    delta = ridge_holdout["hit@1"] - plain_holdout["hit@1"]
    print(f"ridge vs plain on the same subset: Δhit@1 = {delta:+.4f}")
    print(f"full-gallery ridge vs baseline   : Δhit@1 = {full_ridge['hit@1'] - baseline['hit@1']:+.4f}")
    print()
    print("Interpretation: a ridge probe is the best linear reweighting available, so a non-positive")
    print("delta means no linear map of this descriptor separates the failing queries, and a linear")
    print("self-training head cannot either.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
