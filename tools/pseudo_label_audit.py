"""Is self-training on this gallery viable? Measure before building it.

The situation
-------------
Training used 352 identities over 8 833 images. The evaluation gallery holds 503 *other* identities
over 12 141 images, and those descriptors are already computed. Self-training on the gallery would
raise the effective training set by roughly 37% without a single new label — the only lever left
after retrieval was shown to be exhausted inside the current embedding.

Why this script comes first
---------------------------
Self-training fails in a specific, measurable way: pseudo-label errors are amplified, because the
model is trained to reproduce its own mistakes. Whether that happens is decided by the *purity* of
the pseudo-labels at the coverage the training would use — a quantity that costs one k-NN pass to
measure and a GPU-hour to discover by training. So it is measured first.

The design avoids the trap of grading your own homework:

* the 503 identities are split into two disjoint folds, A and B;
* each fold's images are pseudo-labelled by nearest neighbours **in the other fold**, which contains
  the same identities but different photographs;
* purity is therefore measured against the real labels of the labelled fold, not against the
  predictions that produced it — a label that is wrong is visible as wrong.

Because both folds contain every identity, this measures *instance-level* pseudo-label quality, which
is exactly what a training signal consumes. k-NN consensus (top-k vote) is compared against top-1
nearest-neighbour labelling, since the vote is what the literature uses and the comparison shows
whether the extra robustness is real here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.embedding_cache import DEFAULT_CACHE_DIR, cache_key, load
from tools.retrieval_research import l2_normalize


def split_folds(labels: list[str], images_per_fold: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split image indices into two folds that both contain every identity.

    Splitting by identity would create folds with disjoint label sets, which makes the measurement
    meaningless: a pseudo-label can only be right by luck when the true identity is absent from the
    labelled fold. Splitting *within* each identity keeps the label set identical across folds.
    """
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    fold_a: list[int] = []
    fold_b: list[int] = []
    for identity in sorted(set(labels)):
        positions = np.flatnonzero(labels_array == identity)
        rng.shuffle(positions)
        # Interleave rather than cut: with ~24 images per identity this keeps the two folds
        # balanced even when a count is odd.
        fold_a.extend(int(p) for p in positions[::2])
        fold_b.extend(int(p) for p in positions[1::2])
    del images_per_fold  # kept in the signature for callers that later add a cap
    return np.asarray(sorted(fold_a)), np.asarray(sorted(fold_b))


def pseudo_label(
    query: np.ndarray, reference: np.ndarray, reference_labels: list[str], k: int
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Top-k consensus labels for every query row.

    Returns the voted label, the vote share of the winner, and the raw top-1 similarity. The vote
    share is the confidence a threshold would select on.
    """
    similarity = query @ reference.T
    k = int(min(k, similarity.shape[1]))
    top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    ordered = np.take_along_axis(similarity, top, axis=1)
    top = np.take_along_axis(top, np.argsort(-ordered, axis=1, kind="stable"), axis=1)

    labels_array = np.asarray(reference_labels)
    voted: list[str] = []
    share = np.empty(query.shape[0], dtype=np.float32)
    for row in range(query.shape[0]):
        neighbours = labels_array[top[row]]
        counts = Counter(neighbours.tolist())
        winner, count = counts.most_common(1)[0]
        voted.append(str(winner))
        share[row] = count / float(k)
    top1_similarity = np.take_along_axis(similarity, top[:, :1], axis=1)[:, 0].astype(np.float32)
    return voted, share, top1_similarity


def coverage_curve(
    confidence: np.ndarray,
    correct: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    """Purity and coverage as the confidence threshold rises."""
    rows = []
    for threshold in thresholds:
        selected = confidence >= threshold
        if not selected.any():
            rows.append({"threshold": threshold, "coverage": 0.0, "purity": None, "images": 0})
            continue
        rows.append(
            {
                "threshold": threshold,
                "coverage": round(float(selected.mean()), 4),
                "purity": round(float(correct[selected].mean()), 4),
                "images": int(selected.sum()),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pseudo-label purity and coverage on the gallery")
    parser.add_argument("--checkpoint", default="artifacts/train/dinov2s-arcface/best.pt")
    parser.add_argument("--manifest", default="data/manifests/cat_individuals_manifest.jsonl")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--k", type=int, default=5, help="consensus neighbourhood size")
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

    # Both halves of the protocol describe the same 12 644-image corpus, so pool them: the folds
    # should cover every image, not just the ones that happened to be probes.
    vectors = np.concatenate([payload["query"], payload["gallery"]], axis=0)
    labels = list(payload["query_labels"]) + list(payload["gallery_labels"])
    vectors = l2_normalize(vectors)
    print(f"corpus: {vectors.shape[0]} images, {len(set(labels))} identities")

    fold_a, fold_b = split_folds(labels, 0, args.seed)
    print(
        f"fold A: {len(fold_a)} images   fold B: {len(fold_b)} images "
        f"(both hold all {len(set(labels))} identities)"
    )

    labels_array = np.asarray(labels)
    report: dict = {
        "k": args.k,
        "corpus": int(vectors.shape[0]),
        "identities": len(set(labels)),
        "seed": args.seed,
        "directions": {},
    }

    thresholds = [0.0, 0.4, 0.6, 0.8, 0.9, 1.0]
    for name, source, target in (
        ("A labelled -> B pseudo", fold_a, fold_b),
        ("B labelled -> A pseudo", fold_b, fold_a),
    ):
        # The returned top-1 similarity is not used here: the vote share is the confidence a
        # threshold selects on, and purity is measured against the true labels of this fold.
        voted, share, _top1_similarity = pseudo_label(
            vectors[target], vectors[source], labels_array[source].tolist(), args.k
        )
        truth = labels_array[target]
        voted_correct = np.asarray(voted) == truth
        top1_label = labels_array[source][np.argmax(vectors[target] @ vectors[source].T, axis=1)]
        top1_correct = top1_label == truth

        print()
        print(f"--- {name} ({len(target)} images) ---")
        print(f"  top-1 nearest neighbour   purity = {top1_correct.mean():.4f}")
        print(f"  top-{args.k} consensus vote    purity = {voted_correct.mean():.4f}")
        print("  threshold   coverage   purity   images")
        curve = coverage_curve(share, voted_correct, thresholds)
        for row in curve:
            purity = "n/a" if row["purity"] is None else f"{row['purity']:.4f}"
            print(f"  >= {row['threshold']:<9} {row['coverage']:<10.4f} {purity:<8} {row['images']}")

        report["directions"][name] = {
            "images": len(target),
            "top1_purity": round(float(top1_correct.mean()), 4),
            "vote_purity": round(float(voted_correct.mean()), 4),
            "curve": curve,
        }

    # The two directions are independent estimates of the same quantity; agreement between them is
    # the check that a single fold did not get lucky.
    purities = [report["directions"][key]["vote_purity"] for key in report["directions"]]
    print()
    print(f"vote purity across both directions: {purities}  (spread {max(purities) - min(purities):.4f})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
