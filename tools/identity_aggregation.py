"""Is scoring the *identity* better than scoring the single best image?

The observation that motivated this
----------------------------------
All 16 failures of the direct cosine ranking have a **negative but tiny** margin: median -0.0569,
largest -0.0114. Not one is a confident mistake. That is the signature of a scoring rule that
throws away evidence, not of a descriptor that cannot represent the difference: the system compares
a probe against single gallery *images* when the label it must output is an *identity* averaging
about 24 gallery images.

Two aggregate scores are compared against the single-image baseline:

* ``max`` — the identity score is its single most similar gallery image. This is what a top-1
  ranking already implements, so it is the control and must reproduce hit@1 = 0.9682.
* ``topk_mean`` / ``sum`` — the identity score pools its ``k`` most similar gallery images. Under a
  Gaussian noise model of image-level similarity, averaging ``k`` samples of the same identity
  reduces the variance of the identity score by ``k`` while leaving the mean unbiased, so a
  near-tie decided by one unlucky sample can flip. The cost is a bias against identities with fewer
  gallery images, which is why ``k`` is a parameter and not a constant.

The ceiling is reported explicitly: for each aggregation, the best achievable hit@1 over *all*
values of k is computed, so a null result is distinguishable from an unlucky choice of k.
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


def identity_scores(similarity: np.ndarray, gallery_labels: list[str],
                    identities: list[str], mode: str, k: int = 3) -> np.ndarray:
    """``(Q, I)`` scores, one column per identity.

    ``similarity`` is ``(Q, G)`` cosine. Masking rather than looping keeps this a handful of
    vectorised ops, which matters because the whole point is to sweep k cheaply.
    """
    labels = np.asarray(gallery_labels)
    columns = {identity: index for index, identity in enumerate(identities)}
    mask = np.zeros((len(identities), similarity.shape[1]), dtype=bool)
    for identity, index in columns.items():
        mask[index] = labels == identity

    scores = np.empty((similarity.shape[0], len(identities)), dtype=np.float32)
    for index in range(len(identities)):
        block = similarity[:, mask[index]]
        if mode == "max":
            scores[:, index] = block.max(axis=1)
        elif mode == "mean":
            scores[:, index] = block.mean(axis=1)
        else:
            kk = min(k, block.shape[1])
            partitioned = np.partition(block, -kk, axis=1)[:, -kk:]
            scores[:, index] = partitioned.mean(axis=1) if mode == "topk_mean" else partitioned.sum(axis=1)
    return scores


def identity_hit(scores: np.ndarray, query_labels: list[str], identities: list[str]) -> float:
    """Share of queries whose highest-scoring identity is the correct one."""
    predicted = np.asarray(identities)[np.argmax(scores, axis=1)]
    return float((predicted == np.asarray(query_labels)).mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Identity-level vs image-level scoring")
    parser.add_argument("--checkpoint", default="artifacts/train/dinov2s-arcface/best.pt")
    parser.add_argument("--manifest", default="data/manifests/cat_individuals_manifest.jsonl")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    key = cache_key(args.checkpoint, args.manifest, protocol="cat_individuals",
                    image_size=args.image_size, tta=("identity", "hflip"),
                    extra={"queries_per_identity": 1, "seed": 1337})
    payload = load(args.cache_dir, key)
    if payload is None:
        print(f"cache miss for {key}; run tools.retrieval_research first")
        return 1

    query = l2_normalize(payload["query"])
    gallery = l2_normalize(payload["gallery"])
    similarity = (query @ gallery.T).astype(np.float32)
    query_labels = payload["query_labels"]
    gallery_labels = payload["gallery_labels"]
    identities = sorted(set(gallery_labels))

    print(f"queries {len(query_labels)}  gallery {len(gallery_labels)}  identities {len(identities)}")
    baseline = retrieval_metrics(similarity, query_labels, gallery_labels)
    print(f"image-level baseline                hit@1={baseline['hit@1']:.4f} "
          f"mINP={baseline['mINP']:.4f}")
    print()

    rows = []
    for mode in ("max", "topk_mean", "sum", "mean"):
        ks = [3] if mode in ("max", "mean") else list(range(1, args.k_max + 1))
        for k in ks:
            scores = identity_scores(similarity, gallery_labels, identities, mode, k)
            hit = identity_hit(scores, query_labels, identities)
            rows.append({"mode": mode, "k": k, "identity_hit@1": hit})
            if mode in ("max", "mean") or k in (1, 3, 5, 10, 20, args.k_max):
                print(f"  {mode:<10} k={k:<3} identity hit@1 = {hit:.4f}")

    best = max(rows, key=lambda row: row["identity_hit@1"])
    print()
    print(f"best identity-level: {best['mode']} k={best['k']} -> {best['identity_hit@1']:.4f}")
    print(f"image-level baseline                 -> {baseline['hit@1']:.4f}")
    print(f"delta                                -> {best['identity_hit@1'] - baseline['hit@1']:+.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "baseline_image_level": baseline,
            "sweep": rows,
            "best": best,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
