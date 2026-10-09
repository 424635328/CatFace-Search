"""Multi-scale descriptor fusion: does a second resolution carry independent information?

The argument
------------
A single-scale descriptor resamples the image once. Two resolutions see different effective
receptive fields and different aliasing, so their errors are partly independent. If the two
descriptors are ``u`` and ``v`` (unit norm, dimension ``D``), then

* **concatenation** ``[u ; v] / sqrt(2)`` has cosine ``(u.u' + v.v') / 2`` with another pair, i.e.
  the *arithmetic mean of the two similarities*. It cannot lose either scale's signal, and for two
  independent unbiased estimators of the same quantity it reduces the variance of the score by half.
* **averaging** ``(u + v) / ||u + v||`` is not the same operation: it sums the vectors and then
  rescales, which cancels components that disagree in sign and can destroy information the
  concatenation keeps. It is the weaker rule of the two in general, so both are measured rather
  than assumed.

Both are label-free and apply identically to query and gallery, so neither can leak.

The null hypothesis is that the second scale adds nothing and the fused score merely averages away
the better scale's advantage. A win here is a real improvement in the descriptor, not a scoring
trick, which is what the exhausted post-processing search of the earlier experiment implies is
needed.
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


def _key(checkpoint: str, manifest: str, image_size: int, views: tuple[str, ...]) -> str:
    return cache_key(checkpoint, manifest, protocol="cat_individuals", image_size=image_size,
                     tta=views, extra={"queries_per_identity": 1, "seed": 1337})


def load_scale(directory: str, checkpoint: str, manifest: str, image_size: int,
               views: tuple[str, ...]) -> dict | None:
    return load(directory, _key(checkpoint, manifest, image_size, views))


def fuse(a: np.ndarray, b: np.ndarray, mode: str, weight: float = 0.5) -> np.ndarray:
    """Combine two unit-norm descriptor blocks of identical shape."""
    a = l2_normalize(a)
    b = l2_normalize(b)
    if mode == "concat":
        return l2_normalize(np.concatenate([a * np.sqrt(weight), b * np.sqrt(1.0 - weight)], axis=1))
    if mode == "mean":
        return l2_normalize(weight * a + (1.0 - weight) * b)
    if mode == "a":
        return a
    if mode == "b":
        return b
    raise ValueError(f"unknown fusion mode {mode!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuse descriptors from two resolutions")
    parser.add_argument("--checkpoint", default="artifacts/train/dinov2s-arcface/best.pt")
    parser.add_argument("--manifest", default="data/manifests/cat_individuals_manifest.jsonl")
    parser.add_argument("--size-a", type=int, default=224)
    parser.add_argument("--size-b", type=int, default=288)
    parser.add_argument("--views", default="identity,hflip")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    views = tuple(name.strip() for name in args.views.split(",") if name.strip())
    first = load_scale(args.cache_dir, args.checkpoint, args.manifest, args.size_a, views)
    second = load_scale(args.cache_dir, args.checkpoint, args.manifest, args.size_b, views)
    if first is None or second is None:
        missing = []
        if first is None:
            missing.append(args.size_a)
        if second is None:
            missing.append(args.size_b)
        print(f"cache miss for size(s) {missing}; embed them first with tools.retrieval_research")
        return 1

    # The two runs must describe the same protocol, or a fused score is meaningless.
    # ``load`` normalises these to Python lists, but compare through ``list()`` so a future change
    # returning arrays cannot silently turn this into an elementwise comparison.
    for field in ("query_ids", "gallery_ids", "query_labels", "gallery_labels"):
        if list(first[field]) != list(second[field]):
            raise SystemExit(f"protocol mismatch on {field}: the two caches are not comparable")
    print(f"protocol verified identical: {len(first['query_ids'])} queries, "
          f"{len(first['gallery_ids'])} gallery")

    query_labels = first["query_labels"]
    gallery_labels = first["gallery_labels"]
    rows: list[dict] = []

    def evaluate(name: str, q: np.ndarray, g: np.ndarray) -> None:
        similarity = (l2_normalize(q) @ l2_normalize(g).T).astype(np.float32)
        metrics = retrieval_metrics(similarity, query_labels, gallery_labels)
        rows.append({"name": name, "dim": int(q.shape[1]), **metrics})
        print(f"  {name:<26} dim={q.shape[1]:<5} hit@1={metrics['hit@1']:.4f} "
              f"hit@5={metrics['hit@5']:.4f} mINP={metrics['mINP']:.4f} mAP={metrics['mAP']:.4f}")

    print()
    evaluate(f"scale {args.size_a} only", first["query"], first["gallery"])
    evaluate(f"scale {args.size_b} only", second["query"], second["gallery"])
    for mode in ("concat", "mean"):
        for weight in (0.3, 0.5, 0.7):
            evaluate(f"{mode} w={weight} ({args.size_a}/{args.size_b})",
                     fuse(first["query"], second["query"], mode, weight),
                     fuse(first["gallery"], second["gallery"], mode, weight))

    best = max(rows, key=lambda row: (row["hit@1"], row["mINP"]))
    baseline = rows[0]
    print()
    print(f"best: {best['name']} hit@1={best['hit@1']:.4f} mINP={best['mINP']:.4f}")
    print(f"vs {baseline['name']}: Δhit@1={best['hit@1'] - baseline['hit@1']:+.4f} "
          f"ΔmINP={best['mINP'] - baseline['mINP']:+.4f}")
    print(f"hit@1 resolution is 1/{len(query_labels)} = "
          f"{1.0 / len(query_labels):.4f} per query, so a delta below that is one query")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "checkpoint": args.checkpoint,
            "sizes": [args.size_a, args.size_b],
            "views": list(views),
            "queries": len(query_labels),
            "per_query_resolution": 1.0 / len(query_labels),
            "results": rows,
            "best": best,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
