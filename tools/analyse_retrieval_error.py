"""Where is the remaining error, exactly?

Two neighbourhood-graph re-rankers both reproduced ``hit@1 = 0.9682`` exactly, which is either a
genuine ceiling or two bad implementations. This script separates those explanations by measuring
the query-level decomposition of the error, which no aggregate metric shows:

* **rank of the best correct match.** If the correct identity is at rank 2-5 for many queries, then
  re-ranking has headroom and the ceiling claim is wrong. If the correct identity is ranked
  hundreds, the failure is representational and no re-ranking can recover it.
* **margin distribution.** A negative margin means the wrong identity is genuinely closer.
* **label collisions.** The 5 queries whose top-1 is their own photograph under a second label are
  not winnable by any scorer, by construction.
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


def analyse(similarity: np.ndarray, query_labels: list[str], gallery_labels: list[str],
            query_ids: list[str], gallery_ids: list[str],
            collision_pairs: set[frozenset] | None) -> dict:
    labels_query = np.asarray(query_labels)
    labels_gallery = np.asarray(gallery_labels)
    relevant = labels_query[:, None] == labels_gallery[None, :]

    # Rank of the best correct match per query (1 = top-1 already correct).
    masked = np.where(relevant, similarity, -np.inf)
    best_correct = masked.max(axis=1)
    best_wrong = np.where(relevant, -np.inf, similarity).max(axis=1)
    rank_of_best_correct = (similarity > best_correct[:, None]).sum(axis=1) + 1

    correct = best_correct >= best_wrong
    errors = np.flatnonzero(~correct)

    buckets = Counter()
    for rank in rank_of_best_correct:
        if rank == 1:
            buckets["rank 1 (correct)"] += 1
        elif rank <= 2:
            buckets["rank 2"] += 1
        elif rank <= 5:
            buckets["rank 3-5"] += 1
        elif rank <= 10:
            buckets["rank 6-10"] += 1
        elif rank <= 50:
            buckets["rank 11-50"] += 1
        else:
            buckets["rank >50"] += 1

    detail = []
    collision_count = 0
    for index in errors:
        top1 = int(np.argmax(similarity[index]))
        pair = frozenset((query_ids[index], gallery_ids[top1]))
        is_collision = bool(collision_pairs and pair in collision_pairs)
        collision_count += int(is_collision)
        detail.append({
            "query_id": query_ids[index],
            "truth": str(labels_query[index]),
            "predicted": str(labels_gallery[top1]),
            "rank_of_best_correct": int(rank_of_best_correct[index]),
            "best_correct_similarity": float(best_correct[index]),
            "best_wrong_similarity": float(best_wrong[index]),
            "margin": float(best_correct[index] - best_wrong[index]),
            "label_collision": is_collision,
        })

    margins_correct = (best_correct - best_wrong)[correct]
    return {
        "queries": int(similarity.shape[0]),
        "correct": int(correct.sum()),
        "errors": int((~correct).sum()),
        "errors_that_are_label_collisions": collision_count,
        "rank_of_best_correct_histogram": dict(buckets),
        "errors_detail": detail,
        "margin_correct": {
            "min": float(margins_correct.min()),
            "p05": float(np.percentile(margins_correct, 5)),
            "median": float(np.median(margins_correct)),
        },
        "margin_errors": {
            "max": float((best_correct - best_wrong)[~correct].max()),
            "median": float(np.median((best_correct - best_wrong)[~correct])),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decompose the residual retrieval error")
    parser.add_argument("--checkpoint", default="artifacts/train/dinov2s-arcface/best.pt")
    parser.add_argument("--manifest", default="data/manifests/cat_individuals_manifest.jsonl")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--collisions", default="docs/diagnostics/cross-identity-duplicates.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    key = cache_key(args.checkpoint, args.manifest, protocol="cat_individuals",
                    image_size=args.image_size, tta=("identity", "hflip"),
                    extra={"queries_per_identity": 1, "seed": 1337})
    payload = load(args.cache_dir, key)
    if payload is None:
        print(f"cache miss for {key}; run tools.retrieval_research first")
        return 1

    collision_pairs: set[frozenset] = set()
    collision_path = Path(args.collisions)
    if collision_path.is_file():
        data = json.loads(collision_path.read_text(encoding="utf-8"))
        for finding in data.get("findings", []):
            if finding.get("same_photograph"):
                collision_pairs.add(frozenset((finding["image_a"], finding["image_b"])))

    query = l2_normalize(payload["query"])
    gallery = l2_normalize(payload["gallery"])
    similarity = (query @ gallery.T).astype(np.float32)

    report = analyse(similarity, payload["query_labels"], payload["gallery_labels"],
                     payload["query_ids"], payload["gallery_ids"], collision_pairs)
    report["collision_pairs_available"] = len(collision_pairs)

    print(f"queries                          : {report['queries']}")
    print(f"correct                          : {report['correct']}")
    print(f"errors                           : {report['errors']}")
    print(f"  of which label collisions      : {report['errors_that_are_label_collisions']}")
    print(f"  genuinely wrong                : {report['errors'] - report['errors_that_are_label_collisions']}")
    print()
    print("rank of the best correct match (how much a re-ranker could recover):")
    for name in ("rank 1 (correct)", "rank 2", "rank 3-5", "rank 6-10", "rank 11-50", "rank >50"):
        if name in report["rank_of_best_correct_histogram"]:
            print(f"   {name:<20} {report['rank_of_best_correct_histogram'][name]}")
    print()
    print(f"margin on correct queries  median={report['margin_correct']['median']:.4f} "
          f"p05={report['margin_correct']['p05']:.4f}")
    print(f"margin on error queries    median={report['margin_errors']['median']:.4f} "
          f"max={report['margin_errors']['max']:.4f}")
    print()
    print("wrong queries:")
    for row in sorted(report["errors_detail"], key=lambda r: r["rank_of_best_correct"]):
        tag = " <-- LABEL COLLISION" if row["label_collision"] else ""
        print(f"   truth={row['truth']:<10} pred={row['predicted']:<10} "
              f"rank_of_best_correct={row['rank_of_best_correct']:<5} "
              f"margin={row['margin']:+.4f}{tag}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
