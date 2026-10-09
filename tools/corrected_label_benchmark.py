"""The benchmark under a corrected label set.

Two numbers are worth reporting for a corpus with a known labelling defect, and they answer different
questions:

* **as published** — scored against the labels the corpus ships. This is the number comparable with
  other work and with any future run on the same data.
* **corrected** — scored after merging the identity labels that were verified to be the same cat.
  This estimates what the model would reach if the labels were right, which is the number that says
  how good the descriptor actually is.

Reporting only the first understates the model on 5 of 503 queries that cannot be answered by any
scorer. Reporting only the second would be claiming credit for a data defect. So both are computed
from one similarity matrix, and the difference is attributed query by query.

The merge is not a rescoring trick: it is derived from pixel-verified same-photograph pairs
(``docs/diagnostics/cross-identity-duplicates.json``), and every query whose outcome changes is
listed so the change can be inspected rather than trusted.
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


def merge_labels(labels: list[str], union_find: dict[str, str]) -> list[str]:
    return [union_find.get(label, label) for label in labels]


def build_merge_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Resolve label collisions into a single representative per connected component.

    Union-find rather than pairwise renaming, because collisions are transitive in principle: if
    A shares a photograph with B and B with C, all three are one cat and iterating pairwise rules
    would produce a different answer depending on order.
    """
    parent: dict[str, str] = {}

    def find(name: str) -> str:
        parent.setdefault(name, name)
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for left, right in pairs:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            # Keep the lexicographically smaller label as representative so the result does not
            # depend on the order the pairs happen to appear in.
            low, high = sorted((root_left, root_right))
            parent[high] = low

    return {name: find(name) for name in list(parent)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark under a corrected label set")
    parser.add_argument("--checkpoint", default="artifacts/train/dinov2s-arcface/best.pt")
    parser.add_argument("--manifest", default="data/manifests/cat_individuals_manifest.jsonl")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--collisions", default="docs/diagnostics/cross-identity-duplicates.json")
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

    collisions_path = Path(args.collisions)
    if not collisions_path.is_file():
        print(f"collision evidence not found at {collisions_path}; run tools.find_duplicates first")
        return 1
    findings = json.loads(collisions_path.read_text(encoding="utf-8")).get("findings", [])

    # Only pixel-verified pairs may merge labels. An embedding-similarity shortlist is not evidence
    # that two files are the same photograph, and merging on it would be laundering a guess into a
    # corrected metric.
    verified = [(row["identity_a"], row["identity_b"]) for row in findings if row.get("same_photograph")]
    merge = build_merge_map(verified)
    changed = {key_name: value for key_name, value in merge.items() if key_name != value}
    print(f"verified same-photograph pairs : {len(verified)}")
    print(f"labels merged away             : {len(changed)}")
    for name, representative in sorted(changed.items()):
        print(f"   {name} -> {representative}")

    query = l2_normalize(payload["query"])
    gallery = l2_normalize(payload["gallery"])
    similarity = (query @ gallery.T).astype(np.float32)

    original_query, original_gallery = payload["query_labels"], payload["gallery_labels"]
    fixed_query = merge_labels(original_query, merge)
    fixed_gallery = merge_labels(original_gallery, merge)

    as_published = retrieval_metrics(similarity, original_query, original_gallery)
    corrected = retrieval_metrics(similarity, fixed_query, fixed_gallery)

    def top1(labels_query, labels_gallery) -> np.ndarray:
        order = np.argmax(similarity, axis=1)
        return np.asarray(labels_gallery)[order] == np.asarray(labels_query)

    hit_before = top1(original_query, original_gallery)
    hit_after = top1(fixed_query, fixed_gallery)

    print()
    print(f"{'metric':<10}{'as published':>14}{'corrected':>12}{'delta':>10}")
    for metric in ("hit@1", "hit@5", "mINP", "mAP", "mRR"):
        before, after = as_published[metric], corrected[metric]
        print(f"{metric:<10}{before:>14.4f}{after:>12.4f}{after - before:>+10.4f}")

    flipped = np.flatnonzero(hit_before != hit_after)
    print()
    print(f"queries whose top-1 changed  : {len(flipped)}")
    for row in flipped:
        print(
            f"   {payload['query_ids'][row]:<44} {original_query[row]:<10} -> "
            f"{'correct' if hit_after[row] else 'wrong'} "
            f"(predicted {fixed_gallery[int(np.argmax(similarity[row]))]})"
        )

    report = {
        "checkpoint": args.checkpoint,
        "verified_pairs": len(verified),
        "merged_labels": changed,
        "as_published": as_published,
        "corrected": corrected,
        "delta": {
            metric: round(corrected[metric] - as_published[metric], 6)
            for metric in ("hit@1", "hit@5", "mINP", "mAP", "mRR")
        },
        "queries_changed": [
            {
                "query_id": payload["query_ids"][row],
                "label_as_published": original_query[row],
                "label_corrected": fixed_query[row],
                "correct_as_published": bool(hit_before[row]),
                "correct_corrected": bool(hit_after[row]),
            }
            for row in flipped.tolist()
        ],
        "unit_of_measure": 1.0 / len(original_query),
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten to {out_path}")
    print()
    print(
        f"report both numbers: {as_published['hit@1']:.4f} as published, "
        f"{corrected['hit@1']:.4f} corrected. One query is {report['unit_of_measure']:.4f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
