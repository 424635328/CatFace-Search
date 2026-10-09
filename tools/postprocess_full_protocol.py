"""Do the selected post-processing transforms help or hurt on the FULL protocol?

The question
------------
``docs/diagnostics/postprocess-tuning.json`` selects ``pca + DBA + αQE`` on the **compact** protocol
(76 queries / 1 834 gallery) and reports ``mINP`` improving 0.6574 -> 0.7234 there. But mINP is not
comparable across gallery sizes — a bigger gallery means more relevant items per identity, which
pushes the *last* relevant item to a better rank and inflates mINP mechanically. So the compact
selection says nothing about the full benchmark, which is what gets published.

This evaluates the same candidate configurations on the full protocol (503 / 12 141) using the cached
descriptors, so the only thing that changes is the protocol. If whitening helps, the published
configuration stands. If it hurts, the published mINP is being degraded by a transform that was
chosen under a different measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catface.eval.benchmark import PostprocessConfig, apply_postprocessing
from tools.embedding_cache import DEFAULT_CACHE_DIR, cache_key, load
from tools.retrieval_research import retrieval_metrics

CANDIDATES = {
    "none": PostprocessConfig(),
    "dba": PostprocessConfig(dba=True, dba_k=3, dba_alpha=3.0),
    "aqe": PostprocessConfig(query_expansion="aqe", aqe_top_k=3, aqe_alpha=3.0),
    "pca+dba+aqe (published)": PostprocessConfig(
        whiten="pca",
        whiten_dim=0,
        dba=True,
        dba_k=3,
        dba_alpha=3.0,
        query_expansion="aqe",
        aqe_top_k=3,
        aqe_alpha=3.0,
    ),
    "pcaw256+dba+aqe": PostprocessConfig(
        whiten="pcaw",
        whiten_dim=256,
        dba=True,
        dba_k=3,
        dba_alpha=3.0,
        query_expansion="aqe",
        aqe_top_k=3,
        aqe_alpha=3.0,
    ),
    "pca128+dba+aqe": PostprocessConfig(
        whiten="pca",
        whiten_dim=128,
        dba=True,
        dba_k=3,
        dba_alpha=3.0,
        query_expansion="aqe",
        aqe_top_k=3,
        aqe_alpha=3.0,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-processing on the full protocol")
    parser.add_argument("--checkpoint", default="artifacts/train/dinov2s-arcface/best.pt")
    parser.add_argument("--manifest", default="data/manifests/cat_individuals_manifest.jsonl")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--split",
        default="test",
        choices=("test", "val"),
        help="the cache holds the protocol built over the whole corpus; selecting "
        "'val' restricts scoring to the val identities, which is what makes a "
        "selection legitimate rather than post hoc",
    )
    parser.add_argument("--splits-dir", default="data/manifests/cat_individuals_splits")
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

    query = np.asarray(payload["query"], dtype=np.float32)
    gallery = np.asarray(payload["gallery"], dtype=np.float32)
    query_labels = np.asarray(payload["query_labels"])
    gallery_labels = np.asarray(payload["gallery_labels"])

    # Restricting to one identity split is what turns this from a post-hoc comparison into a
    # selection procedure: the configuration is chosen on val identities and its effect is then
    # reported on test identities it never saw.
    #
    # The split files list *image ids* (1 901 of them for val), not identity names, so the identities
    # are resolved through the manifest. Treating the ids as identity names selected nothing, which
    # surfaced as a "0 of 503 kept" filter and NaN metrics rather than as an error.
    if args.split != "test":
        from catface.data.manifest import Manifest

        wanted_images = {
            line.strip()
            for line in (Path(args.splits_dir) / f"{args.split}.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        wanted = {
            record.identity for record in Manifest.load(args.manifest) if record.image_id in wanted_images
        }
        if not wanted:
            print(f"no identities resolved for split {args.split!r}; check --splits-dir")
            return 1
        keep_query = np.isin(query_labels, list(wanted))
        query = query[keep_query]
        # The gallery stays complete: entries from other splits are distractors, and removing them
        # would make the task easier than the benchmark's.
        print(
            f"scoring only the {args.split} identities ({len(wanted)} of them): "
            f"{int(keep_query.sum())} of {len(payload['query_labels'])} queries kept"
        )
        query_labels = query_labels[keep_query]

    query_labels = query_labels.tolist()
    gallery_labels = gallery_labels.tolist()
    print(f"full protocol: {query.shape[0]} queries / {gallery.shape[0]} gallery")
    print()

    rows = []
    header = f"{'configuration':<26}{'hit@1':>8}{'hit@5':>8}{'mINP':>9}{'mAP':>9}{'mRR':>9}"
    print(header)
    print("-" * len(header))
    for label, config in CANDIDATES.items():
        transformed_query, transformed_gallery, diagnostics = apply_postprocessing(query, gallery, config)
        similarity = (transformed_query @ transformed_gallery.T).astype(np.float32)
        metrics = retrieval_metrics(similarity, query_labels, gallery_labels)
        rows.append(
            {
                "configuration": label,
                "config": config.describe(),
                "descriptor_dim": int(diagnostics.get("query_dim", 0)),
                **metrics,
            }
        )
        print(
            f"{label:<26}{metrics['hit@1']:>8.4f}{metrics['hit@5']:>8.4f}"
            f"{metrics['mINP']:>9.4f}{metrics['mAP']:>9.4f}{metrics['mRR']:>9.4f}"
        )

    baseline = rows[0]
    published = next((r for r in rows if r["configuration"].startswith("pca+dba+aqe")), None)
    print()
    if published:
        print(f"published configuration vs no post-processing, on the {args.split} split:")
        for metric in ("hit@1", "mINP", "mAP", "mRR"):
            delta = published[metric] - baseline[metric]
            verdict = "better" if delta > 0 else ("worse" if delta < 0 else "same")
            print(
                f"   {metric:<7} {baseline[metric]:.4f} -> {published[metric]:.4f}  {delta:+.4f}  ({verdict})"
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "protocol": {
                        "queries": int(query.shape[0]),
                        "gallery": int(gallery.shape[0]),
                        "unit": 1.0 / query.shape[0],
                    },
                    "results": rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
