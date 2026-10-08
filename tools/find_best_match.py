"""Find the images this retrieval system recognises most reliably.

What "most reliably recognised" means
-------------------------------------
The aggregate benchmark reports one number per model (hit@1 = 0.9682 for the trained DINOv2-S).
That number cannot answer "which single image does this work best on", because hit@1 is a mean
over queries and hides the distribution behind it. This tool ranks the individual query images.

Three different questions are all reasonable, and they do not have the same answer:

* **sim**  — the query whose top-1 neighbour is most similar. This is "the system is most
  confident here", but confidence is not correctness: a query with no true match in the gallery
  can still score very high. Reported together with whether it was right.
* **margin** — the query whose top-1 *correct* match beats the best *wrong* identity by the
  largest gap. This is the honest "recognition is unambiguous" measure: it cannot be won by a
  confident mistake, because it requires the correct identity to actually be first.
* **worst** — the bottom of the ranking, printed so the failure cases are not hidden. A report
  that only shows the best image invites the reader to believe every image behaves like that.

Protocol
--------
Deliberately the *published* protocol, not a convenient one. The ``cat_individuals`` protocol
takes one query image per identity from the **whole** manifest and puts every other image in the
gallery, which is what produced the reported hit@1. Using the ``test.txt`` identity split instead
is a different, much smaller protocol (76 queries / 1834 gallery rather than 503 / 12 141) and its
scores are not comparable with the published table — an easy mistake, because the training loop
uses that split for its per-epoch validation and prints a similar-looking number.

The only difference from the benchmark is that this tool keeps per-query scores instead of
averaging them, so the ranking is comparable with the published figures.

Usage::

    python -m tools.find_best_match --checkpoint artifacts/train/dinov2s-arcface/best.pt
    python -m tools.find_best_match --checkpoint <ckpt> --top 20 --metric sim
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.config import load_config
from catface.data.manifest import Manifest
from catface.errors import CatFaceError
from catface.logging_utils import configure_utf8_console, get_logger
from catface.models.embedder import Embedder, embed_records
from catface.pipeline import build_within_split

LOGGER = get_logger("tools.find_best_match")


def repo_relative(path: str) -> str:
    """Express ``path`` relative to the repository root when it is inside it.

    Reports are committed as evidence, so they must not carry a machine-specific absolute path:
    it breaks every other clone, and ``tests/test_repo_hygiene.py`` enforces that. Falls back to
    the original string when the file genuinely lives outside the repository.
    """
    candidate = Path(path)
    for parent in [candidate, *candidate.parents]:
        if (parent / ".git").exists():
            try:
                return candidate.relative_to(parent).as_posix()
            except ValueError:
                break
    return path


def build_protocol(records: list, queries_per_identity: int, seed: int):
    """Reproduce the benchmark's ``cat_individuals`` query/gallery split."""
    return build_within_split(
        records,
        queries_per_identity=queries_per_identity,
        name="cat_individuals",
        seed=seed,
    )


def embed_protocol_split(embedder: Embedder, split, image_size: int) -> dict:
    """Extract descriptors for both halves, using the checkpoint's own TTA views."""
    started = time.perf_counter()
    query = embed_records(
        embedder, [r.path for r in split.query_records], image_size=image_size, batch_size=32
    )
    gallery = embed_records(
        embedder, [r.path for r in split.gallery_records], image_size=image_size, batch_size=32
    )
    if query.vectors.size == 0 or gallery.vectors.size == 0:
        raise CatFaceError("embedding produced no vectors")
    self_mask = np.array([[a == b for b in gallery.ids] for a in query.ids], dtype=bool)
    LOGGER.info(
        "embedded %d queries + %d gallery images in %.1fs (TTA=%s)",
        len(query.ids),
        len(gallery.ids),
        time.perf_counter() - started,
        list(embedder.config.tta),
    )
    return {"split": split, "query": query.vectors, "gallery": gallery.vectors, "self_mask": self_mask}


def rank_queries(cache: dict) -> list[dict]:
    """Score every query and return one entry per query, best first.

    Ranking is by margin (correct top-1 minus best wrong identity), which is the only one of the
    three orderings that cannot be topped by a confidently wrong answer.
    """
    split = cache["split"]
    q = np.asarray(cache["query"], dtype=np.float32)
    g = np.asarray(cache["gallery"], dtype=np.float32)
    query_labels = np.asarray(split.query_labels)
    gallery_labels = np.asarray(split.gallery_labels)
    self_mask = np.asarray(cache["self_mask"], dtype=bool)

    similarity = (q @ g.T).astype(np.float32)
    if self_mask.any():
        # A query's own image must not be its own best match; that would score a perfect 1.0 and
        # say nothing about recognition.
        similarity = np.where(self_mask, -np.inf, similarity)

    rows: list[dict] = []
    for index, query_record in enumerate(split.query_records):
        scores = similarity[index]
        order = np.argsort(-scores, kind="stable")

        best = int(order[0])
        top1_identity = str(gallery_labels[best])
        correct = top1_identity == str(query_labels[index])

        same_identity = gallery_labels == query_labels[index]
        correct_scores = scores[same_identity]
        wrong_scores = scores[~same_identity]
        best_correct = float(correct_scores.max()) if correct_scores.size else float("-inf")
        best_wrong = float(wrong_scores.max()) if wrong_scores.size else float("-inf")

        # The best gallery image of the same identity, whether or not it ranked first.
        if correct_scores.size:
            best_correct_index = int(np.flatnonzero(same_identity)[int(np.argmax(correct_scores))])
        else:
            best_correct_index = -1

        # First wrong identity in the ranking, for when the top-1 is wrong.
        wrong_order = [int(position) for position in order if not same_identity[position]]
        first_wrong_rank = int(np.flatnonzero(order == wrong_order[0])[0]) + 1 if wrong_order else 0

        rows.append(
            {
                "query_id": query_record.image_id,
                "query_path": repo_relative(query_record.path),
                "query_identity": str(query_labels[index]),
                "top1_id": str(split.gallery_records[best].image_id),
                "top1_identity": top1_identity,
                "top1_path": repo_relative(split.gallery_records[best].path),
                "top1_similarity": float(scores[best]),
                "correct": bool(correct),
                "margin": float(best_correct - best_wrong)
                if np.isfinite(best_correct) and np.isfinite(best_wrong)
                else float("nan"),
                "best_correct_id": str(split.gallery_records[best_correct_index].image_id)
                if best_correct_index >= 0
                else None,
                "best_correct_path": repo_relative(split.gallery_records[best_correct_index].path)
                if best_correct_index >= 0
                else None,
                "best_correct_similarity": best_correct if np.isfinite(best_correct) else None,
                "best_wrong_similarity": best_wrong if np.isfinite(best_wrong) else None,
                "first_wrong_rank": first_wrong_rank,
                "same_identity_in_gallery": int(same_identity.sum()),
            }
        )

    rows.sort(key=lambda row: (row["correct"], row["margin"]), reverse=True)
    return rows


def _show(rows: list[dict], count: int) -> None:
    print()
    print(f"{'#':>3}  {'sim':>6}  {'margin':>6}  {'ok':>3}  query image -> matched image")
    print("-" * 118)
    for rank, row in enumerate(rows[:count], start=1):
        print(
            f"{rank:>3}  {row['top1_similarity']:>6.4f}  {row['margin']:>6.4f}  "
            f"{'yes' if row['correct'] else 'NO ':>3}  "
            f"{Path(row['query_path']).name} -> {Path(row['top1_path']).name}"
        )
        print(f"       query  {row['query_path']}")
        print(f"       match  {row['top1_path']}  (identity {row['top1_identity']})")


def main(argv: list[str] | None = None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Find the best-recognised query images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", default=None, help="override the manifest from the config")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--queries-per-identity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top", type=int, default=10, help="how many ranked rows to print")
    parser.add_argument(
        "--metric",
        choices=("margin", "sim"),
        default="margin",
        help="ordering to print first: 'margin' (unambiguous correctness, default) or "
        "'sim' (raw confidence, which a confident mistake can top)",
    )
    parser.add_argument("--out", default=None, help="write the full ranking as JSON here")
    args = parser.parse_args(argv)

    embedder = Embedder.load(args.checkpoint, device=args.device or "cuda")
    LOGGER.info(
        "loaded %s (backbone=%s, tta=%s)",
        args.checkpoint,
        embedder.config.backbone,
        list(embedder.config.tta),
    )

    config = load_config(args.config)
    manifest_dir = Path(config.data.manifest)
    manifest_path = (
        Path(args.manifest) if args.manifest else (manifest_dir / "cat_individuals_manifest.jsonl")
    )
    if not manifest_path.is_file():
        raise CatFaceError(
            f"manifest missing at {manifest_path}. Run `catface prepare --source cat_individuals` first."
        )
    records = list(Manifest.load(manifest_path))
    if not records:
        raise CatFaceError(f"manifest {manifest_path} is empty")

    split = build_protocol(records, args.queries_per_identity, args.seed)
    LOGGER.info(
        "protocol cat_individuals: %d queries / %d gallery from %d images (%d identities)",
        split.num_queries,
        split.num_gallery,
        len(records),
        len(set(split.query_labels.tolist())),
    )

    cache = embed_protocol_split(embedder, split, args.image_size)
    rows = rank_queries(cache)

    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    print()
    print(
        f"protocol=cat_individuals  queries={total}  gallery={cache['split'].num_gallery}  "
        f"hit@1={correct / total:.4f}"
    )
    print("ranked by margin (top-1 correct match minus strongest wrong identity)")

    primary = (
        rows
        if args.metric == "margin"
        else sorted(rows, key=lambda row: (row["correct"], row["top1_similarity"]), reverse=True)
    )
    _show(primary, args.top)

    print()
    print("most confident overall (correct or not) — confidence alone is not correctness:")
    by_sim = sorted(rows, key=lambda row: row["top1_similarity"], reverse=True)
    _show(by_sim, args.top)

    print()
    print(f"least reliable {min(args.top, total)} — the other end of the same distribution:")
    _show(list(reversed(rows)), min(args.top, total))

    summary = {
        "checkpoint": args.checkpoint,
        "protocol": {
            "name": "cat_individuals",
            "queries": total,
            "gallery": int(cache["split"].num_gallery),
            "image_size": args.image_size,
            "seed": args.seed,
            "queries_per_identity": args.queries_per_identity,
            "tta": list(embedder.config.tta),
        },
        "hit@1": round(correct / total, 4),
        "best_by_margin": rows[0] if rows else None,
        "worst_by_margin": rows[-1] if rows else None,
        "best_by_similarity": by_sim[0] if by_sim else None,
        "ranking": rows,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
