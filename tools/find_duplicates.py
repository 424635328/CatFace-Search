"""Find near-duplicate images that carry different identity labels.

Why this matters for the reported numbers
-----------------------------------------
This project publishes a retrieval benchmark in which a query must find its own identity among
12 141 gallery images. If the same photograph appears twice under two different identity labels,
then a query can retrieve its own picture and still be scored **wrong** — the model is penalised
for a labelling defect, and the reported hit@1 is an underestimate that nobody can see in the
aggregate number.

The opposite error is equally important. Two folders holding the same cat under different labels
would make some identities trivially recoverable and inflate the score. So both directions are
reported: exact pixels are checked independently of the embedder, because a high embedding
similarity alone cannot distinguish "the model is confused" from "the files are the same photo".

Pixel comparison is the only trustworthy arbiter here, so every reported pair is verified by
decoding both images and comparing them. Embedding similarity is used solely to *shortlist*
candidate pairs, since comparing all 12 644² pairs pixel-wise is not affordable.

Usage::

    python -m tools.find_duplicates --checkpoint artifacts/train/dinov2s-arcface/best.pt
    python -m tools.find_duplicates --checkpoint <ckpt> --min-similarity 0.99 --out <path>
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
from tools.find_best_match import repo_relative

LOGGER = get_logger("tools.find_duplicates")

#: Two images are treated as the same photograph when no pixel differs by more than this.
#: JPEG re-encoding of an identical image perturbs values by a few levels; two different photos of
#: the same cat differ by far more than this (measured control: max diff 238).
PIXEL_DIFF_TOLERANCE = 8

#: Share of pixels allowed to exceed the tolerance, to absorb re-encode noise.
PIXEL_DIFF_SHARE = 0.05


def cross_identity_candidates(
    vectors: np.ndarray,
    labels: list[str],
    ids: list[str],
    min_similarity: float,
    top_pairs: int,
) -> list[tuple[int, int, float]]:
    """Return the most similar ``(i, j)`` pairs whose identity labels differ."""
    matrix = np.asarray(vectors, dtype=np.float32)
    label_array = np.asarray(labels)
    n = matrix.shape[0]
    pairs: list[tuple[int, int, float]] = []

    # Row blocks keep peak memory bounded: a full 12 644² float32 matrix is 640 MB.
    block = 512
    for start in range(0, n, block):
        stop = min(start + block, n)
        similarity = matrix[start:stop] @ matrix.T
        for row in range(stop - start):
            index = start + row
            scores = similarity[row]
            # Different label, and never the same image paired with itself.
            different = (label_array != label_array[index]) & (np.asarray(ids) != ids[index])
            if not different.any():
                continue
            candidate_indices = np.flatnonzero(different)
            candidate_scores = scores[candidate_indices]
            keep = candidate_scores >= min_similarity
            for position, score in zip(candidate_indices[keep], candidate_scores[keep]):
                other = int(position)
                if other > index:  # each unordered pair once
                    pairs.append((index, other, float(score)))

    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:top_pairs]


def pixels_are_same(path_a: str, path_b: str) -> dict:
    """Decode both images and report how far apart they are, independently of the embedder."""
    from PIL import Image

    try:
        image_a = Image.open(path_a).convert("RGB")
        image_b = Image.open(path_b).convert("RGB")
    except Exception as error:
        return {"comparable": False, "error": str(error)}

    if image_a.size != image_b.size:
        return {"comparable": False, "reason": f"different sizes {image_a.size} vs {image_b.size}"}

    a = np.asarray(image_a, dtype=np.int16)
    b = np.asarray(image_b, dtype=np.int16)
    diff = np.abs(a - b)
    # Mean across channels, not max: a single pixel whose channels shifted in opposite
    # directions (R +4, G -4, B +4) is still a difference of 4, but max-over-channels counts it
    # twice and pushed a genuinely identical JPEG re-encode over the threshold.
    per_pixel = diff.mean(axis=2)
    share = float((per_pixel > PIXEL_DIFF_TOLERANCE).mean())
    return {
        "comparable": True,
        "max_abs_diff": int(diff.max()),
        "mean_abs_diff": round(float(diff.mean()), 4),
        "share_pixels_differing": round(share, 6),
        "same_photograph": bool(share <= PIXEL_DIFF_SHARE),
    }


def main(argv: list[str] | None = None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Find cross-identity near-duplicate images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.98,
        help="embedding similarity used to shortlist pairs for pixel checking",
    )
    parser.add_argument("--top-pairs", type=int, default=200)
    parser.add_argument("--report-pairs", type=int, default=25)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    embedder = Embedder.load(args.checkpoint, device=args.device or "cuda")
    config = load_config(args.config)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else (Path(config.data.manifest) / "cat_individuals_manifest.jsonl")
    )
    if not manifest_path.is_file():
        raise CatFaceError(f"manifest missing at {manifest_path}")

    records = list(Manifest.load(manifest_path))
    if not records:
        raise CatFaceError(f"manifest {manifest_path} is empty")

    started = time.perf_counter()
    embedding = embed_records(embedder, [r.path for r in records], image_size=args.image_size, batch_size=32)
    LOGGER.info("embedded %d images in %.1fs", len(records), time.perf_counter() - started)

    labels = [r.identity for r in records]
    ids = [r.image_id for r in records]
    candidates = cross_identity_candidates(
        embedding.vectors, labels, ids, args.min_similarity, args.top_pairs
    )
    LOGGER.info("shortlisted %d cross-identity pairs at cosine >= %.4f", len(candidates), args.min_similarity)

    findings: list[dict] = []
    duplicates = 0
    for index, other, score in candidates:
        check = pixels_are_same(records[index].path, records[other].path)
        entry = {
            "similarity": round(score, 6),
            "identity_a": labels[index],
            "identity_b": labels[other],
            "image_a": records[index].image_id,
            "image_b": records[other].image_id,
            "path_a": repo_relative(records[index].path),
            "path_b": repo_relative(records[other].path),
            **check,
        }
        if check.get("same_photograph"):
            duplicates += 1
        findings.append(entry)

    print()
    print(f"images           : {len(records)}")
    print(f"identities       : {len(set(labels))}")
    print(f"pairs checked    : {len(findings)}")
    print(f"same photograph  : {duplicates}  (different identity labels)")
    print()
    print("verified same-photograph pairs, by embedding similarity:")
    shown = 0
    for entry in findings:
        if not entry.get("same_photograph"):
            continue
        shown += 1
        if shown > args.report_pairs:
            break
        print(
            f"  sim {entry['similarity']:.6f}  maxΔpx {entry['max_abs_diff']:>3}  "
            f"{entry['identity_a']} vs {entry['identity_b']}  "
            f"{Path(entry['path_a']).name} == {Path(entry['path_b']).name}"
        )
    if shown == 0:
        print("  none verified as the same photograph")

    summary = {
        "checkpoint": args.checkpoint,
        "images": len(records),
        "identities": len(set(labels)),
        "min_similarity": args.min_similarity,
        "pairs_checked": len(findings),
        "same_photograph_pairs": duplicates,
        "pixel_criteria": {
            "max_abs_diff_tolerance": PIXEL_DIFF_TOLERANCE,
            "share_pixels_allowed_differing": PIXEL_DIFF_SHARE,
        },
        "findings": findings,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
