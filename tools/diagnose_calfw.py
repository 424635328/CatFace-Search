"""Diagnose the CALFW verification protocol.

A near-chance AUC on a benchmark that should be easy is either a real property of the
data or a defect in the harness. This script separates the two by measuring the task with
progressively stronger references:

1. **Label statistics** — class balance, and whether the export preserved row order.
2. **Image statistics** — resolution and byte-level identity of the two halves.
3. **Trivial baselines** — raw-pixel cosine and a colour histogram. These establish the
   floor: anything the network scores no better than is not evidence of network quality.
4. **Sample pairs** — written to disk for visual inspection.

Usage::

    python -m tools.diagnose_calfw --parquet data/raw/calfw/calfw.parquet --pairs data/calfw/pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.eval.metrics import evaluate_verification, pairwise_cosine


def label_statistics(parquet_path: Path) -> dict:
    """Class balance and the position of each class, read straight from the parquet."""
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(parquet_path)
    targets: list[int] = []
    for batch in handle.iter_batches(batch_size=1024, columns=["target"]):
        targets.extend(int(v) for v in batch.column(0).to_pylist())
    labels = np.array(targets)
    first_change = int(np.argmax(labels != labels[0])) if labels.size else 0
    return {
        "rows": int(labels.size),
        "class_counts": {str(k): int(v) for k, v in sorted(Counter(targets).items())},
        "positive_ratio": float(labels.mean()) if labels.size else 0.0,
        "positive_indices_head": np.where(labels == 1)[0][:5].tolist(),
        "negative_indices_head": np.where(labels == 0)[0][:5].tolist(),
        "first_label_change_at": first_change,
        "looks_sorted": bool(np.all(np.diff(labels) >= 0)) if labels.size else None,
    }


def image_statistics(parquet_path: Path, limit: int = 300) -> dict:
    """Resolution, mode and duplicate structure of the images in the pair file."""
    import pyarrow.parquet as pq
    from PIL import Image

    handle = pq.ParquetFile(parquet_path)
    sizes: list[tuple[int, int]] = []
    modes: Counter = Counter()
    digests: list[str] = []
    looked = 0
    for batch in handle.iter_batches(batch_size=64, columns=["image1", "image2"]):
        images = batch.to_pylist()
        for row in images:
            for key in ("image1", "image2"):
                payload = row[key].get("bytes") if isinstance(row[key], dict) else row[key]
                if not payload:
                    continue
                digest = hashlib.sha1(payload).hexdigest()
                digests.append(digest)
                with Image.open(io.BytesIO(payload)) as image:
                    sizes.append(image.size)
                    modes[image.mode] += 1
                looked += 1
            if looked >= limit:
                break
        if looked >= limit:
            break

    array = np.array(sizes)
    unique = len(set(digests))
    return {
        "images_inspected": looked,
        "width_range": [int(array[:, 0].min()), int(array[:, 0].max())] if array.size else None,
        "height_range": [int(array[:, 1].min()), int(array[:, 1].max())] if array.size else None,
        "all_square": bool(array.size and np.all(array[:, 0] == array[:, 1])),
        "modes": dict(modes),
        "unique_images": unique,
        "duplicate_image_bytes": looked - unique,
        "duplicate_rate": round((looked - unique) / looked, 4) if looked else None,
    }


def raw_pixel_scores(csv_path: Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cosine similarity between downsampled raw pixels — the trivial baseline."""
    from PIL import Image

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        rows = rows[:limit]

    left, right, labels = [], [], []
    for row in rows:
        with Image.open(row["path_a"]) as image:
            a = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32).ravel()
        with Image.open(row["path_b"]) as image:
            b = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32).ravel()
        left.append(a)
        right.append(b)
        labels.append(int(row["label"]))
    return (
        np.vstack(left),
        np.vstack(right),
        np.array(labels),
        pairwise_cosine(np.vstack(left), np.vstack(right)),
    )


def colour_histogram_scores(csv_path: Path, bins: int = 8, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Histogram-intersection baseline: one score per pair, plus the labels.

    A deliberately weak reference that can only see colour composition. A network that
    fails to beat it is not using shape information at all, which makes the failure a
    protocol problem rather than a question of model quality.
    """
    from PIL import Image

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        rows = rows[:limit]

    def histogram(path: str) -> np.ndarray:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32)
        # pixels: (32, 32, 3) -> per-channel counts
        out = np.zeros(bins * 3, dtype=np.float32)
        for channel in range(3):
            values = np.clip(pixels[:, :, channel] / 255.0 * (bins - 1), 0, bins - 1)
            out[channel * bins : (channel + 1) * bins] = np.bincount(
                values.astype(np.int32).ravel(), minlength=bins
            )
        return out / max(out.sum(), 1.0)

    scores, labels = [], []
    for row in rows:
        scores.append(float(np.minimum(histogram(row["path_a"]), histogram(row["path_b"])).sum()))
        labels.append(int(row["label"]))
    return np.array(scores, dtype=np.float32), np.array(labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose the CALFW protocol")
    parser.add_argument("--parquet", default="data/raw/calfw/calfw.parquet")
    parser.add_argument("--pairs", default="data/calfw/pairs.csv")
    parser.add_argument("--csv-out", default="docs/diagnostics/calfw-diagnosis.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    parquet_path = Path(args.parquet)
    pairs_path = Path(args.pairs)
    if not parquet_path.is_file() or not pairs_path.is_file():
        print(f"missing input: {parquet_path} or {pairs_path}", file=sys.stderr)
        return 2

    report: dict = {}
    report["labels"] = label_statistics(parquet_path)
    report["images"] = image_statistics(parquet_path)

    labels_in_csv = []
    with pairs_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels_in_csv.append(int(row["label"]))
    report["export"] = {
        "csv_rows": len(labels_in_csv),
        "csv_positive_ratio": float(np.mean(labels_in_csv)),
        "matches_parquet_counts": (
            report["labels"]["class_counts"].get("1", 0) == int(np.sum(labels_in_csv))
        ),
    }

    left, right, labels, raw_scores = raw_pixel_scores(pairs_path, args.limit)
    raw_metrics = evaluate_verification(raw_scores, labels)
    report["baseline_raw_pixels"] = raw_metrics.to_dict()

    hist_scores, hist_labels = colour_histogram_scores(pairs_path, limit=args.limit)
    hist_metrics = evaluate_verification(hist_scores, hist_labels)
    report["baseline_colour_histogram"] = hist_metrics.to_dict()

    # How separable are the two classes by similarity alone?
    report["score_distribution"] = {
        "positive_mean": float(raw_scores[labels == 1].mean()),
        "negative_mean": float(raw_scores[labels == 0].mean()),
        "positive_std": float(raw_scores[labels == 1].std()),
        "negative_std": float(raw_scores[labels == 0].std()),
    }

    # The decisive check: are positive pairs visually *identical*? If same-identity pairs
    # are near-duplicate crops, any reasonable descriptor scores ~1.0 and the benchmark
    # cannot be failing for model reasons.
    near_identical = float((raw_scores[labels == 1] > 0.99).mean())
    report["near_identical_positive_rate"] = near_identical

    output = Path(args.csv_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nwritten to {output}")

    # Actionable verdict, stated explicitly rather than left to the reader.
    print("\n--- verdict ---")
    if report["labels"]["positive_ratio"] not in (0.4, 0.5, 0.6):
        print(f"class balance is {report['labels']['positive_ratio']:.3f} — check the export")
    if near_identical < 0.05:
        print(
            "positive pairs are NOT near-duplicates, so the task is genuinely hard for "
            "the crop/alignment being used; the low AUC is a property of the protocol"
        )
    else:
        print("positive pairs contain near-duplicates; scores should be near 1.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
