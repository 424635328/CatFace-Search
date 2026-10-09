"""Where should the whitening transform be fitted: the gallery, or the training identities?

The gap this fills
------------------
``PostprocessConfig`` already supports PCA and PCA-whitening, and the tuning sweep selected them on
the gallery. But the transform's *covariance* was always estimated from the evaluation gallery itself,
which is the only labelled data at test time — not the data the model was trained on. The training
split has 8 833 images across 352 identities whose labels were used for training, so its covariance is
a better estimate of the descriptor's structure than 12 141 unlabelled-at-fit-time gallery images.

Why whitening should matter here
--------------------------------
Metric learning shapes the space so that identities separate, but it does not remove the strong
anisotropy every DINOv2 descriptor inherits: a few directions carry most of the variance and dominate
cosine similarity while saying nothing about identity. Decorrelating and rescaling by the inverse
standard deviation removes exactly those directions, which is the same argument the project already
documents for gallery-fitted whitening — applied to a better covariance estimate.

The comparison is controlled: identical descriptors, identical protocol, and the only difference is
which split estimated the covariance. The dimension is swept because whitening discards directions,
and how many to keep is an empirical question.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catface.config import load_config
from catface.data.manifest import Manifest
from catface.errors import CatFaceError
from catface.logging_utils import configure_utf8_console, get_logger
from catface.models.embedder import Embedder, embed_records
from tools.embedding_cache import (
    DEFAULT_CACHE_DIR,
    cache_key,
    load,
    load_arrays,
    save_arrays,
)
from tools.retrieval_research import l2_normalize, retrieval_metrics

LOGGER = get_logger("tools.whitening_source")


def split_records(manifest_path: Path, splits_dir: Path, name: str) -> list:
    ids = {
        line.strip()
        for line in (splits_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return [r for r in Manifest.load(manifest_path) if r.image_id in ids]


def fit_whitening(vectors: np.ndarray, dim: int, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """PCA-whitening components: ``(components, scale)`` with ``W = components * scale``.

    Returns the projection matrix as two arrays so the transform can be applied with a single matmul
    and kept inspectable. ``dim=0`` means "keep every direction".
    """
    matrix = l2_normalize(np.asarray(vectors, dtype=np.float64))
    mean = matrix.mean(axis=0, keepdims=True)
    centred = matrix - mean
    # Economy SVD: the number of samples is far below the descriptor width here, so computing the
    # full 512x512 covariance would be both slower and less numerically stable.
    _u, singular, vt = np.linalg.svd(centred, full_matrices=False)
    keep = vt.shape[0] if dim <= 0 else min(int(dim), vt.shape[0])
    components = vt[:keep]
    variance = (singular[:keep] ** 2) / max(centred.shape[0] - 1, 1)
    scale = 1.0 / np.sqrt(np.maximum(variance, eps))
    return components, scale


def apply_whitening(
    vectors: np.ndarray, components: np.ndarray, scale: np.ndarray, mean: np.ndarray | None = None
) -> np.ndarray:
    matrix = l2_normalize(np.asarray(vectors, dtype=np.float64))
    if mean is not None:
        matrix = matrix - mean
    return l2_normalize((matrix @ components.T) * scale)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Compare whitening fitted on train vs on gallery")
    parser.add_argument("--checkpoint", default="artifacts/train/dinov2s-arcface/best.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dims", default="0,512,256,128,64", help="whitened dimensions to try; 0 keeps the descriptor width"
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    manifest_dir = Path(config.data.manifest)
    manifest_path = manifest_dir / "cat_individuals_manifest.jsonl"
    splits_dir = manifest_dir / "cat_individuals_splits"
    if not manifest_path.is_file():
        raise CatFaceError(f"manifest missing at {manifest_path}")

    # --- evaluation protocol, from cache -------------------------------------------------------
    protocol_key = cache_key(
        args.checkpoint,
        manifest_path,
        protocol="cat_individuals",
        image_size=args.image_size,
        tta=("identity", "hflip"),
        extra={"queries_per_identity": 1, "seed": 1337},
    )
    protocol = load(args.cache_dir, protocol_key)
    if protocol is None:
        print(f"protocol cache miss for {protocol_key}; run tools.retrieval_research first")
        return 1
    query = l2_normalize(protocol["query"])
    gallery = l2_normalize(protocol["gallery"])
    baseline = retrieval_metrics(
        (query @ gallery.T).astype(np.float32), protocol["query_labels"], protocol["gallery_labels"]
    )
    print(f"baseline (no whitening)   hit@1={baseline['hit@1']:.4f} mINP={baseline['mINP']:.4f}")

    # --- training descriptors, embedded once and cached ----------------------------------------
    train_records = split_records(manifest_path, splits_dir, "train")
    print(f"training identities' images: {len(train_records)}")
    train_key = cache_key(
        args.checkpoint,
        manifest_path,
        protocol="train-split",
        image_size=args.image_size,
        tta=("identity", "hflip"),
        extra={"split": "train"},
    )

    def produce() -> dict:
        embedder = Embedder.load(args.checkpoint, device=args.device)
        started = time.perf_counter()
        result = embed_records(
            embedder, [r.path for r in train_records], image_size=args.image_size, batch_size=32
        )
        LOGGER.info("embedded %d training images in %.1fs", len(train_records), time.perf_counter() - started)
        vectors = np.asarray(result.vectors, dtype=np.float32)
        return {
            "vectors": vectors,
            "ids": np.asarray([r.image_id for r in train_records], dtype="U256"),
            "labels": np.asarray([r.identity for r in train_records], dtype="U128"),
        }

    train_payload = load_arrays(args.cache_dir, train_key, list_fields=("ids", "labels"))
    if train_payload is not None:
        print("training descriptors: loaded from cache")
    else:
        train_payload = produce()
        save_arrays(args.cache_dir, train_key, train_payload, list_fields=("ids", "labels"))
        print("training descriptors: freshly embedded and cached")
    train_vectors = np.asarray(train_payload["vectors"], dtype=np.float32)
    print(f"training descriptors: {train_vectors.shape}")

    dims = [int(value) for value in args.dims.split(",") if value.strip()]
    rows: list[dict] = []

    def evaluate(label: str, q: np.ndarray, g: np.ndarray, dim: int) -> None:
        metrics = retrieval_metrics(
            (q @ g.T).astype(np.float32), protocol["query_labels"], protocol["gallery_labels"]
        )
        rows.append({"fitting": label, "dim": dim, **metrics})
        print(
            f"  {label:<12} dim={dim:<5} hit@1={metrics['hit@1']:.4f} "
            f"mINP={metrics['mINP']:.4f} mAP={metrics['mAP']:.4f}"
        )

    print()
    print("whitening fitted on the TRAINING identities:")
    for dim in dims:
        components, scale = fit_whitening(train_vectors, dim)
        evaluate(
            "train",
            apply_whitening(query, components, scale),
            apply_whitening(gallery, components, scale),
            dim,
        )

    print()
    print("whitening fitted on the GALLERY (the existing behaviour, as a control):")
    gallery_stack = np.concatenate([query, gallery], axis=0)
    for dim in dims:
        components, scale = fit_whitening(gallery_stack, dim)
        evaluate(
            "gallery",
            apply_whitening(query, components, scale),
            apply_whitening(gallery, components, scale),
            dim,
        )

    best = max(rows, key=lambda row: (row["hit@1"], row["mINP"]))
    print()
    print(
        f"best: {best['fitting']} whitening, dim={best['dim']} -> hit@1={best['hit@1']:.4f} "
        f"mINP={best['mINP']:.4f}"
    )
    print(f"baseline hit@1={baseline['hit@1']:.4f}; delta {best['hit@1'] - baseline['hit@1']:+.4f}")
    print(f"one query is {1.0 / len(protocol['query_labels']):.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "baseline": baseline,
                    "training_images": len(train_records),
                    "protocol": {
                        "queries": len(protocol["query_labels"]),
                        "gallery": len(protocol["gallery_labels"]),
                    },
                    "unit": 1.0 / len(protocol["query_labels"]),
                    "results": rows,
                    "best": best,
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
