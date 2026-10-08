"""Benchmark a HuggingFace image-embedding model (native resize path) for comparison.

Used to answer a specific question: is CALFW hard for *every* encoder, or only for the
timm backbones measured in the main harness? Any model with an ``AutoImageProcessor``
works, which makes it easy to slot in a domain-trained animal-identification checkpoint as
a reference point.

The point of including a domain-adapted model is diagnostic, not competitive: it converts
"the scores look low" into either "this benchmark is not discriminative" (reference also
fails) or "our encoders are the bottleneck" (reference succeeds).

Usage::

    python -m tools.benchmark_hf_embedder \
        --model AvitoTech/DINO-v2-small-for-animal-identification \
        --pairs data/calfw/pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.eval.metrics import evaluate_verification, pairwise_cosine
from catface.eval.postprocess import l2_normalize


def embed_paths(model, processor, paths: list[str], device: str, batch_size: int = 32) -> np.ndarray:
    """Embed image files with a HuggingFace vision model, honouring its own preprocessing."""
    import torch
    from PIL import Image

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            images = []
            for path in chunk:
                with Image.open(path) as handle:
                    images.append(handle.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            # ViT-style models expose the pooled CLS token; fall back to mean-pooling.
            if getattr(outputs, "pooler_output", None) is not None:
                features = outputs.pooler_output
            elif getattr(outputs, "last_hidden_state", None) is not None:
                features = outputs.last_hidden_state[:, 0]
            else:  # pragma: no cover - model-specific
                raise RuntimeError("Model produced no usable pooled output")
            vectors.append(features.float().cpu().numpy())
    matrix = np.vstack(vectors)
    return l2_normalize(matrix)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark a HuggingFace embedding model on CALFW")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", default="data/calfw/pairs.csv")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N pairs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[hf] device={device} model={args.model}")

    with Path(args.pairs).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]
    paths_a = [row["path_a"] for row in rows]
    paths_b = [row["path_b"] for row in rows]
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)

    started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    load_seconds = time.perf_counter() - started
    print(f"[hf] processor input size: {processor.size if hasattr(processor, 'size') else 'unknown'}")

    started = time.perf_counter()
    left = embed_paths(model, processor, paths_a, device, args.batch_size)
    right = embed_paths(model, processor, paths_b, device, args.batch_size)
    embed_seconds = time.perf_counter() - started

    scores = pairwise_cosine(left, right)
    metrics = evaluate_verification(scores, labels, far_targets=(1e-3, 1e-2, 1e-1))

    # A pair-level AUC should agree with a nearest-neighbour view of the same data, which
    # catches an accidental misalignment between scores and labels.
    nn_scores = left @ right.T
    nn_accuracy = float((nn_scores.argmax(axis=1) == np.arange(len(labels))).mean())

    report = {
        "model": args.model,
        "protocol": {"pairs": len(rows), "positives": int(labels.sum())},
        "device": device,
        "load_seconds": round(load_seconds, 2),
        "embed_seconds": round(embed_seconds, 2),
        "descriptor_dim": int(left.shape[1]),
        "verification": metrics.to_dict(),
        "sanity": {
            "exact_pair_is_argmax_rate": nn_accuracy,
            "note": "the diagonal should dominate if scores and labels are aligned",
        },
    }
    print(json.dumps(report, indent=2))

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[hf] written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
