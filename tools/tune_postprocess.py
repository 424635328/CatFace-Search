"""Tune retrieval post-processing on validation identities, then report on test.

Why the split matters
---------------------
PCA-whitening dimension, DBA neighbourhood size and αQE strength are *hyper-parameters*.
Choosing them by looking at test-set scores produces numbers that do not survive contact
with new data. This script therefore:

1. fits and selects the configuration on the **val** identities,
2. re-evaluates the chosen configuration (and the untuned baseline) on the **test**
   identities, which the selection never saw,
3. reports both, so the size of any selection overfit is visible.

Post-processing is fitted on the gallery only, never on gallery+query, because fitting on
both leaks query information into the descriptor space.

Efficiency note
---------------
Descriptor extraction dominates the cost, so the query and gallery are embedded **once per
protocol** and every candidate configuration is then evaluated on the cached vectors.
Re-embedding for each candidate would multiply a minutes-long job by the size of the grid
for no benefit: post-processing is pure linear algebra next to a ViT forward pass.

Usage::

    python -m tools.tune_postprocess --checkpoint artifacts/train/dinov2s-arcface/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.data.manifest import Manifest  # noqa: E402
from catface.errors import CatFaceError  # noqa: E402
from catface.eval.benchmark import PostprocessConfig, apply_postprocessing  # noqa: E402
from catface.eval.metrics import evaluate_retrieval  # noqa: E402
from catface.eval.protocols import Split, build_identity_split  # noqa: E402
from catface.logging_utils import configure_utf8_console, get_logger  # noqa: E402
from catface.models.embedder import Embedder, embed_records  # noqa: E402

LOGGER = get_logger("tools.tune")


def load_split(manifest_path: Path, splits_dir: Path, split: str) -> list:
    """Load the records belonging to one identity-disjoint split."""
    manifest = Manifest.load(manifest_path)
    split_file = splits_dir / f"{split}.txt"
    if not split_file.is_file():
        raise CatFaceError(f"split file not found: {split_file}")
    ids = {line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    records = [r for r in manifest if r.image_id in ids]
    if not records:
        raise CatFaceError(f"split {split!r} selected no records")
    return records


def embed_protocol(
    embedder: Embedder,
    records: list,
    image_size: int,
    queries_per_identity: int = 1,
    seed: int = 1337,
) -> dict:
    """Build the protocol and extract every descriptor exactly once.

    Returns:
        A cache holding the split, the query/gallery vectors and the self-match mask.
    """
    split = build_identity_split(
        records, queries_per_identity=queries_per_identity, seed=seed, name="tune"
    )
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
        "embedded %d queries + %d gallery images in %.1fs",
        len(query.ids), len(gallery.ids), time.perf_counter() - started,
    )
    return {
        "split": split,
        "query": query.vectors,
        "gallery": gallery.vectors,
        "self_mask": self_mask,
    }


def score_cached(cache: dict, config: PostprocessConfig) -> tuple[dict, dict]:
    """Apply ``config`` to cached descriptors and score. No inference involved."""
    q, g, diagnostics = apply_postprocessing(cache["query"], cache["gallery"], config)
    similarity = (q @ g.T).astype(np.float32)
    split: Split = cache["split"]
    mask = cache["self_mask"]
    metrics = evaluate_retrieval(
        similarity, split.query_labels, split.gallery_labels,
        recall_ks=(1, 5, 10), exclude_self=mask if mask.any() else None,
    )
    return metrics.to_dict(), {
        "queries": split.num_queries,
        "gallery": split.num_gallery,
        **diagnostics,
    }


def embed_and_score(
    embedder: Embedder,
    records: list,
    config: PostprocessConfig,
    image_size: int,
    queries_per_identity: int = 1,
    seed: int = 1337,
) -> tuple[dict, dict]:
    """Embed once and score one configuration.

    Kept for callers that need a single configuration; a grid search should use
    :func:`embed_protocol` followed by :func:`score_cached` so it embeds only once.
    """
    cache = embed_protocol(embedder, records, image_size, queries_per_identity, seed)
    return score_cached(cache, config)


def candidate_configs(descriptor_dim: int = 0) -> list[PostprocessConfig]:
    """The hyper-parameter grid, kept small and monotone so the choice is interpretable.

    ``whiten_dim`` of 0 keeps the descriptor width. When ``descriptor_dim`` is known it is
    included as a candidate so at least one option matches the input dimensionality exactly,
    which is the setting most likely to be useful.
    """
    dims: list[int] = [0]
    for candidate in (768, 512, 256, descriptor_dim):
        if candidate and candidate not in dims:
            dims.append(candidate)

    configs: list[PostprocessConfig] = [PostprocessConfig()]
    for mode in ("pca", "pcaw"):
        for dim in dims:
            configs.append(PostprocessConfig(whiten=mode, whiten_dim=dim))
            configs.append(PostprocessConfig(whiten=mode, whiten_dim=dim, dba=True, dba_k=3))
            configs.append(PostprocessConfig(whiten=mode, whiten_dim=dim, query_expansion="aqe"))
            configs.append(PostprocessConfig(whiten=mode, whiten_dim=dim, dba=True,
                                             query_expansion="aqe"))
    configs.append(PostprocessConfig(dba=True))
    configs.append(PostprocessConfig(query_expansion="aqe"))

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[PostprocessConfig] = []
    for config in configs:
        key = json.dumps(config.describe(), sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(config)
    return unique


def main(argv: list[str] | None = None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Tune retrieval post-processing")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifests/cat_individuals_manifest.jsonl")
    parser.add_argument("--splits", default="data/manifests/cat_individuals_splits")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--queries-per-identity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--select-metric", default="hit@1",
                        help="Primary metric used to choose the configuration on val")
    parser.add_argument("--tie-metric", default="mINP",
                        help="Metric that breaks ties on the primary one. hit@1 saturates "
                             "long before mINP does, so without a tie-break the selection "
                             "degenerates to whichever candidate the grid listed first")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    embedder = Embedder.load(args.checkpoint, device=args.device or "cuda")
    LOGGER.info(
        "loaded %s (backbone=%s pooled=%d projected=%d)",
        args.checkpoint, embedder.config.backbone, embedder.backbone.feature_dim,
        embedder.head.embedding_dim,
    )
    manifest_path, splits_dir = Path(args.manifest), Path(args.splits)
    val_records = load_split(manifest_path, splits_dir, "val")
    test_records = load_split(manifest_path, splits_dir, "test")

    # Embed each protocol once; everything below is linear algebra on cached vectors.
    val_cache = embed_protocol(
        embedder, val_records, args.image_size, args.queries_per_identity, args.seed
    )
    test_cache = embed_protocol(
        embedder, test_records, args.image_size, args.queries_per_identity, args.seed
    )

    candidates = candidate_configs(int(val_cache["query"].shape[1]))
    results: list[dict] = []
    grid_started = time.perf_counter()
    for index, config in enumerate(candidates, start=1):
        val_metrics, protocol = score_cached(val_cache, config)
        results.append({"config": config.describe(), "val": val_metrics, "protocol": protocol})
        LOGGER.info(
            "[%2d/%2d] %s -> val hit@1=%.4f mINP=%.4f",
            index, len(candidates), json.dumps(config.describe(), sort_keys=True),
            val_metrics.get("hit@1", float("nan")), val_metrics.get("mINP", float("nan")),
        )
    LOGGER.info(
        "grid of %d configurations scored in %.1fs (descriptors reused)",
        len(candidates), time.perf_counter() - grid_started,
    )

    metric_key = args.select_metric
    tie_key = args.tie_metric

    def selection_key(entry: dict) -> tuple[float, float]:
        metrics = entry["val"]
        return (metrics.get(metric_key, 0.0), metrics.get(tie_key, 0.0))

    results.sort(key=selection_key, reverse=True)
    best = results[0]
    tied = sum(1 for entry in results
               if entry["val"].get(metric_key, 0.0) == best["val"].get(metric_key, 0.0))
    LOGGER.info(
        "selected on val by %s (tie-broken by %s): %s",
        metric_key, tie_key, json.dumps(best["config"], sort_keys=True),
    )
    if tied > 1:
        LOGGER.info(
            "%d of %d candidates tie on %s; %s decides among them because it does not "
            "saturate as early", tied, len(results), metric_key, tie_key,
        )

    best_config = PostprocessConfig(**best["config"])
    baseline_config = PostprocessConfig()
    report: dict = {
        "checkpoint": args.checkpoint,
        "select_metric": metric_key,
        "tie_metric": tie_key,
        "candidates_tied_on_select_metric": tied,
        "selected_on_val": best["config"],
        "candidates_evaluated": len(candidates),
        "val_ranking": [{"config": entry["config"], "metrics": entry["val"]} for entry in results],
    }
    for label, config in (("untuned", baseline_config), ("selected", best_config)):
        test_metrics, test_protocol = score_cached(test_cache, config)
        report[f"test_{label}"] = {"config": config.describe(), "metrics": test_metrics,
                                   "protocol": test_protocol}
        LOGGER.info("test (%s): %s", label, json.dumps(test_metrics, sort_keys=True))

    untuned = report["test_untuned"]["metrics"]
    selected = report["test_selected"]["metrics"]
    report["test_delta"] = {
        key: round(selected.get(key, 0.0) - untuned.get(key, 0.0), 4)
        for key in ("hit@1", "hit@5", "hit@10", "mINP", "mRR")
    }
    # A zero delta is a finding, not a failure: it means the selected configuration matches
    # the untuned baseline on test. Say so, with the reason, instead of leaving the reader to
    # infer it from five zeroes.
    if all(abs(value) < 1e-9 for value in report["test_delta"].values()):
        report["test_delta_note"] = (
            f"Selection changed nothing on test. On val, {tied} of {len(results)} candidates "
            f"tie on {metric_key}; the model has effectively saturated on that metric, so "
            "post-processing cannot improve it. The descriptor itself is the limitation, not "
            "the retrieval post-processing."
        )
    LOGGER.info("test delta (selected - untuned): %s", json.dumps(report["test_delta"]))

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
