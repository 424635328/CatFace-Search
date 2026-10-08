"""Benchmark orchestration: model specs -> protocol -> comparable table.

The runner exists so that every configuration in the final comparison goes through
identical code. In particular:

* one protocol object is built once and reused for every model (same queries, same
  gallery, same self-exclusion mask);
* post-processing options are passed identically to every model;
* models are loaded, evaluated and released one at a time, so a 6 GB GPU can compare a
  ViT-L against a ResNet-50 without both being resident.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .data.manifest import Manifest
from .errors import BenchmarkError, DataError
from .eval.benchmark import (
    ConfigurationResult,
    PostprocessConfig,
    benchmark_configuration,
    compare_configurations,
    render_markdown_summary,
    save_benchmark_report,
    sha1_of_split,
)
from .eval.protocols import Split
from .logging_utils import environment_report, get_logger, new_run_id, timed
from .models.embedder import embed_records
from .pipeline import build_eval_split, build_within_split, make_embedder

LOGGER = get_logger("benchmark")


@dataclass(frozen=True)
class ModelSpec:
    """One row of the comparison table.

    Syntax: ``name=backbone[:checkpoint]``

    * ``baseline=resnet50`` — the configuration this project started from.
    * ``dinov2b=dinov2_vitb14`` — a stronger pretrained backbone.
    * ``finetuned=dinov2_vitb14:artifacts/train/best.pt`` — the trained system. When a
      checkpoint is given, the backbone recorded inside it wins, which prevents a
      config/weights mismatch.
    """

    name: str
    backbone: str
    checkpoint: Path | None = None
    note: str = ""

    @classmethod
    def parse(cls, spec: str) -> ModelSpec:
        if "=" not in spec:
            raise DataError(
                f"Model spec {spec!r} must look like name=backbone[:checkpoint]"
            )
        name, _, rest = spec.partition("=")
        name = name.strip()
        if not name:
            raise DataError(f"Model spec {spec!r} has an empty name")
        rest = rest.strip()
        # Split on the *first* colon only: Windows drive letters contain one.
        backbone, sep, checkpoint = rest.partition(":")
        backbone = backbone.strip()
        checkpoint_path = Path(checkpoint.strip()) if sep and checkpoint.strip() else None
        if checkpoint_path is not None and not checkpoint_path.is_file():
            raise DataError(f"Checkpoint for {name!r} not found: {checkpoint_path}")
        return cls(name=name, backbone=backbone, checkpoint=checkpoint_path)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backbone": self.backbone,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "note": self.note,
        }


def resolve_protocol(
    config: PipelineConfig,
    protocol: str,
    queries_per_identity: int = 1,
    max_gallery_per_identity: int | None = None,
) -> tuple[Split, dict[str, Any]]:
    """Build the evaluation protocol named by ``protocol``.

    ``oiid_identity``
        Within-corpus retrieval on OIID *test* identities (identity-disjoint from
        training). Measures whether the descriptor separates individuals at all.
    ``cat_individuals``
        Within-corpus retrieval on the Kaggle corpus: uncontrolled photos, many images
        per cat. The closest available proxy for production traffic.
    ``cross_dataset``
        Query on OIID test identities, gallery from the Kaggle corpus, restricted to the
        individuals both corpora share. Deliberately hard: it changes camera, framing
        and lighting simultaneously.
    """
    manifest_dir = Path(config.data.manifest)
    metadata: dict[str, Any] = {"protocol": protocol, "policy": "query/gallery, self excluded"}

    if protocol == "oiid_identity":
        manifest_path = manifest_dir / "oiid_cat_manifest.jsonl"
        if not manifest_path.is_file():
            raise DataError(
                f"OIID manifest missing at {manifest_path}. Run "
                "`catface prepare --source oiid_cat` first."
            )
        split_dir = manifest_dir / "oiid_cat_splits"
        test_file = split_dir / "test.txt"
        if not test_file.is_file():
            raise DataError(f"OIID split file missing at {test_file}")
        ids = {line.strip() for line in test_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        manifest = Manifest.load(manifest_path)
        test_records = [r for r in manifest if r.image_id in ids]
        if not test_records:
            raise DataError("OIID test split selected no records")
        split = build_within_split(
            test_records,
            queries_per_identity=queries_per_identity,
            name="oiid_test_identities",
            seed=config.seed,
            max_gallery_per_identity=max_gallery_per_identity,
        )
        metadata.update({
            "corpus": "Oxford-IIIT Pet cats (test identities)",
            "crop": "oiid annotated head box",
            "training_overlap": "none — identities are disjoint from training",
        })
        return split, metadata

    if protocol == "cat_individuals":
        manifest_path = manifest_dir / "cat_individuals_manifest.jsonl"
        if not manifest_path.is_file():
            raise DataError(
                f"Cat-individuals manifest missing at {manifest_path}. Run "
                "`catface prepare --source cat_individuals` first."
            )
        manifest = Manifest.load(manifest_path)
        records = list(manifest)
        split = build_within_split(
            records,
            queries_per_identity=queries_per_identity,
            name="cat_individuals",
            seed=config.seed,
            max_gallery_per_identity=max_gallery_per_identity,
        )
        metadata.update({
            "corpus": "Kaggle Cat Individual Images",
            "crop": "whole image (corpus ships no face boxes)",
            "training_overlap": "none — disjoint from OIID training identities",
        })
        return split, metadata

    if protocol == "cross_dataset":
        query_manifest = manifest_dir / "oiid_cat_manifest.jsonl"
        gallery_manifest = manifest_dir / "cat_individuals_manifest.jsonl"
        for path in (query_manifest, gallery_manifest):
            if not path.is_file():
                raise DataError(f"Required manifest missing: {path}")
        split_dir = manifest_dir / "oiid_cat_splits"
        test_file = split_dir / "test.txt"
        ids = {line.strip() for line in test_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        query_records = [r for r in Manifest.load(query_manifest) if r.image_id in ids]
        # Restrict both sides to the shared identities so any positive pair exists.
        gallery_all = list(Manifest.load(gallery_manifest))
        query_ids = {r.identity for r in query_records}
        gallery_ids = {r.identity for r in gallery_all}
        shared = query_ids & gallery_ids
        if not shared:
            raise BenchmarkError(
                "OIID and Cat-individuals share no identity labels; the cross-dataset "
                "protocol cannot be evaluated. Use a shared labelling scheme first."
            )
        filtered_query = Manifest([r for r in query_records if r.identity in shared])
        filtered_gallery = Manifest([r for r in gallery_all if r.identity in shared])
        filtered_query.save(manifest_dir / "_cross_query.jsonl")
        filtered_gallery.save(manifest_dir / "_cross_gallery.jsonl")
        split = build_eval_split(
            config,
            manifest_dir / "_cross_query.jsonl",
            manifest_dir / "_cross_gallery.jsonl",
            name="cross_dataset",
        )
        metadata.update({
            "corpus": "query=OIID test, gallery=Cat-individuals",
            "shared_identities": len(shared),
            "training_overlap": "none",
        })
        return split, metadata

    raise DataError(f"Unknown protocol: {protocol!r}")


def run_benchmark(
    config: PipelineConfig,
    model_specs: Sequence[str | ModelSpec],
    protocol: str = "oiid_identity",
    queries_per_identity: int = 1,
    max_gallery_per_identity: int | None = None,
    whiten: str = "none",
    whiten_dim: int = 0,
    dba: bool = False,
    aqe: bool = False,
    batch_size: int = 32,
    image_size: int | None = None,
    baseline: str | None = None,
    tag: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate every model spec on one protocol and write the report."""
    specs = [spec if isinstance(spec, ModelSpec) else ModelSpec.parse(spec)
             for spec in model_specs]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise DataError(f"Model names must be unique, got {names}")

    split, protocol_metadata = resolve_protocol(
        config, protocol, queries_per_identity, max_gallery_per_identity
    )
    protocol_metadata.update({
        "queries": split.num_queries,
        "gallery": split.num_gallery,
        "identities_in_gallery": len(set(split.gallery_labels.tolist())),
        "split_sha1": sha1_of_split(split),
        "tta_views": list(config.model.tta),
        "image_size": image_size or config.model.image_size,
        "batch_size": batch_size,
        "postprocess": PostprocessConfig(whiten=whiten, whiten_dim=whiten_dim,
                                         dba=dba, query_expansion="aqe" if aqe else "none").describe(),
        "seed": config.seed,
    })
    LOGGER.info(
        "Protocol %s: %d queries / %d gallery over %d identities (split %s)",
        protocol, split.num_queries, split.num_gallery,
        protocol_metadata["identities_in_gallery"], protocol_metadata["split_sha1"][:12],
    )

    postprocess = PostprocessConfig(
        whiten=whiten, whiten_dim=whiten_dim, dba=dba,
        query_expansion="aqe" if aqe else "none",
    )

    resolved_device = device or config.resolve_device()
    results: list[ConfigurationResult] = []
    for spec in specs:
        LOGGER.info("=== Evaluating %s (backbone=%s) ===", spec.name, spec.backbone)
        with timed(LOGGER, f"evaluate {spec.name}", stage="benchmark"):
            if spec.checkpoint is not None:
                embedder = make_embedder(config, device=resolved_device, checkpoint=spec.checkpoint)
            else:
                # A per-model backbone override keeps the config file's other choices.
                previous = config.model.backbone
                config.model.backbone = spec.backbone
                try:
                    config.model.__post_init__()
                    embedder = make_embedder(config, device=resolved_device)
                finally:
                    config.model.backbone = previous
                    config.model.__post_init__()

            result = benchmark_configuration(
                name=spec.name,
                embedder=embedder,
                split=split,
                postprocess=postprocess,
                batch_size=batch_size,
                image_size=image_size,
                recall_ks=config.eval.recall_ks,
                far_targets=config.eval.far_targets,
                bootstrap_samples=config.eval.bootstrap_samples,
                seed=config.seed,
            )
            result.model["spec"] = spec.describe()
            results.append(result)
            LOGGER.info(
                "%s: hit@1=%.4f mAP@5=%.4f (dim=%d, %.1fs for %d images)",
                spec.name,
                result.retrieval.recall_at.get(1, float("nan")) if result.retrieval else float("nan"),
                result.retrieval.map_at.get(5, float("nan")) if result.retrieval else float("nan"),
                result.descriptor_dim, result.embedding_time_s, result.images_embedded,
            )

        # Release the model before building the next one (6 GB GPUs cannot hold two).
        del embedder
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    comparisons = compare_configurations(
        results, baseline_name=baseline, k=1, samples=config.eval.bootstrap_samples
    )

    run_name = tag or f"{protocol}-{len(specs)}models"
    output_dir = Path(config.output_dir) / "benchmarks" / run_name
    save_benchmark_report(output_dir, results, comparisons, protocol_metadata, run_name)

    return {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "protocol": protocol_metadata,
        "results": [r.to_dict() for r in results],
        "comparisons": comparisons,
        "environment": environment_report(),
        "table": render_markdown_summary(results, comparisons, protocol_metadata, run_name),
    }


@dataclass
class VerificationResult:
    """Verification scores for one model over a labelled pair set."""

    name: str
    model: dict[str, Any]
    metrics: Any
    pairs: int
    embedding_time_s: float
    image_a: np.ndarray | None = None
    image_b: np.ndarray | None = None
    labels: np.ndarray | None = None
    scores: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "pairs": self.pairs,
            "embedding_time_s": round(self.embedding_time_s, 3),
            "verification": self.metrics.to_dict(),
        }


def run_verification(
    config: PipelineConfig,
    model_specs: Sequence[str | ModelSpec],
    pairs_csv: Path,
    limit: int | None = None,
    batch_size: int = 64,
    device: str | None = None,
) -> dict[str, Any]:
    """Benchmark every model on a labelled same/different pair set (CALFW).

    Both halves of every pair are embedded once, then scored by cosine similarity. This
    is the metric that decides whether a user-facing threshold is usable, which recall@k
    cannot answer.
    """
    import csv as csv_module

    from .eval.metrics import evaluate_verification, pairwise_cosine
    from .pipeline import make_embedder

    # Accept both `ModelSpec` objects and their string form, so callers cannot trip
    # over the difference.
    specs = [spec if isinstance(spec, ModelSpec) else ModelSpec.parse(spec)
             for spec in model_specs]

    if not pairs_csv.is_file():
        raise DataError(f"Pairs CSV not found: {pairs_csv}")

    with pairs_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    if limit:
        rows = rows[:limit]
    if not rows:
        raise DataError(f"{pairs_csv} contains no pairs")

    paths_a = [row["path_a"] for row in rows]
    paths_b = [row["path_b"] for row in rows]
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    LOGGER.info("Verification protocol: %d pairs (%d same)", len(rows), int(labels.sum()))

    resolved_device = device or config.resolve_device()
    results: list[VerificationResult] = []
    for spec in specs:
        LOGGER.info("=== Verification: %s (backbone=%s) ===", spec.name, spec.backbone)
        started = time.perf_counter()
        with timed(LOGGER, f"verify {spec.name}", stage="verify"):
            if spec.checkpoint is not None:
                embedder = make_embedder(config, device=resolved_device, checkpoint=spec.checkpoint)
            else:
                previous = config.model.backbone
                config.model.backbone = spec.backbone
                try:
                    config.model.__post_init__()
                    embedder = make_embedder(config, device=resolved_device)
                finally:
                    config.model.backbone = previous
                    config.model.__post_init__()

            embedded_a = embed_records(embedder, paths_a, batch_size=batch_size)
            embedded_b = embed_records(embedder, paths_b, batch_size=batch_size)

        usable = min(len(embedded_a.ids), len(embedded_b.ids))
        if usable < len(rows):
            LOGGER.warning(
                "%s: only %d of %d pairs were readable; scoring the usable subset",
                spec.name, usable, len(rows),
            )
        vectors_a, vectors_b = embedded_a.vectors[:usable], embedded_b.vectors[:usable]
        pair_labels = labels[:usable]
        scores = pairwise_cosine(vectors_a, vectors_b)
        metrics = evaluate_verification(scores, pair_labels, far_targets=config.eval.far_targets)

        results.append(
            VerificationResult(
                name=spec.name,
                model={**embedder.describe_config(), "spec": spec.describe()},
                metrics=metrics,
                pairs=int(usable),
                embedding_time_s=time.perf_counter() - started,
                image_a=vectors_a,
                image_b=vectors_b,
                labels=pair_labels,
                scores=scores,
            )
        )
        LOGGER.info(
            "%s: AUC=%.4f EER=%.4f TAR@FAR=1e-2:%.4f",
            spec.name, metrics.roc_auc, metrics.eer,
            metrics.tar_at_far.get(0.01, float("nan")),
        )

        del embedder
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    output_dir = Path(config.output_dir) / "benchmarks" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": new_run_id(),
        "protocol": {
            "name": "CALFW cat face verification",
            "pairs": len(rows),
            "same_pairs": int(labels.sum()),
            "different_pairs": int(len(labels) - labels.sum()),
            "source": str(pairs_csv),
            "policy": "independent of every training corpus used here",
        },
        "environment": environment_report(),
        "results": [r.to_dict() for r in results],
    }
    (output_dir / "verification.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (output_dir / "verification.md").write_text(
        render_verification_summary(results, payload["protocol"]), encoding="utf-8"
    )
    payload["output_dir"] = str(output_dir)
    payload["table"] = render_verification_summary(results, payload["protocol"])
    return payload


def render_verification_summary(
    results: Sequence[VerificationResult],
    protocol: Mapping[str, Any],
) -> str:
    """Markdown table for the verification benchmarks."""
    lines = ["# Cat-face verification benchmark", ""]
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    for key, value in protocol.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    fars = sorted({far for r in results for far in r.metrics.tar_at_far})
    header = ["Configuration", "pairs", "AUC", "EER", "Acc@EER"] + [f"TAR@FAR={f:g}" for f in fars]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for result in results:
        row = [
            result.name,
            str(result.pairs),
            f"{result.metrics.roc_auc:.4f}",
            f"{result.metrics.eer:.4f}",
            f"{result.metrics.accuracy_at_eer:.4f}",
        ]
        row += [f"{result.metrics.tar_at_far.get(f, float('nan')):.4f}" for f in fars]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "EER = equal error rate (lower is better). TAR@FAR is the true-accept rate at a "
        "fixed false-accept rate; FAR=1e-2 means one-in-a-hundred impostor pairs is "
        "wrongly accepted."
    )
    return "\n".join(lines)


__all__ = [
    "ModelSpec",
    "VerificationResult",
    "render_verification_summary",
    "resolve_protocol",
    "run_benchmark",
    "run_verification",
]
