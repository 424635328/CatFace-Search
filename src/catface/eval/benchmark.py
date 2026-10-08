"""The benchmark runner — one entry point that produces comparable numbers.

Every configuration in a report goes through this function, which is what makes the
numbers comparable: identical splits, identical TTA, identical post-processing
options, identical metric code. Differences between rows in the resulting table can
therefore be attributed to the model, not to a difference in harness.

The runner also records, for each configuration, its embedding time and descriptor
width, because a baseline that wins by 0.3 points at 8x the inference cost is not
necessarily the right production choice.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..data.manifest import FaceRecord
from ..errors import BenchmarkError
from ..logging_utils import environment_report, get_logger, new_run_id
from ..models.embedder import Embedder, embed_records
from .metrics import (
    RetrievalMetrics,
    VerificationMetrics,
    bootstrap_recall_ci,
    evaluate_retrieval,
    evaluate_verification,
    paired_bootstrap_delta,
)
from .postprocess import (
    AlphaQueryExpansion,
    WhiteningTransform,
    alpha_query_expansion,
    database_side_augmentation,
    l2_normalize,
)
from .protocols import Split

LOGGER = get_logger("eval.benchmark")


@dataclass
class PostprocessConfig:
    """Label-free descriptor transforms applied identically to every configuration."""

    whiten: str = "none"
    """``none`` | ``pca`` | ``pcaw``."""
    whiten_dim: int = 0
    """Target dimension (0 keeps the descriptor width)."""
    dba: bool = False
    dba_k: int = 3
    dba_alpha: float = 3.0
    query_expansion: str = "none"
    """``none`` | ``aqe``."""
    aqe_top_k: int = 3
    aqe_alpha: float = 3.0
    aqe_reverse_weight: float = 0.5

    def describe(self) -> dict[str, Any]:
        return {
            "whiten": self.whiten,
            "whiten_dim": self.whiten_dim,
            "dba": self.dba,
            "dba_k": self.dba_k,
            "dba_alpha": self.dba_alpha,
            "query_expansion": self.query_expansion,
            "aqe_top_k": self.aqe_top_k,
            "aqe_alpha": self.aqe_alpha,
            "aqe_reverse_weight": self.aqe_reverse_weight,
        }


@dataclass
class ConfigurationResult:
    """Everything measured for one model configuration."""

    name: str
    model: dict[str, Any]
    postprocess: dict[str, Any]
    retrieval: RetrievalMetrics | None = None
    retrieval_ci: tuple[float, float] | None = None
    verification: VerificationMetrics | None = None
    embedding_time_s: float = 0.0
    images_embedded: int = 0
    descriptor_dim: int = 0
    """Width of the descriptor actually compared (after pooling, projection and whitening)."""
    backbone_dim: int = 0
    """Pooled output width of the backbone, before projection — shows the architecture."""
    projected_dim: int = 0
    """Head output width, before any whitening."""
    notes: list[str] = field(default_factory=list)
    vectors: np.ndarray | None = None
    """Kept in memory for paired significance testing; not serialised."""
    query_labels: np.ndarray | None = None
    gallery_labels: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "postprocess": self.postprocess,
            "embedding_time_s": round(self.embedding_time_s, 3),
            "images_embedded": self.images_embedded,
            "descriptor_dim": self.descriptor_dim,
            "backbone_dim": self.backbone_dim,
            "projected_dim": self.projected_dim,
            "notes": self.notes,
        }
        if self.retrieval is not None:
            payload["retrieval"] = self.retrieval.to_dict()
            if self.retrieval_ci is not None:
                payload["retrieval"]["hit@1_ci95"] = list(self.retrieval_ci)
        if self.verification is not None:
            payload["verification"] = self.verification.to_dict()
        return payload


def _embed_split(
    embedder: Embedder,
    records: Sequence[FaceRecord],
    batch_size: int,
    image_size: int | None,
) -> tuple[np.ndarray, list[str], float, list[str]]:
    """Embed a set of records; returns vectors, ids, elapsed seconds and warnings."""
    paths = [record.path for record in records]
    started = time.perf_counter()
    result = embed_records(
        embedder, paths, image_size=image_size, batch_size=batch_size
    )
    elapsed = time.perf_counter() - started
    return result.vectors, result.ids, elapsed, result.disagreed


def apply_postprocessing(
    query: np.ndarray,
    gallery: np.ndarray,
    config: PostprocessConfig,
    seed: int = 1337,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the configured descriptor transforms.

    Order matters and is fixed: whitening rotates the space, DBA denoises references
    inside that space, and query expansion is last because it depends on the final
    metric geometry. Any other order produces a different (and usually worse) result,
    so it is not configurable.
    """
    diagnostics: dict[str, Any] = {}
    q = l2_normalize(query)
    g = l2_normalize(gallery)

    if config.whiten != "none":
        transform = WhiteningTransform.fit(g, dim=config.whiten_dim, mode=config.whiten)
        q = transform.transform(q)
        g = transform.transform(g)
        diagnostics["whitening"] = {
            "output_dim": transform.output_dim,
            "mode": transform.mode,
        }

    if config.dba:
        g = database_side_augmentation(g, k=config.dba_k, alpha=config.dba_alpha)
        diagnostics["dba"] = {"k": config.dba_k, "alpha": config.dba_alpha}

    if config.query_expansion != "none":
        expansion = AlphaQueryExpansion(
            enabled=True,
            top_k=config.aqe_top_k,
            alpha=config.aqe_alpha,
            reverse_weight=config.aqe_reverse_weight,
        )
        q = alpha_query_expansion(q, g, expansion)
        diagnostics["query_expansion"] = {
            "top_k": config.aqe_top_k,
            "alpha": config.aqe_alpha,
            "reverse_weight": config.aqe_reverse_weight,
        }

    diagnostics["query_dim"] = int(q.shape[1]) if q.ndim == 2 else 0
    diagnostics["gallery_dim"] = int(g.shape[1]) if g.ndim == 2 else 0
    return q, g, diagnostics


def benchmark_configuration(
    name: str,
    embedder: Embedder,
    split: Split,
    postprocess: PostprocessConfig | None = None,
    batch_size: int = 32,
    image_size: int | None = None,
    recall_ks: Sequence[int] = (1, 5, 10),
    far_targets: Sequence[float] = (1e-3, 1e-2, 1e-1),
    bootstrap_samples: int = 500,
    seed: int = 1337,
    verification_pairs: tuple[np.ndarray, np.ndarray] | None = None,
) -> ConfigurationResult:
    """Benchmark one embedder on one split.

    Args:
        name: Row label in the report.
        embedder: Loaded model.
        split: Query/gallery partition.
        postprocess: Descriptor transforms (identical across rows for fair comparison).
        verification_pairs: Optional ``(scores, labels)`` already computed — supplied
            by the caller when a pair protocol runs alongside retrieval.

    Returns:
        A populated :class:`ConfigurationResult`.
    """
    postprocess = postprocess or PostprocessConfig()
    notes: list[str] = []

    query_vectors, query_ids, query_seconds, query_disagreed = _embed_split(
        embedder, split.query_records, batch_size, image_size
    )
    gallery_vectors, gallery_ids, gallery_seconds, gallery_disagreed = _embed_split(
        embedder, split.gallery_records, batch_size, image_size
    )
    if query_vectors.size == 0 or gallery_vectors.size == 0:
        raise BenchmarkError(f"Configuration {name!r} produced no embeddings")

    if len(query_ids) != split.num_queries or len(gallery_ids) != split.num_gallery:
        # Positional alignment between labels and vectors is load-bearing.
        raise BenchmarkError(
            f"Configuration {name!r}: embedded {len(query_ids)}/{split.num_queries} "
            f"queries and {len(gallery_ids)}/{split.num_gallery} gallery images"
        )

    q, g, diagnostics = apply_postprocessing(
        query_vectors, gallery_vectors, postprocess, seed=seed
    )
    if diagnostics.get("query_dim") != diagnostics.get("gallery_dim"):
        raise BenchmarkError(
            f"Query and gallery descriptors have different widths after "
            f"post-processing: {diagnostics}"
        )

    similarity = (q @ g.T).astype(np.float32)
    # Recompute the self-mask against the ids actually embedded, so a filtered
    # gallery cannot leave stale True entries pointing at the wrong column.
    self_mask = np.array(
        [[qid == gid for gid in gallery_ids] for qid in query_ids], dtype=bool
    )

    retrieval = evaluate_retrieval(
        similarity,
        query_labels=split.query_labels,
        gallery_labels=split.gallery_labels,
        recall_ks=recall_ks,
        exclude_self=self_mask if self_mask.any() else None,
    )

    ci: tuple[float, float] | None = None
    if bootstrap_samples > 0 and 1 in recall_ks:
        try:
            ci = bootstrap_recall_ci(
                similarity,
                split.query_labels,
                split.gallery_labels,
                k=1,
                samples=bootstrap_samples,
                seed=seed,
                exclude_self=self_mask if self_mask.any() else None,
            )
        except BenchmarkError:
            ci = None

    verification: VerificationMetrics | None = None
    if verification_pairs is not None:
        scores, labels = verification_pairs
        verification = evaluate_verification(scores, labels, far_targets=far_targets)

    if query_disagreed or gallery_disagreed:
        notes.append(
            f"{len(query_disagreed) + len(gallery_disagreed)} descriptor(s) had weakly "
            "consistent TTA views (possible pose/orientation ambiguity)"
        )
    notes.append(json.dumps(diagnostics, sort_keys=True))

    return ConfigurationResult(
        name=name,
        model=embedder.describe_config(),
        postprocess=postprocess.describe(),
        retrieval=retrieval,
        retrieval_ci=ci,
        verification=verification,
        embedding_time_s=query_seconds + gallery_seconds,
        images_embedded=len(query_ids) + len(gallery_ids),
        descriptor_dim=int(q.shape[1]) if q.ndim == 2 else 0,
        backbone_dim=int(getattr(embedder.backbone, "feature_dim", 0)),
        projected_dim=int(embedder.head.embedding_dim),
        notes=notes,
        vectors=similarity,
        query_labels=split.query_labels,
        gallery_labels=split.gallery_labels,
    )


def compare_configurations(
    results: Sequence[ConfigurationResult],
    baseline_name: str | None = None,
    k: int = 1,
    samples: int = 500,
    seed: int = 1337,
) -> list[dict[str, Any]]:
    """Paired significance tests of every row against a baseline.

    Returns one entry per non-baseline row with the observed delta in recall@k, a
    paired bootstrap confidence interval, and the one-sided p-value
    ``P(candidate is not better)``.
    """
    if not results:
        return []
    baseline = next((r for r in results if r.name == baseline_name), results[0])
    if baseline.vectors is None:
        return []

    comparisons: list[dict[str, Any]] = []
    for candidate in results:
        if candidate.name == baseline.name or candidate.vectors is None:
            continue
        if candidate.vectors.shape != baseline.vectors.shape:
            comparisons.append({
                "candidate": candidate.name,
                "baseline": baseline.name,
                "error": "similarity matrices differ in shape; paired test not applicable",
            })
            continue
        stats = paired_bootstrap_delta(
            candidate.vectors,
            baseline.vectors,
            baseline.query_labels,
            baseline.gallery_labels,
            k=k,
            samples=samples,
            seed=seed,
        )
        comparisons.append({
            "candidate": candidate.name,
            "baseline": baseline.name,
            **stats,
        })
    return comparisons


def save_benchmark_report(
    output_dir: str | Path,
    results: Sequence[ConfigurationResult],
    comparisons: Sequence[dict[str, Any]],
    protocol: dict[str, Any],
    run_name: str,
) -> Path:
    """Write ``metrics.json`` + ``summary.md`` for one benchmark run."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id()

    payload = {
        "run_id": run_id,
        "run_name": run_name,
        "protocol": protocol,
        "environment": environment_report(),
        "results": [r.to_dict() for r in results],
        "comparisons": list(comparisons),
    }
    metrics_path = target / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )

    summary = render_markdown_summary(results, comparisons, protocol, run_name)
    summary_path = target / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    LOGGER.info("Wrote %s and %s", metrics_path, summary_path)
    return metrics_path


def render_markdown_summary(
    results: Sequence[ConfigurationResult],
    comparisons: Sequence[dict[str, Any]],
    protocol: dict[str, Any],
    run_name: str,
) -> str:
    """Render a review-ready Markdown report."""
    lines: list[str] = []
    lines.append(f"# Benchmark report — {run_name}")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    for key, value in protocol.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## Retrieval")
    lines.append("")
    ks = sorted({k for r in results if r.retrieval for k in r.retrieval.hit_at})
    # Column names state the metric precisely, because they are easy to confuse:
    #   hit@k      how often a same-identity image is in the top k  (CMC / top-k accuracy)
    #   fullR@k    what share of that identity's gallery images are in the top k
    #   AP@k       ranking quality, averaged over the relevant items inside the top k
    header = ["Configuration", "dim"] + [f"hit@{k}" for k in ks] + \
             [f"fullR@{k}" for k in ks] + [f"AP@{k}" for k in ks] + \
             ["mINP", "mRR", "emb_s", "imgs"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for result in results:
        if result.retrieval is None:
            continue
        row = [result.name, str(result.descriptor_dim)]
        row += [f"{result.retrieval.hit_at.get(k, float('nan')):.4f}" for k in ks]
        row += [f"{result.retrieval.recall_at.get(k, float('nan')):.4f}" for k in ks]
        row += [f"{result.retrieval.map_at.get(k, float('nan')):.4f}" for k in ks]
        row += [
            f"{result.retrieval.mean_inverse_negative_penalty:.4f}",
            f"{result.retrieval.mean_reciprocal_rank:.4f}",
            f"{result.embedding_time_s:.1f}",
            str(result.images_embedded),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    with_ci = [r for r in results if r.retrieval_ci is not None]
    if with_ci:
        lines.append("### Top-1 hit rate with 95 % bootstrap CI")
        lines.append("")
        lines.append("| Configuration | hit@1 | 95 % CI |")
        lines.append("| --- | --- | --- |")
        for result in with_ci:
            assert result.retrieval is not None and result.retrieval_ci is not None
            lines.append(
                f"| {result.name} | {result.retrieval.hit_at.get(1, float('nan')):.4f} | "
                f"[{result.retrieval_ci[0]:.4f}, {result.retrieval_ci[1]:.4f}] |"
            )
        lines.append("")

    verified = [r for r in results if r.verification is not None]
    if verified:
        lines.append("## Verification")
        lines.append("")
        fars = sorted({far for r in verified if r.verification for far in r.verification.tar_at_far})
        header = ["Configuration", "AUC", "EER"] + [f"TAR@FAR={f:g}" for f in fars]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for result in verified:
            assert result.verification is not None
            row = [
                result.name,
                f"{result.verification.roc_auc:.4f}",
                f"{result.verification.eer:.4f}",
            ]
            row += [f"{result.verification.tar_at_far.get(f, float('nan')):.4f}" for f in fars]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if comparisons:
        lines.append("## Paired significance tests")
        lines.append("")
        lines.append("| Candidate | Baseline | Δhit@k | 95 % CI | P(not better) |")
        lines.append("| --- | --- | --- | --- | --- |")
        for entry in comparisons:
            if "error" in entry:
                lines.append(
                    f"| {entry['candidate']} | {entry['baseline']} | n/a | n/a | "
                    f"{entry['error']} |"
                )
                continue
            ci = entry["ci95"]
            lines.append(
                f"| {entry['candidate']} | {entry['baseline']} | "
                f"{entry['observed_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"{entry['p_a_not_better']:.3f} |"
            )
        lines.append("")

    notes = [(r.name, n) for r in results for n in r.notes]
    if notes:
        lines.append("## Diagnostics")
        lines.append("")
        for name, note in notes:
            lines.append(f"- **{name}**: {note}")
        lines.append("")

    return "\n".join(lines)


def sha1_of_split(split: Split) -> str:
    """Content hash of a split, so a report can be tied to the exact image set."""
    digest = __import__("hashlib").sha1()
    for record in sorted(split.query_records, key=lambda r: r.image_id):
        digest.update(f"Q{record.image_id}".encode())
    for record in sorted(split.gallery_records, key=lambda r: r.image_id):
        digest.update(f"G{record.image_id}".encode())
    return digest.hexdigest()


__all__ = [
    "ConfigurationResult",
    "PostprocessConfig",
    "apply_postprocessing",
    "benchmark_configuration",
    "compare_configurations",
    "render_markdown_summary",
    "save_benchmark_report",
    "sha1_of_split",
]
