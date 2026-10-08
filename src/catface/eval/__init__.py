"""Evaluation-layer exports."""

from __future__ import annotations

from .benchmark import (
    ConfigurationResult,
    PostprocessConfig,
    apply_postprocessing,
    benchmark_configuration,
    compare_configurations,
    render_markdown_summary,
    save_benchmark_report,
)
from .metrics import (
    RetrievalMetrics,
    VerificationMetrics,
    bootstrap_recall_ci,
    cosine_similarity_matrix,
    evaluate_retrieval,
    evaluate_verification,
    paired_bootstrap_delta,
    pairwise_cosine,
)
from .postprocess import (
    AlphaQueryExpansion,
    WhiteningTransform,
    alpha_query_expansion,
    database_side_augmentation,
    l2_normalize,
)
from .protocols import Split, assert_identity_disjoint, build_identity_split

__all__ = [
    "AlphaQueryExpansion",
    "ConfigurationResult",
    "PostprocessConfig",
    "RetrievalMetrics",
    "Split",
    "VerificationMetrics",
    "WhiteningTransform",
    "alpha_query_expansion",
    "apply_postprocessing",
    "assert_identity_disjoint",
    "benchmark_configuration",
    "bootstrap_recall_ci",
    "build_identity_split",
    "compare_configurations",
    "cosine_similarity_matrix",
    "database_side_augmentation",
    "evaluate_retrieval",
    "evaluate_verification",
    "l2_normalize",
    "paired_bootstrap_delta",
    "pairwise_cosine",
    "render_markdown_summary",
    "save_benchmark_report",
]
