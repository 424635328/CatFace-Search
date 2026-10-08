"""Model-layer exports."""

from __future__ import annotations

from .backbone import Backbone, BackboneOutput, available_backbones, build_backbone
from .embedder import (
    CHECKPOINT_VERSION,
    Embedder,
    EmbedderConfig,
    EmbeddingResult,
    build_tta_transforms,
    embed_records,
)
from .heads import MetricHead, TripletLoss, VarianceRegulariser
from .pooling import GeM, pool_tokens

__all__ = [
    "CHECKPOINT_VERSION",
    "Backbone",
    "BackboneOutput",
    "Embedder",
    "EmbedderConfig",
    "EmbeddingResult",
    "GeM",
    "MetricHead",
    "TripletLoss",
    "VarianceRegulariser",
    "available_backbones",
    "build_backbone",
    "build_tta_transforms",
    "embed_records",
    "pool_tokens",
]
