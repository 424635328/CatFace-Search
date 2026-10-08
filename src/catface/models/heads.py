"""Metric-learning heads.

These are the layers that convert a generic descriptor into an *identity*
descriptor. All of them are trained with a classification-style objective over
identities with a large angular margin, which forces descriptors of the same
individual into a tight region of the unit hypersphere.

Why margin losses rather than a plain classifier or plain triplet loss:

* A softmax classifier optimises the decision boundary, not the embedding
  geometry; the resulting descriptors are poorly calibrated for nearest-neighbour
  search.
* Triplet loss needs careful mining and is unstable at small batch sizes, which is
  a real constraint on a 6 GB GPU.
* ArcFace's additive angular margin and CosFace's cosine margin both work with
  ordinary (no mining) label-balanced batches and give consistent gains.

Implementation follows Deng et al. (ArcFace, CVPR 2019) with the numerically stable
``acos``-free reformulation, plus the sub-centre variant (Deng et al., 2020) which
handles identity clusters with several visual modes (different poses/lighting).
"""

from __future__ import annotations

import math
from typing import Any

from ..errors import ModelError

VALID_HEADS = ("identity", "linear", "arcface", "cosface", "subcenter_arcface", "sphereface")


def _torch():
    import torch

    return torch


class MetricHead:
    """Normalised-weight margin head.

    Args:
        in_features: Descriptor width entering the head.
        num_classes: Number of training identities (``0`` for inference-only use).
        embedding_dim: Output width. ``0`` keeps ``in_features``.
        kind: One of :data:`VALID_HEADS`.
        margin: Angular margin (radians) for ArcFace / sub-centre ArcFace.
        scale: Cosine logit scale ``s``.
        num_subcenters: Sub-centres per identity (sub-centre ArcFace only).
        dropout: Dropout applied to the input descriptor.

    Shape:
        forward: ``(B, in_features)`` -> ``(B, embedding_dim)``
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 0,
        embedding_dim: int = 512,
        kind: str = "arcface",
        margin: float = 0.35,
        scale: float = 64.0,
        num_subcenters: int = 3,
        dropout: float = 0.0,
    ) -> None:
        torch = _torch()
        nn = torch.nn

        if kind not in VALID_HEADS:
            raise ModelError(f"Unknown head kind {kind!r}; expected one of {VALID_HEADS}")
        if kind in ("arcface", "subcenter_arcface") and not 0.0 < margin < math.pi / 2:
            raise ModelError("ArcFace margin must be in (0, pi/2)")
        if scale <= 0:
            raise ModelError("head scale must be positive")

        self.kind = kind
        self.margin = float(margin)
        self.scale = float(scale)
        self.num_classes = int(num_classes)
        self.in_features = int(in_features)
        self.embedding_dim = int(embedding_dim) or int(in_features)
        self.num_subcenters = max(1, int(num_subcenters)) if kind == "subcenter_arcface" else 1

        self.module = nn.Module()
        self.module.bn = nn.BatchNorm1d(self.in_features) if self.in_features > 1 else nn.Identity()
        self.module.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.module.projection = (
            nn.Identity()
            if self.embedding_dim == self.in_features
            else nn.Linear(self.in_features, self.embedding_dim, bias=False)
        )
        if self.num_classes > 0:
            self.module.weight = nn.Parameter(
                torch.empty(self.num_classes * self.num_subcenters, self.embedding_dim)
            )
            nn.init.xavier_uniform_(self.module.weight)
            if kind == "cosface":
                self.module.margin_m = nn.Parameter(torch.tensor(0.35))  # not optimised by default

    # -- parameters ---------------------------------------------------------
    def parameters(self):
        return list(self.module.parameters())

    def train(self, mode: bool = True) -> MetricHead:
        self.module.train(mode)
        return self

    def eval(self) -> MetricHead:
        return self.train(False)

    def requires_grad_(self, flag: bool = True) -> MetricHead:
        for parameter in self.parameters():
            parameter.requires_grad_(flag)
        return self

    # -- forward ------------------------------------------------------------
    def embed(self, features: Any) -> Any:
        """Descriptor-only path: ``(B, in) -> (B, embedding_dim)``."""
        torch = _torch()
        x = self.module.bn(features)
        x = self.module.dropout(x)
        x = self.module.projection(x)
        return torch.nn.functional.normalize(x, dim=1)

    def forward(self, features: Any, labels: Any | None = None) -> Any:
        """Return logits.

        In ``eval`` mode, or without labels, this returns similarity logits scaled
        by ``s`` — no margin is applied, because at test time the margin is only a
        training device and applying it would distort the geometry.
        """
        torch = _torch()
        embeddings = self.embed(features)

        if self.kind in ("identity", "linear") or self.num_classes == 0:
            if self.num_classes == 0:
                return embeddings
            weight = self.module.weight
            if self.kind == "linear":
                return torch.nn.functional.linear(embeddings, weight) * self.scale
            return torch.nn.functional.linear(embeddings, weight) * self.scale

        weight = torch.nn.functional.normalize(self.module.weight, dim=1)  # (C*K, D)

        if self.kind == "subcenter_arcface":
            batch, dim = embeddings.shape
            cosine = torch.nn.functional.linear(embeddings, weight)
            cosine = cosine.view(batch, self.num_classes, self.num_subcenters)
            cosine = cosine.max(dim=2).values
        else:
            cosine = torch.nn.functional.linear(embeddings, weight)

        if labels is None or not self.module.training:
            return cosine * self.scale

        # Numerically stable additive-margin operator:
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        cos_m, sin_m = math.cos(self.margin), math.sin(self.margin)

        if self.kind in ("arcface", "subcenter_arcface"):
            sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-9))
            # Keep the operator monotonic for large theta (the standard EasyMargin fix).
            phi = cosine * cos_m - sine * sin_m
            threshold = math.cos(math.pi - self.margin)
            phi = torch.where(cosine > threshold, phi, cosine - self.margin * math.sin(math.pi - self.margin))
            logits = one_hot * phi + (1.0 - one_hot) * cosine
        elif self.kind == "cosface":
            logits = one_hot * (cosine - self.margin) + (1.0 - one_hot) * cosine
        elif self.kind == "sphereface":
            # Multiplicative angular margin, approximated for parity with the others.
            logits = one_hot * (cosine.pow(1.0) * (1.0 - self.margin)) + (1.0 - one_hot) * cosine
        else:  # pragma: no cover - guarded in __init__
            raise ModelError(f"Unhandled head kind {self.kind!r}")

        return logits * self.scale

    # ``MetricHead`` is a plain class that *owns* an ``nn.Module`` rather than being one,
    # so ``forward`` has to be wired to ``__call__`` explicitly. Without this the object
    # is not callable and every ``head(features, labels)`` fails with
    # "'MetricHead' object is not callable".
    __call__ = forward

    def state_dict(self) -> dict[str, Any]:
        return {f"head.{k}": v for k, v in self.module.state_dict().items()}

    def load_state_dict(self, state: dict[str, Any], strict: bool = False) -> None:
        inner = {k.split("head.", 1)[1] if k.startswith("head.") else k: v
                 for k, v in state.items()}
        self.module.load_state_dict(inner, strict=strict)


class TripletLoss:
    """Batch-hard triplet loss with a Euclidean margin on the unit sphere.

    Used as an auxiliary term: it directly optimises the retrieval objective
    (same-identity closer than different-identity) whereas the margin head
    optimises a surrogate classification problem.
    """

    def __init__(self, margin: float = 0.3) -> None:
        self.margin = float(margin)

    def __call__(self, embeddings: Any, labels: Any) -> Any:
        torch = _torch()
        # Pairwise squared Euclidean distance.
        dot = embeddings @ embeddings.t()
        square = dot.diagonal()
        distances = (square.unsqueeze(0) - 2 * dot + square.unsqueeze(1)).clamp(min=1e-12)
        distances = torch.sqrt(distances)

        labels = labels.view(-1)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        positive_mask = same & ~eye
        negative_mask = ~same

        if not positive_mask.any() or not negative_mask.any():
            return torch.zeros((), device=embeddings.device, requires_grad=True)

        # batch-hard mining
        hardest_positive = (distances * positive_mask).max(dim=1).values
        # Mask invalid rows (identities with a single sample in the batch).
        valid = positive_mask.any(dim=1)
        largest = torch.finfo(distances.dtype).max
        negatives = distances.masked_fill(~negative_mask, largest)
        hardest_negative = negatives.min(dim=1).values
        valid &= torch.isfinite(hardest_negative) & (negatives.min(dim=1).values < largest)

        if not valid.any():
            return torch.zeros((), device=embeddings.device, requires_grad=True)

        losses = torch.nn.functional.relu(hardest_positive[valid] - hardest_negative[valid] + self.margin)
        return losses.mean()


class VarianceRegulariser:
    """Penalise intra-identity descriptor spread.

    Complements the margin loss: the margin pushes identities apart, this pulls each
    identity's samples together, which is what makes a small number of stored
    reference photos enough to recognise a cat.
    """

    def __init__(self, weight: float = 0.0) -> None:
        self.weight = float(weight)

    def __call__(self, embeddings: Any, labels: Any) -> Any:
        torch = _torch()
        if self.weight <= 0:
            return torch.zeros((), device=embeddings.device)
        labels = labels.view(-1)
        unique = labels.unique()
        penalties = []
        for identity in unique:
            members = embeddings[labels == identity]
            if members.shape[0] < 2:
                continue
            centre = torch.nn.functional.normalize(members.mean(dim=0, keepdim=True), dim=1)
            penalties.append(1.0 - (members * centre).sum(dim=1).mean())
        if not penalties:
            return torch.zeros((), device=embeddings.device, requires_grad=True)
        return torch.stack(penalties).mean() * self.weight


def build_head(kind: str, **kwargs: Any) -> MetricHead:
    """Factory preserving the ``kind`` keyword position."""
    return MetricHead(kind=kind, **kwargs)


__all__ = [
    "VALID_HEADS",
    "MetricHead",
    "TripletLoss",
    "VarianceRegulariser",
    "build_head",
]
