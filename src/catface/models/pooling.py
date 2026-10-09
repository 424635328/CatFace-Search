"""Descriptor pooling and projection.

The choice here matters more than it looks. An ImageNet classifier's penultimate
activation is dominated by *semantic* features (this is a cat, this is fur), which is
why it performs poorly on "is this the same cat". Pooling and projection are the
cheapest levers that change what information survives into the embedding.
"""

from __future__ import annotations

from typing import Any

from ..errors import ModelError


def _torch():
    import torch

    return torch


class GeM:
    """Generalised-mean pooling with a learnable exponent.

    ``p -> 1`` approaches average pooling, ``p -> +inf`` approaches max pooling.
    For fine-grained faces, a ``p`` around 3 usually beats both because it keeps
    peak activations (whisker, ear, eye markings) from being averaged away.

    Behaves like a module: call it, and read ``p`` back as a tensor.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True) -> None:
        torch = _torch()
        self.eps = eps
        self.p = torch.nn.Parameter(torch.ones(1) * p, requires_grad=trainable)

    def __call__(self, tokens: Any) -> Any:
        """Pool ``(B, N, D)`` tokens into ``(B, D)``."""
        clamped = tokens.clamp(min=self.eps).pow(self.p)
        return clamped.mean(dim=1).pow(1.0 / self.p)

    def parameters(self):
        yield self.p


VALID_POOLING_MODES = ("auto", "cls", "gap", "gem", "cls_gap")


def resolve_pooling(mode: str, has_class_token: bool) -> str:
    """Turn ``auto`` into the pooling that suits the architecture.

    Convolutional backbones have no class token, so they must use spatial pooling; ViTs
    expose one, and concatenating it with the patch mean (DINOv3-style) keeps global
    semantics while retaining local texture. Choosing wrongly raises a confusing error
    deep inside the forward pass, so the decision is made explicitly and once.

    Args:
        mode: Requested pooling, possibly ``"auto"``.
        has_class_token: Whether the backbone emits a leading class token.

    Returns:
        A concrete pooling mode that the backbone can satisfy.

    Raises:
        ModelError: If the mode is unknown, or requires a class token the backbone lacks.
    """
    if mode not in VALID_POOLING_MODES:
        raise ModelError(f"Unknown pooling mode {mode!r}; expected one of {VALID_POOLING_MODES}")
    if mode == "auto":
        return "cls_gap" if has_class_token else "gap"
    if mode in ("cls", "cls_gap") and not has_class_token:
        raise ModelError(
            f"pooling={mode!r} requires a class token, which this backbone does not "
            "provide. Use 'gap', 'gem' or 'auto'."
        )
    return mode


def pool_tokens(
    tokens: Any,
    mode: str,
    prefix_tokens: int = 0,
    gem: GeM | None = None,
) -> Any:
    """Pool a token sequence according to ``mode``.

    Args:
        tokens: ``(B, N, D)`` sequence.
        mode: ``auto`` | ``cls`` | ``gap`` | ``gem`` | ``cls_gap``.
        prefix_tokens: Leading non-spatial tokens (1 for ViTs with a class token).
        gem: Required when ``mode == 'gem'``.

    Returns:
        ``(B, D)`` or ``(B, 2D)`` for ``cls_gap``.
    """
    torch = _torch()
    if tokens is None:
        raise ModelError("Backbone produced no tokens; pooling requires tokens")
    if tokens.ndim == 2:
        # A backbone that already pooled; nothing to do.
        return tokens

    mode = resolve_pooling(mode, has_class_token=prefix_tokens > 0)
    prefix = tokens[:, :prefix_tokens] if prefix_tokens else None
    patches = tokens[:, prefix_tokens:] if prefix_tokens else tokens

    if mode == "cls":
        if prefix is None:
            raise ModelError("pooling='cls' requires a backbone with a class token")
        return prefix[:, 0]
    if mode == "gap":
        return patches.mean(dim=1)
    if mode == "gem":
        if gem is None:
            raise ModelError("pooling='gem' requires a GeM instance")
        return gem(patches)
    if mode == "cls_gap":
        if prefix is None:
            raise ModelError("pooling='cls_gap' requires a backbone with a class token")
        # DINOv3-style concatenation: the class token keeps global semantics while
        # the patch mean retains local texture.
        return torch.cat([prefix[:, 0], patches.mean(dim=1)], dim=-1)
    raise ModelError(f"Unknown pooling mode: {mode!r}")


def pooled_dim(pooling: str, base_dim: int) -> int:
    """Output width of a pooling mode given a backbone's token width.

    ``auto`` is resolved optimistically to the class-token variant, because only the
    backbone knows whether a class token exists; ``Embedder`` re-resolves it at
    construction time once the backbone has been built.
    """
    if pooling in ("cls_gap", "auto"):
        return base_dim * 2
    return base_dim


__all__ = ["GeM", "pool_tokens", "pooled_dim", "resolve_pooling"]
