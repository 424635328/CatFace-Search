"""Backbone registry.

A backbone answers exactly two questions: *what tensor does it produce for an
image*, and *how wide is that tensor*. Everything else — projection, margin loss,
normalisation, TTA — is layered on top, which is what makes backbones
interchangeable and comparable under one benchmark.

Every backbone returns a :class:`BackboneOutput` holding the pooled descriptor and,
where available, the spatial token map. Keeping the token map optional lets a
Global-Average-Pooled CNN and a CLS-token transformer share one interface without
pretending they are the same thing.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from ..errors import ModelError
from ..logging_utils import get_logger

LOGGER = get_logger("models.backbone")


@dataclass(slots=True)
class BackboneOutput:
    """Result of one forward pass through a backbone."""

    descriptor: Any
    """``(B, D)`` pooled feature used for the embedding."""
    tokens: Any | None = None
    """``(B, N, D)`` spatial tokens when the backbone exposes them."""
    prefix_tokens: int = 0
    """Number of leading tokens in ``tokens`` that are not spatial patches."""


class Backbone(Protocol):
    """Structural type every backbone must satisfy."""

    name: str
    feature_dim: int
    supports_tokens: bool

    def __call__(self, images: Any) -> BackboneOutput:  # pragma: no cover - protocol
        ...


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ModelError(
            "PyTorch is required. Install it with `pip install torch torchvision`."
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# torchvision CNNs
# ---------------------------------------------------------------------------
class TorchvisionBackbone:
    """ResNet-family backbone whose classifier head is removed.

    The original project used ``nn.Sequential(*list(resnet50(pretrained).children())[:-1])``,
    which silently drops the final ``AdaptiveAvgPool2d``. For a 224x224 input the
    remaining feature map is already ``7x7``, so the numbers happened to look right —
    but the descriptor was then ``mean``-reduced by the caller over the wrong axis,
    and any input size other than 224 produced a wrong-shaped embedding. This wrapper
    keeps the pooling layer and applies it explicitly.
    """

    supports_tokens = True
    #: Leading non-spatial tokens the backbone emits. A CNN has none, so `auto` pooling
    #: resolves to 'gap'; declaring this explicitly removes the need to sniff for a
    #: `features` attribute, which timm's ResNet does not expose.
    prefix_tokens = 0

    def __init__(
        self,
        architecture: str = "resnet50",
        pretrained: str | Path | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        torch = _torch()
        import torch.nn as nn
        import torchvision.models as tvm

        if architecture not in ("resnet50", "resnet101", "resnet34", "resnet18", "wide_resnet50_2"):
            raise ModelError(f"Unsupported torchvision architecture: {architecture}")

        weights_enum = getattr(tvm, f"{architecture.capitalize()}_Weights", None)
        local = Path(pretrained) if pretrained else None

        constructor = getattr(tvm, architecture)
        if local and local.is_file():
            LOGGER.info("Loading %s weights from %s", architecture, local)
            network = constructor(weights=None)
            state = torch.load(local, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # A checkpoint saved from an ``nn.Sequential`` wrapper needs re-keying.
            if any(key.startswith("0.") for key in list(state)[:5]):
                state = {key.split(".", 1)[1] if key[0].isdigit() else key: value
                         for key, value in state.items()}
                state = {key: value for key, value in state.items()
                         if not key.startswith("fc.")}
            missing, unexpected = network.load_state_dict(state, strict=False)
            if missing:
                LOGGER.warning("Missing keys when loading %s: %d", architecture, len(missing))
            if unexpected:
                LOGGER.warning("Unexpected keys when loading %s: %d", architecture, len(unexpected))
        else:
            if local:
                raise ModelError(f"Checkpoint not found: {local}")
            network = constructor(weights="DEFAULT" if weights_enum else None) if weights_enum \
                else constructor(pretrained=True)

        # Split into feature extractor (through the last conv block) + pooling.
        self.features = nn.Sequential(
            network.conv1, network.bn1, network.relu, network.maxpool,
            network.layer1, network.layer2, network.layer3, network.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.name = architecture
        self.feature_dim = int(network.fc.in_features)
        self.gradient_checkpointing = gradient_checkpointing

    def __call__(self, images: Any) -> BackboneOutput:
        torch = _torch()

        def forward(tensor: Any) -> Any:
            return self.features(tensor)

        if self.gradient_checkpointing and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            feature_map = checkpoint(forward, images, use_reentrant=False)
        else:
            feature_map = forward(images)
        pooled = self.pool(feature_map).flatten(1)
        tokens = feature_map.flatten(2).transpose(1, 2)
        return BackboneOutput(descriptor=pooled, tokens=tokens, prefix_tokens=0)


# ---------------------------------------------------------------------------
# DINOv2 / DINOv3 (torch.hub)
# ---------------------------------------------------------------------------
class Dinov2Backbone:
    """DINOv2 self-supervised ViT loaded via ``torch.hub``.

    DINOv2 descriptors are trained to be linearly separable for *image-level*
    semantics, which transfers far better to fine-grained identity matching than
    ImageNet-supervised features. The pretrained weights are a strong baseline even
    before any fine-tuning.
    """

    supports_tokens = True
    prefix_tokens = 1  # class token

    _HUB_ENTRY = {
        "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14"),
        "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14"),
        "dinov2_vitl14": ("facebookresearch/dinov2", "dinov2_vitl14"),
        "dinov2_vitg14": ("facebookresearch/dinov2", "dinov2_vitg14"),
    }

    def __init__(
        self,
        architecture: str = "dinov2_vitb14",
        pretrained: str | Path | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        torch = _torch()
        if architecture not in self._HUB_ENTRY:
            raise ModelError(f"Unsupported DINOv2 architecture: {architecture}")
        repo, entry = self._HUB_ENTRY[architecture]
        self.name = architecture
        try:
            self.network = torch.hub.load(repo, entry, trust_repo=True, verbose=False)
        except Exception as exc:
            raise ModelError(
                f"Could not load {architecture} from torch.hub ({repo}). An internet "
                f"connection is required the first time. Original error: {exc}"
            ) from exc

        local = Path(pretrained) if pretrained else None
        if local:
            if not local.is_file():
                raise ModelError(f"Checkpoint not found: {local}")
            state = torch.load(local, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {key.replace("backbone.", "").replace("module.", ""): value
                     for key, value in state.items()}
            missing, unexpected = self.network.load_state_dict(state, strict=False)
            LOGGER.info("Loaded %s checkpoint (%d missing, %d unexpected keys)",
                        architecture, len(missing), len(unexpected))

        self.feature_dim = int(self.network.embed_dim)
        self.patch_size = int(getattr(self.network, "patch_size", 14))
        self.gradient_checkpointing = gradient_checkpointing

    def __call__(self, images: Any) -> BackboneOutput:
        torch = _torch()

        def forward(tensor: Any) -> Any:
            return self.network.forward_features(tensor)

        if self.gradient_checkpointing and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            tokens = checkpoint(forward, images, use_reentrant=False)
        else:
            tokens = forward(images)
        return BackboneOutput(
            descriptor=tokens[:, 0],
            tokens=tokens,
            prefix_tokens=1,
        )


# ---------------------------------------------------------------------------
# DINOv2 / DINOv3 via timm (HuggingFace-hosted weights)
# ---------------------------------------------------------------------------
class TimmVitBackbone:
    """Transformer backbone loaded from timm, with HuggingFace as the weight source.

    Preferred over the ``torch.hub`` DINOv2 path in constrained environments:
    ``torch.hub`` pulls raw ``.pth`` files from GitHub, which is subject to unauthenticated
    API rate limits (``HTTP 403: rate limit exceeded``), whereas timm caches weights under
    ``~/.cache/huggingface``, supports resumption, and honours ``HF_ENDPOINT`` mirrors.

    ``forward_features`` on a ViT returns ``(B, 1 + N, D)`` with a leading class token,
    which is exactly the token layout the pooling layer expects.
    """

    supports_tokens = True

    def __init__(
        self,
        timm_name: str,
        pretrained: str | Path | None = None,
        load_pretrained: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ModelError(
                "timm is required for the DINOv2 backbone: pip install timm"
            ) from exc
        self.name = timm_name.replace("/", "_")
        local = Path(pretrained) if pretrained else None
        if local is not None and not local.is_file():
            raise ModelError(f"Checkpoint not found: {local}")
        try:
            self.network = timm.create_model(
                timm_name,
                pretrained=load_pretrained and local is None,
                num_classes=0,
                checkpoint_path=str(local) if local else "",
            )
            # DINOv2's published weights were trained at 518x518, but a face crop at
            # 224x224 carries ample identity signal at a quarter of the compute and a
            # quarter of the activation memory — decisive on a 6 GB GPU. Allowing dynamic
            # input size makes the position embeddings interpolate to whatever resolution
            # the pipeline asks for, instead of asserting on the training resolution.
            self._enable_dynamic_input_size()
        except Exception as exc:
            raise ModelError(
                f"Could not build {timm_name} from timm. The first run needs network "
                f"access to download weights (timm caches them under "
                f"~/.cache/huggingface). Original error: {exc}"
            ) from exc
        self.feature_dim = int(self.network.num_features)
        # ``patch_size`` is a 2-tuple on some timm versions (height, width) and a scalar
        # on others; normalise it so downstream code can rely on an int.
        raw_patch = getattr(getattr(self.network, "patch_embed", None), "patch_size", 14)
        self.patch_size = int(raw_patch[0]) if isinstance(raw_patch, (tuple, list)) else int(raw_patch)
        self.gradient_checkpointing = gradient_checkpointing

        # Not every timm ViT has a class token (some are patch-token only), so the token
        # layout is read off the instantiated model rather than assumed.
        self.prefix_tokens = int(getattr(self.network, "num_prefix_tokens", 1))

    def _enable_dynamic_input_size(self) -> None:
        """Accept input resolutions other than the pretrained one.

        DINOv2's published weights were trained at 518x518, and timm pins
        ``PatchEmbed.strict_img_size=True`` to that size, so a 224x224 face crop raises
        ``Input height (224) doesn't match model (518)``.

        The positional embedding is *not* resized here. timm 1.0.30's own
        ``dynamic_img_size`` branch (and its ``dynamic_img_pad`` variant) assume a
        channels-last ``B,H,W,C`` tensor, while ``PatchEmbed`` returns a channels-first
        ``B,C,H,W`` map and ``forward_features`` hands ``_pos_embed`` a 3-D ``B,N,C``
        sequence. Enabling that branch therefore raises
        ``ValueError: not enough values to unpack (expected 4, got 3)``. This class performs
        the interpolation itself in :meth:`_resample_pos_embed_if_needed`, from a fixed
        native starting point.
        """
        patch_embed = getattr(self.network, "patch_embed", None)
        if patch_embed is None:
            LOGGER.debug("%s has no patch_embed; leaving input size alone", self.name)
            return
        for attribute, value in (
            ("strict_img_size", False),
            ("dynamic_img_pad", False),
            ("img_size", None),
        ):
            if hasattr(patch_embed, attribute):
                setattr(patch_embed, attribute, value)
        # The registered parameter stays at its native resolution forever. The correctly
        # sized tensor for the current input lives in ``_pos_embed_cache`` and is swapped in
        # only for the duration of a forward pass. That makes interpolation idempotent by
        # construction and keeps ``state_dict`` shapes stable across resolutions.
        self._native_pos_embed = self.network.pos_embed
        self._pos_embed_cache: dict[tuple[int, int], Any] = {}
        self._in_resolution_scope = False
        native_tokens = int(self._native_pos_embed.shape[1])
        prefix_count = int(self.network.num_prefix_tokens)
        native_patches = native_tokens - prefix_count
        self._native_grid = int(round(math.sqrt(max(native_patches, 1))))
        if self._native_grid * self._native_grid != native_patches:
            self._native_grid = 0  # not a square grid; leave the embedding alone

    def _resample_pos_embed_if_needed(self, height: int, width: int) -> None:
        """Select (and cache) the positional embedding that matches the input resolution.

        The class-token row is preserved and only the patch rows are bicubically resampled,
        which is what DINOv2's own implementation does when the resolution changes.
        """
        patch_embed = getattr(self.network, "patch_embed", None)
        if patch_embed is None:
            return
        patch_h, patch_w = patch_embed.patch_size
        if height % patch_h or width % patch_w:
            raise ModelError(
                f"Input size {height}x{width} must be divisible by the patch size "
                f"{patch_h}x{patch_w}"
            )
        grid = (height // patch_h, width // patch_w)
        patch_embed.img_size = (height, width)

        if self._native_grid == 0:
            LOGGER.warning(
                "%s: positional embedding has a non-square patch grid; resolution changes "
                "are unsupported and the native embedding is used", self.name,
            )
            self._active_pos_embed = self._native_pos_embed
            return
        if grid == (self._native_grid, self._native_grid):
            self._active_pos_embed = self._native_pos_embed
            return

        cached = self._pos_embed_cache.get(grid)
        if cached is None:
            cached = self._resample_pos_embed(grid)
            self._pos_embed_cache[grid] = cached
        self._active_pos_embed = cached

    def _resample_pos_embed(self, grid: tuple[int, int]) -> Any:
        """Bicubically resample the *native* positional embedding to ``grid``.

        Always derived from ``self._native_pos_embed``, never from a previously resampled
        tensor, so two calls for the same grid produce identical results.
        """
        torch = _torch()
        base = self._native_pos_embed
        prefix_count = int(self.network.num_prefix_tokens)
        embed_dim = int(base.shape[2])
        prefix = base[:, :prefix_count]
        patches = base[:, prefix_count:]
        patches = patches.reshape(1, self._native_grid, self._native_grid, embed_dim)
        patches = patches.permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            patches, size=grid, mode="bicubic", align_corners=False
        )
        resized = resized.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], embed_dim)
        # `pos_embed` is a registered Parameter, so the swap-in must be a Parameter too.
        return torch.nn.Parameter(torch.cat([prefix, resized], dim=1), requires_grad=False)

    @contextmanager
    def resolution_scope(self, height: int, width: int) -> Iterator[None]:
        """Hold the correctly sized positional embedding for a whole forward *and* backward.

        Must wrap the backward pass whenever ``gradient_checkpointing`` is enabled: the
        checkpointed recompute re-runs the forward, so restoring the native embedding early
        makes the recomputation disagree with the graph that was recorded, which either
        raises a shape error or yields wrong gradients. Either way the optimiser step is
        discarded and the model silently stops learning.
        """
        self._resample_pos_embed_if_needed(height, width)
        original = self.network.pos_embed
        self.network.pos_embed = self._active_pos_embed
        self._in_resolution_scope = True
        try:
            yield
        finally:
            self.network.pos_embed = original
            self._in_resolution_scope = False

    def __call__(self, images: Any) -> BackboneOutput:
        height, width = int(images.shape[-2]), int(images.shape[-1])
        with self.resolution_scope(height, width):
            return self._forward(images)

    def _forward(self, images: Any) -> BackboneOutput:
        torch = _torch()  # used for checkpointing below

        def forward(tensor: Any) -> Any:
            return self.network.forward_features(tensor)

        if self.gradient_checkpointing and torch.is_grad_enabled():
            if not self._in_resolution_scope:
                # Failing loudly is essential: the alternative is a silently skipped
                # optimiser step and a training run that never converges.
                raise ModelError(
                    "gradient_checkpointing is enabled, so the positional-embedding scope "
                    "must stay open across the backward pass. Wrap the full step in "
                    "`backbone.resolution_scope(h, w)` (see catface.train.loop.Trainer)."
                )
            from torch.utils.checkpoint import checkpoint

            tokens = checkpoint(forward, images, use_reentrant=False)
        else:
            tokens = forward(images)
        if tokens.ndim == 4:
            # Some timm ViTs return a feature map rather than a token sequence.
            tokens = tokens.flatten(2).transpose(1, 2)
        return BackboneOutput(descriptor=tokens[:, 0], tokens=tokens, prefix_tokens=1)


#: timm model ids backing the named transformer architectures.
TIMM_ALIASES: dict[str, str] = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
}


# ---------------------------------------------------------------------------
# CLIP vision tower
# ---------------------------------------------------------------------------
class ClipBackbone:
    """CLIP ViT vision tower, used as a descriptor extractor.

    CLIP is included as a *contrastive* reference point: its training objective is
    image-text alignment, not instance discrimination, so it is a useful control
    when attributing gains to architecture versus objective.
    """

    supports_tokens = True
    prefix_tokens = 1  # class token

    _VARIANTS = {
        "clip_vitb32": ("ViT-B-32", "openai"),
        "clip_vitb16": ("ViT-B-16", "openai"),
        "clip_vitl14": ("ViT-L-14", "openai"),
    }

    def __init__(
        self,
        architecture: str = "clip_vitb32",
        pretrained: str | Path | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        torch = _torch()
        if architecture not in self._VARIANTS:
            raise ModelError(f"Unsupported CLIP architecture: {architecture}")
        model_name, tag = self._VARIANTS[architecture]
        self.name = architecture
        try:
            import open_clip
        except ImportError as exc:
            raise ModelError(
                "open_clip is required for the CLIP backbone: pip install open_clip_torch"
            ) from exc
        self.network, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=tag
        )
        if pretrained:
            local = Path(pretrained)
            if not local.is_file():
                raise ModelError(f"Checkpoint not found: {local}")
            state = torch.load(local, map_location="cpu", weights_only=False)
            self.network.load_state_dict(state.get("state_dict", state), strict=False)
        self.feature_dim = int(self.network.visual.output_dim)
        self.gradient_checkpointing = gradient_checkpointing

    def __call__(self, images: Any) -> BackboneOutput:
        visual = self.network.visual
        if hasattr(visual, "forward_intermediates") and False:  # pragma: no cover
            tokens = visual.forward_intermediates(images)
        # ``forward`` returns the final projection; tokens come from the transformer.
        tokens = visual.transformer(images)
        # open_clip returns a single tensor when there is no class token.
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        return BackboneOutput(descriptor=tokens[:, 0], tokens=tokens, prefix_tokens=1)


# ---------------------------------------------------------------------------
# timm (escape hatch for any architecture)
# ---------------------------------------------------------------------------
class TimmBackbone:
    """Any ``timm`` model, used as a forward-compatible escape hatch."""

    supports_tokens = True

    def __init__(
        self,
        timm_name: str,
        pretrained: str | Path | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ModelError("timm is required for the timm backbone: pip install timm") from exc
        self.name = timm_name.replace("/", "_")
        self.network = timm.create_model(
            timm_name,
            pretrained=not bool(pretrained),
            num_classes=0,
            checkpoint_path=str(pretrained) if pretrained else "",
        )
        self.feature_dim = int(self.network.num_features)
        self.gradient_checkpointing = gradient_checkpointing
        # ``timm`` reports the class/register-token count on ViTs. When the attribute is
        # absent the layout is decided by the *shape* of ``forward_features`` output (see
        # ``__call__``), not by guessing from the module tree — timm's ResNet exposes neither
        # ``num_prefix_tokens`` nor ``features`` and would otherwise look like a ViT.
        self.prefix_tokens = int(getattr(self.network, "num_prefix_tokens", 0))
        self._layout_known = "num_prefix_tokens" in getattr(self.network, "__dict__", {}) or hasattr(
            self.network, "num_prefix_tokens"
        )

    def __call__(self, images: Any) -> BackboneOutput:
        tokens = self.network.forward_features(images)
        if tokens.ndim == 4:  # CNN feature map
            self.prefix_tokens = 0
            pooled = tokens.mean(dim=(2, 3))
            return BackboneOutput(descriptor=pooled, tokens=tokens.flatten(2).transpose(1, 2),
                                  prefix_tokens=0)
        if not self._layout_known:
            self.prefix_tokens = 1
            self._layout_known = True
        return BackboneOutput(
            descriptor=tokens[:, 0], tokens=tokens, prefix_tokens=self.prefix_tokens
        )


_FACTORIES: dict[str, Callable[..., Backbone]] = {
    "resnet18": lambda **kw: TorchvisionBackbone("resnet18", **{k: v for k, v in kw.items() if k == "pretrained"}),
    "resnet34": lambda **kw: TorchvisionBackbone("resnet34", **{k: v for k, v in kw.items() if k == "pretrained"}),
    "resnet50": lambda **kw: TorchvisionBackbone("resnet50", **{k: v for k, v in kw.items() if k == "pretrained"}),
    "resnet101": lambda **kw: TorchvisionBackbone("resnet101", **{k: v for k, v in kw.items() if k == "pretrained"}),
    "wide_resnet50_2": lambda **kw: TorchvisionBackbone("wide_resnet50_2", **{k: v for k, v in kw.items() if k == "pretrained"}),
    # DINOv2 weights come from timm/HuggingFace; see TimmVitBackbone for why.
    **{
        alias: (lambda name: lambda **kw: TimmVitBackbone(
            TIMM_ALIASES[name],
            pretrained=kw.get("pretrained"),
            gradient_checkpointing=kw.get("gradient_checkpointing", False),
        ))(alias)
        for alias in TIMM_ALIASES
    },
    "clip_vitb32": lambda **kw: ClipBackbone("clip_vitb32", **{k: v for k, v in kw.items() if k == "pretrained"}),
}


def backbone_family(name: str) -> str:
    """Group a backbone into ``cnn`` / ``vit`` / ``clip`` / ``other`` for reporting."""
    if name in ("resnet18", "resnet34", "resnet50", "resnet101", "wide_resnet50_2"):
        return "cnn"
    if name in TIMM_ALIASES:
        return "vit"
    if name.startswith("clip_"):
        return "clip"
    return "other"


def available_backbones() -> list[str]:
    """Names accepted by :func:`build_backbone`."""
    return sorted(set(_FACTORIES) | {"timm"})


def build_backbone(
    name: str,
    pretrained: str | Path | None = None,
    timm_name: str | None = None,
    gradient_checkpointing: bool = False,
) -> Backbone:
    """Instantiate a backbone by registry name."""
    if name == "timm":
        if not timm_name:
            raise ModelError("timm_name is required when backbone == 'timm'")
        return TimmBackbone(timm_name, pretrained=pretrained,
                            gradient_checkpointing=gradient_checkpointing)
    if name not in _FACTORIES:
        raise ModelError(f"Unknown backbone {name!r}. Available: {available_backbones()}")
    kwargs: dict[str, Any] = {"pretrained": pretrained}
    if name.startswith("dinov2"):
        kwargs["gradient_checkpointing"] = gradient_checkpointing
    return _FACTORIES[name](**kwargs)


__all__ = [
    "TIMM_ALIASES",
    "Backbone",
    "BackboneOutput",
    "available_backbones",
    "backbone_family",
    "build_backbone",
]
