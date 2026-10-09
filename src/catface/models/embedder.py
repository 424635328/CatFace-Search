"""The :class:`Embedder` — one facade for training and inference.

Everything that turns pixels into a unit-norm identity descriptor lives here, so
training and scoring cannot drift apart. The most common real-world bug in a
retrieval system is an *inference/training mismatch*: preprocessing, resolution,
or normalisation differing between the two paths. ``Embedder`` is the single
implementation of that path, and :meth:`Embedder.describe_config` records exactly
which variant produced a given set of vectors.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ArtifactError, ModelError
from ..logging_utils import get_logger
from .backbone import Backbone, BackboneOutput, build_backbone
from .heads import MetricHead
from .pooling import GeM, pool_tokens, pooled_dim, resolve_pooling

LOGGER = get_logger("models.embedder")

#: ImageNet statistics — the convention every backbone here was trained under.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CHECKPOINT_VERSION = 2


def _torch():
    import torch

    return torch


@dataclass
class EmbedderConfig:
    """Everything needed to rebuild an embedder bit-for-bit."""

    backbone: str = "dinov2_vitb14"
    pretrained: str | None = None
    timm_name: str | None = None
    embedding_dim: int = 512
    pooling: str = "auto"
    head: str = "arcface"
    head_margin: float = 0.35
    head_scale: float = 64.0
    head_num_subcenters: int = 3
    image_size: int = 224
    l2_normalize: bool = True
    tta: tuple[str, ...] = ("identity", "hflip")
    gradient_checkpointing: bool = False
    checkpoint_version: int = CHECKPOINT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone,
            "pretrained": self.pretrained,
            "timm_name": self.timm_name,
            "embedding_dim": self.embedding_dim,
            "pooling": self.pooling,
            "head": self.head,
            "head_margin": self.head_margin,
            "head_scale": self.head_scale,
            "head_num_subcenters": self.head_num_subcenters,
            "image_size": self.image_size,
            "l2_normalize": self.l2_normalize,
            "tta": list(self.tta),
            "gradient_checkpointing": self.gradient_checkpointing,
            "checkpoint_version": self.checkpoint_version,
        }


def _reconcile_pos_embed(network: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Resample an incoming `pos_embed` to the grid the network currently expects.

    Positional embeddings for ViTs are resolution dependent: a checkpoint written after a
    224x224 forward holds 257 rows (1 class + 16x16 patches) while the pristine model holds
    1370 (1 + 37x37 for the 518x518 training resolution). Both describe the same weights, so
    the tensor is interpolated rather than rejected.
    """
    key = "pos_embed"
    if key not in state or not hasattr(network, "pos_embed"):
        return state
    incoming = state[key]
    target = network.pos_embed
    if tuple(incoming.shape) == tuple(target.shape):
        return state

    torch = _torch()
    prefix = int(getattr(network, "num_prefix_tokens", 1))
    incoming_tokens = int(incoming.shape[1])
    target_tokens = int(target.shape[1])
    incoming_patches = incoming_tokens - prefix
    target_patches = target_tokens - prefix
    incoming_grid = round(math.sqrt(incoming_patches))
    target_grid = round(math.sqrt(target_patches))
    if (
        incoming_patches <= 0
        or incoming_grid * incoming_grid != incoming_patches
        or target_grid * target_grid != target_patches
    ):
        # Not a square patch grid: drop it and let the model keep its own embedding.
        LOGGER.warning(
            "Cannot reconcile pos_embed shapes %s -> %s; keeping model weights",
            tuple(incoming.shape),
            tuple(target.shape),
        )
        state = dict(state)
        state.pop(key, None)
        return state

    embed_dim = int(incoming.shape[2])
    head = incoming[:, :prefix]
    patches = incoming[:, prefix:].reshape(1, incoming_grid, incoming_grid, embed_dim)
    patches = patches.permute(0, 3, 1, 2)
    resized = torch.nn.functional.interpolate(
        patches, size=(target_grid, target_grid), mode="bicubic", align_corners=False
    )
    resized = resized.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, embed_dim)
    state = dict(state)
    state[key] = torch.cat([head, resized], dim=1)
    LOGGER.info(
        "Resampled checkpoint pos_embed from %dx%d to %dx%d patches",
        incoming_grid,
        incoming_grid,
        target_grid,
        target_grid,
    )
    return state


class Embedder:
    """Backbone + pooling + metric head, usable for training and scoring."""

    def __init__(
        self,
        config: EmbedderConfig,
        num_classes: int = 0,
        device: str = "cpu",
    ) -> None:
        torch = _torch()
        self.config = config
        self.device = torch.device(device)

        self.backbone: Backbone = build_backbone(
            config.backbone,
            pretrained=config.pretrained,
            timm_name=config.timm_name,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        # Resolve "auto" now that the backbone's capabilities are known, so the pooling
        # width used by the head matches the pooling actually applied at forward time.
        # The backbone declares its token layout rather than this class guessing from the
        # module tree (timm's ResNet exposes no `features` attribute, which made the
        # heuristic report a class token that does not exist).
        self.pooling = resolve_pooling(
            config.pooling,
            has_class_token=int(getattr(self.backbone, "prefix_tokens", 0)) > 0,
        )
        self.gem = GeM() if self.pooling == "gem" else None
        pooled = pooled_dim(self.pooling, self.backbone.feature_dim)

        self.head = MetricHead(
            in_features=pooled,
            num_classes=num_classes,
            embedding_dim=config.embedding_dim,
            kind=config.head,
            margin=config.head_margin,
            scale=config.head_scale,
            num_subcenters=config.head_num_subcenters,
        )
        self.to(self.device)

    # -- module plumbing ----------------------------------------------------
    def _modules(self) -> list[Any]:
        return [self.backbone]  # type: ignore[list-item]

    def to(self, device: Any) -> Embedder:
        """Move every learnable tensor to ``device`` and record it.

        The head holds a ``BatchNorm1d`` (and optionally a projection), so moving only
        the backbone leaves part of the model on the CPU and produces a device-mismatch
        error inside the normalisation layer.
        """
        torch = _torch()
        self.device = torch.device(device)
        for module in self._network_modules():
            module.to(self.device)
        self.head.module.to(self.device)
        if self.gem is not None:
            self.gem.p = torch.nn.Parameter(
                self.gem.p.data.to(self.device), requires_grad=self.gem.p.requires_grad
            )
        return self

    def _network_modules(self) -> list[Any]:
        """The ``nn.Module`` objects held by the backbone, for device moves."""
        modules = []
        network = getattr(self.backbone, "network", None)
        if network is not None:
            modules.append(network)
        features = getattr(self.backbone, "features", None)
        if features is not None:
            modules.append(features)
        for attr in ("pool",):
            value = getattr(self.backbone, attr, None)
            if value is not None:
                modules.append(value)
        return modules

    def train(self, mode: bool = True) -> Embedder:
        for module in self._network_modules():
            module.train(mode)
        self.head.train(mode)
        return self

    def eval(self) -> Embedder:
        return self.train(False)

    def parameters(self) -> list[Any]:
        parameters: list[Any] = []
        for module in self._network_modules():
            parameters.extend([p for p in module.parameters() if p.requires_grad])
        parameters.extend(self.head.parameters())
        if self.gem is not None:
            parameters.extend(list(self.gem.parameters()))
        return parameters

    def named_parameter_groups(
        self,
        lr: float,
        backbone_lr_scale: float = 1.0,
        weight_decay: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Split parameters into backbone (scaled LR) and head groups.

        Fine-tuning a pretrained backbone at the same rate as a randomly initialised
        head destroys the pretrained features in the first few hundred steps;
        a lower backbone LR is what makes transfer learning work on a small corpus.
        """
        backbone_ids = set()
        backbone_params: list[Any] = []
        for module in self._network_modules():
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in backbone_ids:
                    backbone_ids.add(id(parameter))
                    backbone_params.append(parameter)

        head_params = [p for p in self.head.parameters() if p.requires_grad]
        if self.gem is not None:
            head_params.extend([p for p in self.gem.parameters() if p.requires_grad])

        groups = [
            {
                "params": backbone_params,
                "lr": lr * backbone_lr_scale,
                "weight_decay": weight_decay,
                "name": "backbone",
            }
        ]
        # Biases and normalisation gains are conventionally not decayed.
        decay, no_decay = [], []
        for parameter in head_params:
            (no_decay if parameter.ndim <= 1 else decay).append(parameter)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay, "name": "head_decay"})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": "head_nodecay"})
        return groups

    # -- forward ------------------------------------------------------------
    @contextmanager
    def resolution_scope(self, height: int, width: int) -> Iterator[None]:
        """Delegate to the backbone so a training loop can hold the embedding scope open.

        Needed because the scope must outlive the forward pass: with gradient checkpointing
        the backward recomputes the forward, and a scope that closed early would corrupt it.
        Backbones without positional embeddings simply yield.
        """
        scope = getattr(self.backbone, "resolution_scope", None)
        if scope is None:
            yield
            return
        with scope(height, width):
            yield

    def pool_features(self, images: Any) -> Any:
        """Backbone output pooled to ``(B, pooled_dim)``, **not** projected.

        The distinction between this and :meth:`embed` is load-bearing. ``head.embed``
        applies BatchNorm + projection + L2-normalisation, so a training loop that fed
        :meth:`embed`'s output back into ``head(...)`` would project twice, and the head's
        ``BatchNorm1d`` — sized for the pooled width — would raise
        ``running_mean should contain <pooled_dim> elements not <embedding_dim>``.
        """
        images = self._to_device(images)
        output: BackboneOutput = self.backbone(images)
        if output.tokens is not None:
            return pool_tokens(output.tokens, self.pooling, prefix_tokens=output.prefix_tokens, gem=self.gem)
        return output.descriptor

    def forward(self, images: Any, labels: Any | None = None) -> Any:
        """Full training path: ``images -> logits``."""
        return self.head(self.pool_features(images), labels)

    # ``Embedder`` owns an ``nn.Module`` rather than being one, so ``forward`` must be
    # wired to ``__call__`` explicitly; otherwise the object is not callable.
    __call__ = forward

    def _to_device(self, tensor: Any) -> Any:
        """Move an input batch onto the embedder's device.

        Done here rather than at every call site: one forgotten ``.to(device)`` produces
        a confusing "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)"
        error, and it is the kind of omission that only shows up on GPU machines.
        """
        if tensor is None or not hasattr(tensor, "to"):
            return tensor
        if getattr(tensor, "device", None) == self.device:
            return tensor
        return tensor.to(self.device)

    def embed(self, images: Any) -> Any:
        """Inference path: ``images -> unit-norm descriptor``.

        This is a *pure inference* entry point: it forces evaluation mode and disables
        autograd, restoring the previous mode on exit. Two real failures this prevents:

        * the head's ``BatchNorm1d`` raises ``Expected more than 1 value per channel``
          when a caller embeds a single image while the model is still in train mode;
        * descriptors computed with dropout active are not reproducible, so the same
          image would produce different vectors on different calls.
        """
        torch = _torch()
        was_training = self.head.module.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad():
                embeddings = self.head.embed(self.pool_features(images))
                if self.config.l2_normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
                return embeddings
        finally:
            if was_training:
                self.train(True)

    # -- checkpointing ------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Flatten every learnable tensor into one dict with stable prefixes."""
        state: dict[str, Any] = {}
        network = getattr(self.backbone, "network", None)
        if network is not None:
            for key, value in network.state_dict().items():
                state[f"backbone.{key}"] = value
        features = getattr(self.backbone, "features", None)
        if features is not None:
            for key, value in features.state_dict().items():
                state[f"features.{key}"] = value
        for key, value in self.head.state_dict().items():
            state[key] = value
        if self.gem is not None:
            state["gem.p"] = self.gem.p.detach().clone()
        return state

    def save(self, path: str | Path, extra: Mapping[str, Any] | None = None) -> Path:
        """Persist a self-describing checkpoint (config + weights)."""
        torch = _torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CHECKPOINT_VERSION,
            "embedder_config": self.config.to_dict(),
            "num_classes": self.head.num_classes,
            "class_names": getattr(self, "class_names", None),
            "state_dict": self.state_dict(),
            "extra": dict(extra or {}),
        }
        torch.save(payload, target)
        LOGGER.info("Saved checkpoint to %s", target)
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        strict: bool = False,
    ) -> Embedder:
        """Rebuild an embedder from a checkpoint produced by :meth:`save`."""
        torch = _torch()
        source = Path(path)
        if not source.is_file():
            raise ArtifactError(f"Checkpoint not found: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "embedder_config" not in payload:
            raise ArtifactError(
                f"{source} is not a catface checkpoint (expected keys 'embedder_config' and 'state_dict')"
            )
        version = payload.get("version")
        if version != CHECKPOINT_VERSION:
            raise ArtifactError(
                f"{source} has checkpoint version {version}, this build expects {CHECKPOINT_VERSION}"
            )
        config = EmbedderConfig(**payload["embedder_config"])
        embedder = cls(config, num_classes=int(payload.get("num_classes", 0)), device=device)
        missing, unexpected = cls._load_split(embedder, payload["state_dict"], strict)
        if missing:
            LOGGER.warning("Checkpoint missing %d key(s), e.g. %s", len(missing), missing[:3])
        if unexpected:
            LOGGER.warning("Checkpoint has %d unexpected key(s), e.g. %s", len(unexpected), unexpected[:3])
        embedder.class_names = payload.get("class_names")
        embedder.eval()
        return embedder

    @staticmethod
    def _load_split(
        embedder: Embedder,
        state: Mapping[str, Any],
        # Kept as ``strict`` rather than renamed to ``_strict`` because both call sites pass it
        # as a keyword argument; renaming the parameter silently breaks them at runtime, which
        # is exactly what happened when this was first "fixed" to satisfy the linter.
        strict: bool,  # noqa: ARG004 - part of the call contract, not an unused argument
    ) -> tuple[list[str], list[str]]:
        """Load backbone and head weights separately, tolerating absent groups."""
        head_state = {k: v for k, v in state.items() if k.startswith("head.")}
        gem_state = {k: v for k, v in state.items() if k == "gem.p"}
        rest = {k: v for k, v in state.items() if k not in head_state and k not in gem_state}

        missing: list[str] = []
        unexpected: list[str] = []
        network = getattr(embedder.backbone, "network", None)
        if network is not None:
            stripped = {k.split("backbone.", 1)[1]: v for k, v in rest.items() if k.startswith("backbone.")}
            # `pos_embed` has as many rows as the resolution used during the last forward
            # pass, so its shape can legitimately differ between the checkpoint and the
            # freshly built model. Resample the incoming grid instead of failing the load.
            stripped = _reconcile_pos_embed(network, stripped)
            result = network.load_state_dict(stripped, strict=False)
            missing += list(result.missing_keys)
            unexpected += list(result.unexpected_keys)
        features = getattr(embedder.backbone, "features", None)
        if features is not None:
            stripped = {k.split("features.", 1)[1]: v for k, v in rest.items() if k.startswith("features.")}
            result = features.load_state_dict(stripped, strict=False)
            missing += list(result.missing_keys)
            unexpected += list(result.unexpected_keys)
        if head_state:
            result = embedder.head.module.load_state_dict(
                {k.split("head.", 1)[1]: v for k, v in head_state.items()}, strict=False
            )
            missing += list(result.missing_keys)
            unexpected += list(result.unexpected_keys)
        if gem_state and embedder.gem is not None:
            embedder.gem.p = _torch().nn.Parameter(gem_state["gem.p"].to(embedder.device))
        return missing, unexpected

    def load_state_dict_into(
        self,
        state: Mapping[str, Any],
        skip_prefixes: Sequence[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Load a flat state dict back into this embedder.

        Used to restore the best-validation weights at the end of training, by the EMA
        bookkeeping, and by resume. Returns ``(missing, unexpected)`` so callers can log
        rather than silently accept a partial load.

        Args:
            state: Flat state dict as produced by :meth:`state_dict`.
            skip_prefixes: Key prefixes to ignore entirely. Needed when the caller has
                deliberately accepted a state from a different class space: the margin
                head's weight matrix cannot fit this model, but the backbone still
                transfers and re-initialising the head is the intended outcome.
        """
        if skip_prefixes:
            state = {
                key: value
                for key, value in state.items()
                if not any(key.startswith(prefix) for prefix in skip_prefixes)
            }
        return self._load_split(self, state, strict=False)

    def describe_config(self) -> dict[str, Any]:
        """Human- and machine-readable description stamped onto every artifact."""
        return {
            **self.config.to_dict(),
            "backbone_feature_dim": int(self.backbone.feature_dim),
            "num_classes": int(self.head.num_classes),
            "device": str(self.device),
        }


# ---------------------------------------------------------------------------
# Batch embedding with TTA
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingResult:
    """Descriptors for a batch of images, in input order."""

    vectors: np.ndarray
    ids: list[str] = field(default_factory=list)
    tta_views: int = 1
    disagreed: list[str] = field(default_factory=list)
    """Ids whose TTA views disagreed strongly — a useful noise indicator."""


TTATransform = Any


def build_tta_transforms(
    image_size: int, views: Sequence[str] = ("identity", "hflip")
) -> list[tuple[str, TTATransform]]:
    """Build the geometric views averaged for one descriptor.

    Only geometry is augmented: photometric augmentation would change the descriptor
    in a way that has nothing to do with identity, and colour jitter at test time
    measurably hurts face matching.
    """
    import torchvision.transforms as T  # noqa: N812 - T is the torchvision convention

    base = [
        T.Resize(int(image_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
    transforms: list[tuple[str, TTATransform]] = []
    for view in views:
        if view == "identity":
            transforms.append((view, T.Compose(base)))
        elif view == "hflip":
            transforms.append((view, T.Compose([*base, T.RandomHorizontalFlip(p=1.0)])))
        elif view == "scale_112":
            transforms.append(
                (
                    view,
                    T.Compose(
                        [
                            T.Resize(int(image_size * 1.4), interpolation=T.InterpolationMode.BICUBIC),
                            T.CenterCrop(image_size),
                            T.ToTensor(),
                            T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
                        ]
                    ),
                )
            )
        else:
            raise ModelError(f"Unknown TTA view: {view!r}")
    return transforms


def embed_records(
    embedder: Embedder,
    paths: Sequence[str | Path],
    image_size: int | None = None,
    batch_size: int = 32,
    views: Sequence[str] | None = None,
    _num_workers: int = 0,
    normalize: bool = True,
) -> EmbeddingResult:
    """Embed a list of image files with TTA, in batches.

    Failures are reported per file rather than aborting the run: on real corpora a
    handful of unreadable images is normal, and silently dropping them would make
    index positions disagree with the manifest.
    """
    torch = _torch()
    from PIL import Image

    size = image_size or embedder.config.image_size
    view_list = list(views or embedder.config.tta)
    transforms = build_tta_transforms(size, view_list)

    stacked: list[np.ndarray] = []
    kept_ids: list[str] = []
    disagreed: list[str] = []
    was_training = embedder.head.module.training
    embedder.eval()

    with torch.no_grad():
        batch_tensors: list[Any] = []
        batch_ids: list[str] = []
        disk_cache: dict[str, Any] = {}

        def flush() -> None:
            nonlocal batch_tensors, batch_ids, disk_cache
            if not batch_tensors:
                return
            # (V, B, C, H, W) -> average descriptor over views after L2 normalisation,
            # so no single view can dominate the mean by magnitude.
            per_view = []
            for view_index in range(len(view_list)):
                tensor = torch.stack([views_[view_index] for views_ in batch_tensors]).to(embedder.device)
                vectors = embedder.embed(tensor)
                if normalize:
                    vectors = torch.nn.functional.normalize(vectors, dim=1)
                per_view.append(vectors)
            stack = torch.stack(per_view)  # (V, B, D)
            mean = stack.mean(dim=0)
            if normalize:
                mean = torch.nn.functional.normalize(mean, dim=1)
            if len(view_list) > 1:
                agreement = (stack * mean.unsqueeze(0)).sum(dim=2).min(dim=0).values
                for identifier, score in zip(batch_ids, agreement.detach().cpu().tolist()):
                    if score < 0.6:
                        disagreed.append(identifier)
            stacked.append(mean.detach().cpu().numpy().astype(np.float32))
            batch_tensors = []
            batch_ids = []
            disk_cache = {}

        for path in paths:
            key = str(path)
            if key in disk_cache:
                continue
            try:
                with Image.open(path) as handle:
                    image = handle.convert("RGB")
                    batch_tensors.append([transform(image) for _, transform in transforms])
                    batch_ids.append(key)
            except Exception as exc:
                LOGGER.warning("Could not read %s: %s", key, exc)
                continue
            kept_ids.append(key)
            if len(batch_tensors) >= batch_size:
                flush()
        flush()

    vectors = np.vstack(stacked) if stacked else np.zeros((0, 0), dtype=np.float32)
    if len(kept_ids) != len(vectors):
        # Keep the contract explicit: ids and vectors are positionally aligned.
        raise ArtifactError(f"Embedding bookkeeping mismatch: {len(kept_ids)} ids for {len(vectors)} vectors")
    if was_training:
        embedder.train(True)
    return EmbeddingResult(vectors=vectors, ids=kept_ids, tta_views=len(view_list), disagreed=disagreed)


__all__ = [
    "CHECKPOINT_VERSION",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "Embedder",
    "EmbedderConfig",
    "EmbeddingResult",
    "build_tta_transforms",
    "embed_records",
]
