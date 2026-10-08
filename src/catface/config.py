"""Typed configuration objects — the single source of truth for every run.

Design notes
------------
* Every tunable lives in a dataclass with an explicit type and default. YAML is
  parsed *into* these objects and unknown keys are rejected loudly: a silently
  ignored typo in a config file is the classic source of "the experiment did not
  do what the report claims".
* Paths are normalised to absolute :class:`pathlib.Path` so that a config can be
  executed from any working directory (the legacy scripts all assumed CWD).
* A config is serialisable back to JSON/YAML, and its hash is stamped onto
  artifacts, so an index can always be traced to the exact recipe that built it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .errors import ConfigError

T = TypeVar("T")

VALID_BACKBONES = (
    "resnet50",
    "resnet101",
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "clip_vitb32",
    "timm",
)

VALID_HEADS = ("identity", "linear", "arcface", "cosface", "subcenter_arcface")
VALID_LOSSES = ("arcface", "cosface", "subcenter_arcface", "triplet", "ce", "none")
VALID_POOLING = ("auto", "cls", "gap", "gem", "cls_gap")


def _resolve(path: str | Path) -> Path:
    """Return an absolute path without requiring the target to exist."""
    return Path(path).expanduser().resolve()


def _encode(value: Any) -> Any:
    """Recursively convert a config value into JSON/YAML-serialisable primitives.

    Dataclass sections must be unwrapped by iterating their *fields*: ``dataclasses.asdict``
    applied to a root object returns the nested dataclass instances themselves rather
    than dictionaries, so a naive implementation produces a fingerprint that compares
    equal for every pair of configs and therefore never detects a change.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in value.items()}
    return value


class _ConfigSection:
    """Mixin giving every config section the same serialisation helpers."""

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this section."""
        return _encode({f.name: getattr(self, f.name) for f in fields(self)})  # type: ignore[arg-type]

    # ``to_dict`` is provided as an alias: it reads more naturally at call sites and
    # matches the naming used in reports.
    def to_dict(self) -> dict[str, Any]:
        return self.to_mapping()


def _require_keys(data: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {unknown} in {where}. Allowed: {sorted(allowed)}"
        )


@dataclass
class DataConfig(_ConfigSection):
    """Where images live and how crops are produced."""
    root: Path = Path("data")
    manifest: Path = Path("data/manifests")
    crops_dir: Path = Path("data/faces")
    tile: int = 256
    """Square output size for every stored face crop (pixels)."""
    image_size: int = 224
    """Model input resolution; may differ from ``tile`` (resized at load time)."""
    min_face_px: int = 48
    """Crops whose shorter side is below this are rejected as unusable."""
    head_ratio: float = 0.34
    """Legacy heuristic: fraction down from the box top to the face centre."""
    pad_ratio: float = 0.35
    """Context margin added around the detected face region."""
    alignment: str = "none"
    """``none`` | ``landmark_similarity`` — geometric face normalisation."""
    blur_var_threshold: float = 18.0
    """Variance-of-Laplacian floor; below this a crop is flagged ``blurry``."""
    luminance_range: tuple[int, int] = (25, 235)
    """Acceptable mean-luminance band for a usable crop."""
    dedupe: bool = True
    """Drop byte-identical crops so duplicated files cannot inflate recall."""
    num_workers: int = 4
    seed: int = 1337

    def __post_init__(self) -> None:
        self.root = _resolve(self.root)
        self.manifest = _resolve(self.manifest)
        self.crops_dir = _resolve(self.crops_dir)
        if self.image_size <= 0 or self.tile <= 0:
            raise ConfigError("data.image_size and data.tile must be positive")
        if not 0.0 <= self.pad_ratio <= 2.0:
            raise ConfigError("data.pad_ratio must be within [0, 2]")
        if self.alignment not in ("none", "landmark_similarity"):
            raise ConfigError(f"Unsupported alignment: {self.alignment!r}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DataConfig:
        _require_keys(data, [f.name for f in fields(cls)], "data")
        payload = dict(data)
        if "luminance_range" in payload:
            payload["luminance_range"] = tuple(payload["luminance_range"])
        return cls(**payload)


@dataclass
class ModelConfig(_ConfigSection):
    """Which network produces embeddings and how it is initialised."""

    backbone: str = "dinov2_vitb14"
    pretrained: str | Path | None = None
    """Local checkpoint path, or ``None`` to fetch published weights."""
    timm_name: str | None = None
    """Required when ``backbone == 'timm'``."""
    embedding_dim: int = 512
    pooling: str = "auto"
    head: str = "identity"
    head_num_subcenters: int = 3
    head_margin: float = 0.35
    head_scale: float = 64.0
    image_size: int = 224
    freeze_backbone: bool = False
    gradient_checkpointing: bool = False
    l2_normalize: bool = True
    tta: tuple[str, ...] = ("identity", "hflip")
    """Test-time views averaged into one descriptor."""

    def __post_init__(self) -> None:
        if isinstance(self.pretrained, str):
            self.pretrained = _resolve(self.pretrained)
        if self.backbone not in VALID_BACKBONES:
            raise ConfigError(
                f"model.backbone must be one of {VALID_BACKBONES}, got {self.backbone!r}"
            )
        if self.backbone == "timm" and not self.timm_name:
            raise ConfigError("model.timm_name is required when backbone == 'timm'")
        if self.pooling not in VALID_POOLING:
            raise ConfigError(f"model.pooling must be one of {VALID_POOLING}")
        if self.head not in VALID_HEADS:
            raise ConfigError(f"model.head must be one of {VALID_HEADS}")
        if self.embedding_dim <= 0:
            raise ConfigError("model.embedding_dim must be positive")
        self.tta = tuple(self.tta)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ModelConfig:
        _require_keys(data, [f.name for f in fields(cls)], "model")
        payload = dict(data)
        if "tta" in payload:
            payload["tta"] = tuple(payload["tta"])
        return cls(**payload)


@dataclass
class TrainConfig(_ConfigSection):
    """Metric-learning training recipe."""

    loss: str = "arcface"
    epochs: int = 30
    batch_size: int = 64
    optimizer: str = "adamw"
    lr: float = 3e-4
    backbone_lr_scale: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    scheduler: str = "cosine"
    label_smoothing: float = 0.05
    triplet_weight: float = 0.3
    triplet_margin: float = 0.3
    amp: bool = True
    grad_clip: float = 10.0
    ema_decay: float = 0.0
    """``>0`` keeps an exponential-moving-average copy of the weights."""
    samples_per_identity: int = 4
    """PK sampling: images per identity per batch."""
    identities_per_batch: int = 16
    """PK sampling: identities per batch."""
    val_every: int = 1
    early_stop_patience: int = 8
    num_workers: int = 4
    seed: int = 1337
    output_dir: Path = Path("artifacts/train")

    def __post_init__(self) -> None:
        if self.loss not in VALID_LOSSES:
            raise ConfigError(f"train.loss must be one of {VALID_LOSSES}")
        if self.epochs < 1:
            raise ConfigError("train.epochs must be >= 1")
        if self.batch_size < 2:
            raise ConfigError("train.batch_size must be >= 2")
        self.output_dir = _resolve(self.output_dir)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrainConfig:
        _require_keys(data, [f.name for f in fields(cls)], "train")
        return cls(**data)


@dataclass
class EvalConfig(_ConfigSection):
    """Protocol and metrics for benchmarking."""

    protocol: str = "identity_split"
    """``identity_split`` (closed/open-set retrieval) | ``verification_pairs``."""
    recall_ks: tuple[int, ...] = (1, 5, 10)
    far_targets: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    same_identity_excluded: bool = True
    """Exclude the query's own gallery entry (standard re-ID protocol)."""
    gallery_ratio: float = 1.0
    """Fraction of non-query images used as gallery (1.0 = full protocol)."""
    bootstrap_samples: int = 500
    """Bootstrap resamples for confidence intervals on the headline metric."""
    whiten: str = "none"
    """``none`` | ``pca`` | ``pcaw`` — descriptor post-processing fitted on gallery."""
    whiten_dim: int = 0
    """Target dimension for whitening; ``0`` keeps the input dimension."""
    query_expansion: str = "none"
    """``none`` | ``aqe`` | ``dba`` — descriptor-side augmentation."""

    def __post_init__(self) -> None:
        self.recall_ks = tuple(int(k) for k in self.recall_ks)
        self.far_targets = tuple(float(f) for f in self.far_targets)
        if any(k < 1 for k in self.recall_ks):
            raise ConfigError("eval.recall_ks must all be >= 1")
        if self.whiten not in ("none", "pca", "pcaw"):
            raise ConfigError("eval.whiten must be one of none|pca|pcaw")
        if self.query_expansion not in ("none", "aqe", "dba"):
            raise ConfigError("eval.query_expansion must be one of none|aqe|dba")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EvalConfig:
        _require_keys(data, [f.name for f in fields(cls)], "eval")
        payload = dict(data)
        for key in ("recall_ks", "far_targets"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)


@dataclass
class IndexConfig(_ConfigSection):
    """How the vector index is built and searched."""

    kind: str = "flat_ip"
    """``flat_ip`` | ``ivf_pq`` | ``hnsw`` | ``numpy``."""
    nlist: int = 1024
    nprobe: int = 32
    m_pq: int = 32
    nbits: int = 8
    hnsw_m: int = 32
    ef_search: int = 64
    output_dir: Path = Path("artifacts/index")

    def __post_init__(self) -> None:
        if self.kind not in ("flat_ip", "ivf_pq", "hnsw", "numpy"):
            raise ConfigError(f"Unsupported index.kind: {self.kind!r}")
        self.output_dir = _resolve(self.output_dir)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> IndexConfig:
        _require_keys(data, [f.name for f in fields(cls)], "index")
        return cls(**data)


@dataclass
class PipelineConfig(_ConfigSection):
    """Top-level configuration aggregating every stage."""

    run_name: str = "catface-run"
    seed: int = 1337
    device: str = "auto"
    """``auto`` | ``cpu`` | ``cuda`` | ``cuda:N``."""
    log_level: str = "INFO"
    output_dir: Path = Path("artifacts")
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    extra: dict[str, Any] = field(default_factory=dict)
    """Free-form carry-through values (notes, ticket ids) recorded verbatim."""

    def __post_init__(self) -> None:
        # Captured before any path is resolved, because ``_resolve`` replaces the relative
        # form with an absolute one. ``fingerprint`` needs the base to recover the relative
        # relationship, so that the same config keeps one identity across directories.
        if not hasattr(self, "_base_dir"):
            object.__setattr__(self, "_base_dir", Path.cwd())
        self.output_dir = _resolve(self.output_dir)
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ConfigError(f"Unsupported device: {self.device!r}")

    # -- construction -------------------------------------------------------
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PipelineConfig:
        """Build a validated config from a plain mapping."""
        _require_keys(
            data,
            ("run_name", "seed", "device", "log_level", "output_dir",
             "data", "model", "train", "eval", "index", "extra"),
            "root",
        )
        payload: dict[str, Any] = {
            k: v for k, v in data.items()
            if k not in ("data", "model", "train", "eval", "index")
        }
        if "output_dir" in payload:
            payload["output_dir"] = _resolve(payload["output_dir"])
        return cls(
            data=DataConfig.from_mapping(data.get("data", {})),
            model=ModelConfig.from_mapping(data.get("model", {})),
            train=TrainConfig.from_mapping(data.get("train", {})),
            eval=EvalConfig.from_mapping(data.get("eval", {})),
            index=IndexConfig.from_mapping(data.get("index", {})),
            **payload,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load a YAML file; ``data.root``-relative paths stay relative to it."""
        config_path = _resolve(path)
        if not config_path.is_file():
            raise ConfigError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")
        return cls.from_mapping(raw)

    # -- serialisation ------------------------------------------------------
    # ``to_mapping`` / ``to_dict`` are inherited from ``_ConfigSection``; this class adds
    # only the fingerprint, which is what ties artifacts back to a recipe.

    def fingerprint(self) -> str:
        """Stable short hash of the *semantic* config, independent of where it was loaded.

        Every path in the config is resolved to an absolute path at construction time, so
        hashing the resolved values would make the same recipe look like a different one when
        run from another directory or another checkout. Paths are therefore re-expressed
        relative to the current working directory before hashing, which is exactly the
        grounding they were resolved against.

        This is what makes a fingerprint usable as a recipe identity: two runs can be compared
        by hash without first normalising their paths, and an artifact can be matched to the
        config that produced it wherever that config was invoked from.
        """
        payload = self.to_mapping()

        # Output locations are stripped outright: where a run writes its results is not part
        # of the recipe, and treating it as such made two identical experiments look like two.
        # ``output_dir`` has a per-section default, so all four sites are removed.
        payload.pop("output_dir", None)
        for section in ("train", "index"):
            if isinstance(payload.get(section), MutableMapping):
                payload[section].pop("output_dir", None)

        def relativise(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            try:
                candidate = Path(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return value
            # Only paths that a config would legitimately carry; a bare string such as a
            # backbone name must not be reshaped.
            if not candidate.is_absolute():
                return value
            for anchor in (Path(self._base_dir), Path.cwd()):
                try:
                    return candidate.relative_to(anchor).as_posix()
                except ValueError:
                    continue
            # Outside both anchors — an external dataset or checkpoint. Keep it absolute,
            # because that information is genuinely part of the recipe.
            return candidate.as_posix()

        def walk(node: Any) -> Any:
            if isinstance(node, MutableMapping):
                return {key: walk(item) for key, item in node.items()}
            if isinstance(node, (list, tuple)):
                return [walk(item) for item in node]
            return relativise(node)

        blob = json.dumps(walk(payload), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def resolve_device(self) -> str:
        """Turn ``auto`` into a concrete device string for this machine."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # pragma: no cover - torch absent
            pass
        return "cpu"


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> PipelineConfig:
    """Load a config file, then apply dotted ``overrides`` on top.

    Overrides use ``section.key`` notation, e.g. ``{"model.backbone": "resnet50"}``.
    Applying them after parsing keeps a single validation path for both file and
    CLI-supplied values.
    """
    if path is None:
        config = PipelineConfig()
        mapping: dict[str, Any] = config.to_mapping()
    else:
        mapping = PipelineConfig.from_yaml(path).to_mapping()

    for dotted, value in (overrides or {}).items():
        cursor: MutableMapping[str, Any] = mapping
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = cursor.get(part)
            if not isinstance(node, MutableMapping):
                raise ConfigError(f"Cannot override {dotted!r}: {part!r} is not a section")
            cursor = node
        if parts[-1] not in cursor:
            raise ConfigError(f"Cannot override unknown key {dotted!r}")
        cursor[parts[-1]] = value

    return PipelineConfig.from_mapping(mapping)


def dump_config(config: PipelineConfig, path: str | Path) -> Path:
    """Write a resolved config to YAML (used for run provenance)."""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_mapping(), handle, sort_keys=False, allow_unicode=True)
    return target


__all__ = [
    "VALID_BACKBONES",
    "VALID_HEADS",
    "VALID_LOSSES",
    "VALID_POOLING",
    "DataConfig",
    "EvalConfig",
    "IndexConfig",
    "ModelConfig",
    "PipelineConfig",
    "TrainConfig",
    "dump_config",
    "load_config",
]
