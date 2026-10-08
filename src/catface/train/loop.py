"""Metric-learning training loop.

What this loop is for
---------------------
A backbone pretrained on ImageNet is a *classifier*: its descriptor says "this is a
cat". Identity retrieval needs "this is *that* cat", which is a different objective.
This loop retrains the descriptor with a margin head over training identities.

Design choices that matter for a small corpus on a small GPU:

* **PK sampling** — each batch draws ``identities_per_batch`` identities and
  ``samples_per_identity`` images each. With a random shuffle, a batch of 32 rarely
  contains two images of the same cat, so both the margin loss and the auxiliary
  triplet term see almost no positive pairs.
* **Split learning rates** — the head starts random, the backbone does not. Training
  them at the same rate destroys pretrained features within a few hundred steps.
* **Identity-disjoint validation** — model selection uses held-out *identities*, and
  the selection metric is retrieval quality, not training loss. Validation loss is
  anti-correlated with retrieval quality often enough that it is not used.
* **EMA** — an exponential-moving-average copy of the weights is a cheap, reliable
  accuracy win and removes the need to guess the best epoch precisely.
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ModelError
from ..eval.metrics import evaluate_retrieval
from ..eval.protocols import Split
from ..logging_utils import get_logger
from ..models.embedder import Embedder, embed_records
from ..models.heads import TripletLoss, VarianceRegulariser

LOGGER = get_logger("train.loop")

TRAIN_RECORD_VERSION = 2
STATE_FORMAT_VERSION = 1

#: Resumable training state inside the run directory.
STATE_FILENAME = "train_state.pt"
#: Presence of this file pauses a run at the next safe point (end of an epoch).
PAUSE_FILENAME = "PAUSE"
#: Best-validation weights, written at the end of a run (also kept on pause).
BEST_FILENAME = "best.pt"


@dataclass
class TrainConfigResolved:
    """Fully resolved training hyper-parameters (config + corpus-derived values)."""

    loss: str = "arcface"
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    backbone_lr_scale: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    scheduler: str = "cosine"
    label_smoothing: float = 0.05
    triplet_weight: float = 0.3
    triplet_margin: float = 0.3
    amp: bool = True
    grad_clip: float = 1.0
    ema_decay: float = 0.0
    identities_per_batch: int = 16
    samples_per_identity: int = 4
    val_every: int = 1
    early_stop_patience: int = 8
    num_workers: int = 4
    seed: int = 1337
    image_size: int = 224
    output_dir: Path = Path("artifacts/train")
    fingerprint: str = ""
    """Semantic hash of the originating pipeline config, recorded in the resumable state."""

    def as_dict(self) -> dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}


@dataclass
class EpochRecord:
    """One epoch of training history."""

    epoch: int
    loss: float
    learning_rate: float
    seconds: float
    val_recall_at_1: float | None = None
    val_map: float | None = None
    val_identities: int | None = None
    best_so_far: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TrainingHistory:
    """Full training record, persisted next to the checkpoint."""

    config: dict[str, Any]
    class_names: list[str]
    epochs: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = -1.0
    stopped_early: bool = False
    pause_reason: str | None = None
    """Why a run stopped short of its schedule, or ``None`` if it ran to completion.

    Kept distinct from ``stopped_early`` on purpose: a paused run is not a converged one,
    and conflating them would let an interrupted experiment be reported as finished.
    """
    version: int = TRAIN_RECORD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "config": self.config,
            "class_names": self.class_names,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "stopped_early": self.stopped_early,
            "pause_reason": self.pause_reason,
            "epochs": [e.to_dict() for e in self.epochs],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingHistory":
        """Rebuild a history from :meth:`to_dict` output, tolerating missing fields."""
        epochs = [
            EpochRecord(**{k: v for k, v in entry.items() if k in EpochRecord.__annotations__})
            for entry in payload.get("epochs", [])
        ]
        return cls(
            config=payload.get("config", {}),
            class_names=list(payload.get("class_names", [])),
            epochs=epochs,
            best_epoch=int(payload.get("best_epoch", -1)),
            best_score=float(payload.get("best_score", -1.0)),
            stopped_early=bool(payload.get("stopped_early", False)),
            pause_reason=payload.get("pause_reason"),
        )


@dataclass
class TrainingState:
    """Everything needed to continue a stopped run *exactly* where it left off.

    A checkpoint of the model weights alone is not enough to resume: the optimiser's
    momentum, the learning-rate schedule's position and the sampler's epoch all carry state
    that, if discarded, silently changes the optimisation trajectory. The RNG states are
    included so a resumed run reproduces what an uninterrupted one would have done.
    """

    version: int = STATE_FORMAT_VERSION
    epoch: int = 0
    """Number of epochs fully completed. Training resumes at ``epoch + 1``."""
    global_step: int = 0
    epochs_planned: int = 0
    history: TrainingHistory | None = None
    model: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    scaler: dict[str, Any] = field(default_factory=dict)
    best_state: dict[str, Any] | None = None
    ema_state: dict[str, Any] | None = None
    patience: int = 0
    rng: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: str = ""
    class_names: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        """Small, loggable summary (never the weights themselves)."""
        return {
            "version": self.version,
            "epoch": self.epoch,
            "epochs_planned": self.epochs_planned,
            "global_step": self.global_step,
            "best_epoch": self.history.best_epoch if self.history else None,
            "best_score": self.history.best_score if self.history else None,
            "patience": self.patience,
            "has_ema": self.ema_state is not None,
        }


def _rng_state() -> dict[str, Any]:
    """Capture Python/NumPy/Torch RNG state so a resumed run is reproducible."""
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        pass
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    """Restore RNG state captured by :func:`_rng_state`, ignoring anything absent."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:  # pragma: no cover
        LOGGER.warning("Could not restore torch RNG state; the resumed run may differ")


def _atomic_torch_save(payload: Any, path: Path) -> Path:
    """Write a torch checkpoint atomically.

    A checkpoint written in place is destroyed if the process dies mid-write, which is
    exactly the situation resumable training is meant to survive. Writing to a temporary
    file and renaming means the previous checkpoint stays valid until the new one is
    complete.
    """
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


class PKBatchSampler:
    """Yields index batches shaped as ``identities_per_batch x samples_per_identity``.

    Produces plain Python lists of indices so it can be handed to a
    :class:`torch.utils.data.DataLoader` without custom collation logic.
    """

    def __init__(
        self,
        labels: Sequence[str],
        identities_per_batch: int,
        samples_per_identity: int,
        batches_per_epoch: int,
        seed: int = 1337,
    ) -> None:
        groups: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            groups.setdefault(label, []).append(index)
        # Identities with a single image can never form a positive pair inside a batch.
        self.groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if not self.groups:
            raise ModelError(
                "No identity has at least 2 training images; PK sampling cannot form "
                "positive pairs. Supply more images per identity."
            )
        self.identities = sorted(self.groups)
        self.p = min(identities_per_batch, len(self.identities))
        self.k = samples_per_identity
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for a new epoch; keeps epochs distinct but reproducible."""
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 7919)
        for _ in range(self.batches_per_epoch):
            chosen = rng.choice(self.identities, size=self.p, replace=False)
            batch: list[int] = []
            for identity in chosen:
                pool = self.groups[identity]
                # ``replace=False`` when possible so a batch never shows the same photo
                # twice, which would teach the model that identical pixels are the
                # whole task.
                if len(pool) >= self.k:
                    picks = rng.choice(len(pool), size=self.k, replace=False)
                else:
                    picks = rng.choice(len(pool), size=self.k, replace=True)
                batch.extend(pool[i] for i in picks)
            yield batch

    def dataset_labels(self) -> list[str]:
        """Identity label per dataset position, in dataset index order."""
        labels = [""] * (sum(len(v) for v in self.groups.values()))
        for identity, indices in self.groups.items():
            for index in indices:
                labels[index] = identity
        return labels


class IdentityImageDataset:
    """Map-style dataset of face tiles with identity labels.

    Deliberately small: it decodes an image, applies the transform, returns a tensor.
    Augmentation is *geometric and mild* — identity must survive it, so aggressive
    colour jitter or large rotations would inject label noise.
    """

    def __init__(
        self,
        paths: Sequence[str],
        labels: Sequence[int],
        image_size: int = 224,
        train: bool = True,
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.image_size = image_size
        self.train = train
        self.transform = self._build_transform(normalize_mean, normalize_std)
        self.failures = 0

    def _build_transform(self, mean: Sequence[float], std: Sequence[float]):
        import torchvision.transforms as T

        if not self.train:
            return T.Compose([
                T.Resize(int(self.image_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=list(mean), std=list(std)),
            ])
        return T.Compose([
            T.RandomResizedCrop(
                self.image_size, scale=(0.75, 1.0), ratio=(0.9, 1.11),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
            T.RandomErasing(p=0.15, scale=(0.02, 0.12), value="random"),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image

        path = self.paths[index]
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                tensor = self.transform(image)
        except Exception as exc:
            self.failures += 1
            LOGGER.warning("Falling back to a blank tile for %s (%s)", path, exc)
            import torch

            tensor = torch.zeros(3, self.image_size, self.image_size)
        return tensor, self.labels[index]


class Trainer:
    """Owns the optimisation loop, validation, checkpointing and history."""

    def __init__(
        self,
        embedder: Embedder,
        train_records: Sequence[Any],
        val_split: Split,
        config: TrainConfigResolved,
        device: str = "cuda",
    ) -> None:
        torch = _torch()
        self.torch = torch
        self.embedder = embedder
        self.config = config
        self.device = device

        identities = sorted({r.identity for r in train_records if r.identity})
        if not identities:
            raise ModelError("Training records carry no identity labels")
        self.class_names = identities
        self.class_to_index = {name: index for index, name in enumerate(identities)}

        # The dataset and the sampler must share one index space.
        #
        # ``PKBatchSampler`` drops identities that have fewer than two images, because such
        # an identity can never form a positive pair inside a batch. If the dataset were
        # built from the *unfiltered* records while the sampler yields indices into the
        # filtered list, those indices would address the wrong images — training would
        # proceed without error while learning from misaligned labels. Filtering here, once,
        # keeps the two lists identical by construction.
        counts = Counter(r.identity for r in train_records if r.identity)
        usable = {identity for identity, count in counts.items() if count >= 2}
        dropped = len(identities) - len(usable)
        if dropped:
            LOGGER.warning(
                "Dropping %d identity/ies with a single training image (no positive pair "
                "possible); %d identities remain", dropped, len(usable),
            )
        eligible = [r for r in train_records if r.identity in usable]
        if not eligible:
            raise ModelError(
                "No identity has at least 2 training images; PK sampling cannot form "
                "positive pairs. Supply more images per identity."
            )
        self.class_names = sorted(usable)
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}

        self.paths = [r.path for r in eligible]
        self.labels = [self.class_to_index[r.identity] for r in eligible]

        self.dataset = IdentityImageDataset(
            self.paths, self.labels, image_size=config.image_size, train=True
        )
        batches_per_epoch = max(
            1,
            len(self.paths) // max(config.identities_per_batch * config.samples_per_identity, 1),
        )
        self.sampler = PKBatchSampler(
            [r.identity for r in eligible],
            identities_per_batch=config.identities_per_batch,
            samples_per_identity=config.samples_per_identity,
            batches_per_epoch=batches_per_epoch,
            seed=config.seed,
        )
        assert len(self.sampler.groups) == len(self.class_names), (
            "dataset class space and sampler identity set diverged"
        )
        self.records = eligible
        self.val_split = val_split

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(batches_per_epoch)
        # bfloat16 over float16: bf16 shares fp32's exponent range, so it needs no loss
        # scaling and therefore no scaler. Measured on this corpus, fp16 produced `inf`
        # gradient norms on a substantial fraction of steps, which made clip_grad_norm_
        # return NaN and silently discarded those optimiser updates; bf16 produced none.
        self.amp_dtype = torch.bfloat16 if config.amp else torch.float32
        self.use_amp = bool(config.amp) and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.triplet = TripletLoss(margin=config.triplet_margin)
        self.variance = VarianceRegulariser(weight=0.0)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        self.ema_state: dict[str, Any] | None = None
        self.history = TrainingHistory(
            config=config.as_dict(), class_names=self.class_names
        )
        self.best_state: dict[str, Any] | None = None

        # -- pause / resume -------------------------------------------------
        self.epochs_planned = int(config.epochs)
        self.global_step = 0
        self.patience = 0
        self._pause_requested = False
        self._pause_reason: str | None = None
        self._budget_deadline: float | None = None
        output_dir = Path(config.output_dir)
        self.state_path = output_dir / STATE_FILENAME
        self.best_path = output_dir / BEST_FILENAME
        self.pause_path = output_dir / PAUSE_FILENAME
        self.config_fingerprint = str(getattr(config, "fingerprint", "") or "")
        self._previous_signal_handlers: dict[int, Any] = {}

    # -- setup --------------------------------------------------------------
    def _build_optimizer(self):
        torch = _torch()
        groups = self.embedder.named_parameter_groups(
            lr=self.config.lr,
            backbone_lr_scale=self.config.backbone_lr_scale,
            weight_decay=self.config.weight_decay,
        )
        groups = [g for g in groups if g["params"]]
        if self.config.loss in ("triplet",):
            # A pure-triplet objective needs a classifier head only as a projection.
            LOGGER.info("Loss=triplet: the margin head is used as a projection only")
        if not groups:
            raise ModelError("Optimizer received no trainable parameters")
        return torch.optim.AdamW(groups)

    def _build_scheduler(self, steps_per_epoch: int):
        torch = _torch()
        total_steps = max(1, steps_per_epoch * self.config.epochs)
        warmup_steps = max(1, steps_per_epoch * self.config.warmup_epochs)
        if self.config.scheduler == "constant":
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        if self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, steps_per_epoch * 10), gamma=0.1
            )
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: _warmup_cosine(step, warmup_steps, total_steps),
        )

    # -- pause and resume ---------------------------------------------------
    def request_pause(self, reason: str = "manual") -> None:
        """Ask the loop to stop at the next safe point. Reentrant and idempotent.

        A "safe point" is the end of an epoch, so pausing never interrupts an optimiser
        step or leaves a half-written checkpoint. The first reason wins, because a signal
        that arrives while a budget pause is already pending should not relabel it.
        """
        if not self._pause_requested:
            self._pause_requested = True
            self._pause_reason = reason
            LOGGER.info("Pause requested (%s); will stop at the end of this epoch", reason)

    @property
    def pause_requested(self) -> bool:
        return self._pause_requested

    def _pause_sources(self) -> list[str]:
        """External pause signals, checked cheaply at each epoch boundary."""
        reasons: list[str] = []
        if self._budget_deadline is not None and time.perf_counter() >= self._budget_deadline:
            reasons.append("time-budget")
        if self.pause_path.exists():
            reasons.append("pause-file")
        return reasons

    def _install_signal_handlers(self) -> int:
        """Turn SIGINT/SIGTERM into a graceful pause. Returns the number installed.

        SIGTERM matters for scheduler-driven runs: without a handler the job is killed
        mid-epoch and all progress since the last checkpoint is lost. SIGKILL cannot be
        caught, which is precisely why a checkpoint is written every epoch.
        """
        installed = 0

        def handler(signum: int, _frame: Any) -> None:  # pragma: no cover - signal path
            self.request_pause(f"signal-{signum}")

        for name in ("SIGINT", "SIGTERM"):
            number = getattr(signal, name, None)
            if number is None:
                continue
            try:
                self._previous_signal_handlers[number] = signal.getsignal(number)
                signal.signal(number, handler)
                installed += 1
            except (ValueError, OSError):
                # Not on the main thread, or unsupported on this platform.
                LOGGER.debug("Could not install a %s handler", name)
        return installed

    def _remove_signal_handlers(self) -> None:
        for number, previous in self._previous_signal_handlers.items():
            try:
                signal.signal(number, previous)
            except (ValueError, OSError, TypeError):  # pragma: no cover
                pass
        self._previous_signal_handlers.clear()

    def capture_state(self) -> TrainingState:
        """Snapshot everything required to continue this run later."""
        model_state = {k: v.detach().cpu() for k, v in self.embedder.state_dict().items()}
        return TrainingState(
            epoch=int(self.history.epochs[-1].epoch) if self.history.epochs else 0,
            global_step=int(self.global_step),
            epochs_planned=int(self.epochs_planned),
            history=self.history,
            model=model_state,
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict() if self.scaler is not None else {},
            best_state=(
                {k: v.detach().cpu() for k, v in self.best_state.items()}
                if self.best_state is not None
                else None
            ),
            ema_state=(
                {k: v.detach().cpu() for k, v in self.ema_state.items()}
                if self.ema_state is not None
                else None
            ),
            patience=int(self.patience),
            rng=_rng_state(),
            config_fingerprint=self.config_fingerprint,
            class_names=list(self.class_names),
        )

    def save_state(self, path: str | Path | None = None) -> Path:
        """Atomically persist the resumable state. Also refreshes ``history.json``."""
        target = Path(path) if path is not None else self.state_path
        state = self.capture_state()
        saved = _atomic_torch_save(state, target)
        # The JSON history is written alongside so a paused run is inspectable without
        # loading torch tensors.
        self.history.save(target.parent / "training_history.json")
        LOGGER.info(
            "Checkpointed resumable state at epoch %d -> %s",
            state.epoch, saved,
        )
        return saved

    def load_state(self, path: str | Path | None = None, strict: bool = True) -> TrainingState:
        """Restore a state produced by :meth:`save_state`, ready to continue training.

        Args:
            path: State file; defaults to ``<output_dir>/train_state.pt``.
            strict: When ``True``, refuse a state whose class list or epoch count differs
                from this trainer's, because resuming into a different identity set or a
                shorter schedule silently produces a different experiment.

        Returns:
            The restored :class:`TrainingState`.

        Raises:
            ModelError: If the file is absent, has an incompatible version, or — under
                ``strict`` — does not match this trainer's configuration.
        """
        torch = _torch()
        source = Path(path) if path is not None else self.state_path
        if not source.is_file():
            raise ModelError(f"No resumable state at {source}")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(payload, TrainingState):
            raise ModelError(
                f"{source} does not contain a TrainingState (found {type(payload).__name__})"
            )
        state: TrainingState = payload
        if state.version != STATE_FORMAT_VERSION:
            raise ModelError(
                f"{source} has state version {state.version}, this build expects "
                f"{STATE_FORMAT_VERSION}"
            )
        if strict:
            if state.class_names and state.class_names != list(self.class_names):
                raise ModelError(
                    f"{source} was trained on {len(state.class_names)} identities but this "
                    f"trainer has {len(self.class_names)}; refusing to resume into a "
                    "different class space"
                )
            if state.epoch > self.epochs_planned:
                raise ModelError(
                    f"{source} is at epoch {state.epoch} but the schedule has only "
                    f"{self.epochs_planned} epochs; raise --epochs to continue"
                )

        # Under ``strict=False`` the caller has accepted an incompatible state, so tensors
        # that do not fit are skipped rather than raising: the backbone weights are still
        # useful and the head is re-initialised for the current identity set.
        head_compatible = not state.class_names or state.class_names == list(self.class_names)
        self.embedder.load_state_dict_into(
            state.model,
            skip_prefixes=None if head_compatible else ("head.",),
        )
        if not head_compatible:
            LOGGER.warning(
                "Class space differs (%d -> %d); the margin head was re-initialised and "
                "only the backbone weights were carried over",
                len(state.class_names), len(self.class_names),
            )
        self.optimizer.load_state_dict(state.optimizer)
        self.scheduler.load_state_dict(state.scheduler)
        if state.scaler and self.scaler is not None:
            self.scaler.load_state_dict(state.scaler)
        self.best_state = state.best_state
        self.ema_state = state.ema_state
        self.patience = int(state.patience)
        self.global_step = int(state.global_step)
        self.history = state.history if state.history is not None else self.history
        self.epochs_planned = max(int(state.epochs_planned), int(self.epochs_planned))
        _restore_rng(state.rng)
        LOGGER.info("Resuming from %s: %s", source, json.dumps(state.describe(), default=str))
        return state

    # -- training -----------------------------------------------------------
    def fit(
        self,
        max_seconds: float | None = None,
        handle_signals: bool = True,
    ) -> TrainingHistory:
        """Train, checkpointing after every epoch so the run can be paused and resumed.

        Args:
            max_seconds: Wall-clock budget. When exceeded, training pauses cleanly at the
                next epoch boundary instead of being killed. Useful for fitting a run into
                a maintenance window or a shared machine.
            handle_signals: Install SIGINT/SIGTERM handlers that pause gracefully rather
                than aborting mid-epoch.

        Pause is requested by any of: ``SIGINT``/``SIGTERM``, a ``PAUSE`` file in the run
        directory, ``Trainer.request_pause()``, or the ``max_seconds`` budget. In every case
        the run stops at the next epoch boundary, writes ``train_state.pt``, and leaves
        ``history.stopped_early`` untouched so a paused run is not mistaken for a converged
        one. Resume by calling :meth:`load_state` then :meth:`fit` again.
        """
        torch = self.torch
        if max_seconds is not None and max_seconds > 0:
            self._budget_deadline = time.perf_counter() + float(max_seconds)
            LOGGER.info("Time budget set: %.1f minutes", max_seconds / 60.0)
        installed = self._install_signal_handlers() if handle_signals else 0
        if installed:
            LOGGER.info("Graceful pause installed for %d signal(s) (SIGINT/SIGTERM)", installed)

        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.device.startswith("cuda"),
            persistent_workers=self.config.num_workers > 0,
        )
        start_epoch = (self.history.epochs[-1].epoch + 1) if self.history.epochs else 1
        if start_epoch > 1:
            LOGGER.info(
                "Continuing from epoch %d of %d (%d epochs already recorded)",
                start_epoch, self.epochs_planned, len(self.history.epochs),
            )

        try:
            for epoch in range(start_epoch, self.epochs_planned + 1):
                # Check before doing work so a pause file dropped between epochs — or a
                # request made before `fit` was ever called — costs no training time.
                external = self._pause_sources()
                if external and not self._pause_requested:
                    self.request_pause(external[0])
                if self._pause_requested:
                    LOGGER.info(
                        "Not starting epoch %d: %s", epoch, self._pause_reason
                    )
                    break

                self.sampler.set_epoch(epoch)
                self.embedder.train(True)
                started = time.perf_counter()
                losses: list[float] = []
                learning_rate = self.optimizer.param_groups[0]["lr"]

                n_batches = len(loader)
                for batch_index, (images, labels) in enumerate(loader, start=1):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad(set_to_none=True)
                    # The positional-embedding scope must stay open across the *backward*
                    # pass: gradient checkpointing recomputes the forward, so restoring the
                    # native embedding early makes the recomputation disagree with the
                    # recorded graph. That failure mode is silent — the AMP scaler discards
                    # the step and the model never updates while the loss stays finite.
                    with self.embedder.resolution_scope(int(images.shape[-2]), int(images.shape[-1])):
                        with torch.amp.autocast("cuda", enabled=self.use_amp,
                                                dtype=self.amp_dtype):
                            # Pooled (unprojected) features: the head applies the projection.
                            features = self.embedder.pool_features(images)
                            loss = self._compute_loss(labels, features)
                        self.scaler.scale(loss).backward()
                    if self.config.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.embedder.parameters(), self.config.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    if self.config.ema_decay > 0:
                        self._update_ema()
                    losses.append(float(loss.detach()))
                    self.global_step += 1

                    # A single epoch can take minutes on a 6 GB card, so emit a running line.
                    # Without it a long run is a black box and a stall is indistinguishable
                    # from slow progress.
                    if batch_index % 25 == 0 or batch_index == n_batches:
                        running = float(np.mean(losses[-25:]))
                        elapsed_in_epoch = time.perf_counter() - started
                        rate = elapsed_in_epoch / batch_index
                        remaining = (n_batches - batch_index) * rate
                        LOGGER.info(
                            "epoch %d/%d batch %d/%d loss=%.4f (last-25 mean) "
                            "%.2f s/batch, ~%.1f min left in epoch",
                            epoch, self.epochs_planned, batch_index, n_batches, running,
                            rate, remaining / 60.0,
                        )

                mean_loss = float(np.mean(losses)) if losses else float("nan")
                epoch_seconds = time.perf_counter() - started
                record = EpochRecord(
                    epoch=epoch, loss=mean_loss, learning_rate=learning_rate,
                    seconds=round(epoch_seconds, 2),
                )

                if epoch % max(1, self.config.val_every) == 0 or epoch == self.epochs_planned:
                    recall, map_score, identities = self.validate()
                    record.val_recall_at_1 = recall
                    record.val_map = map_score
                    record.val_identities = identities
                    record.best_so_far = max(self.history.best_score, recall)
                    if recall > self.history.best_score + 1e-6:
                        self.history.best_score = recall
                        self.history.best_epoch = epoch
                        self.best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.embedder.state_dict().items()
                        }
                        self.patience = 0
                    else:
                        self.patience += 1

                self.history.epochs.append(record)
                LOGGER.info(
                    "epoch %d/%d loss=%.4f val_R@1=%s (%.1fs)",
                    epoch, self.epochs_planned, mean_loss,
                    f"{record.val_recall_at_1:.4f}" if record.val_recall_at_1 is not None else "n/a",
                    epoch_seconds,
                    extra={"stage": "train", "epoch": epoch, "metric": "loss", "value": mean_loss},
                )

                # Checkpoint every epoch: this is what makes an unannounced kill survivable.
                self.save_state()

                if self.patience >= self.config.early_stop_patience > 0:
                    LOGGER.info("Early stopping after %d epochs without improvement", self.patience)
                    self.history.stopped_early = True
                    self.save_state()
                    break

                external = self._pause_sources()
                if external and not self._pause_requested:
                    self.request_pause(external[0])
                if self._pause_requested:
                    break
        finally:
            self._remove_signal_handlers()

        reason = self._pause_reason if self._pause_requested else None
        if reason is None:
            # Finished the schedule: restore the validated weights so the exported
            # checkpoint is the best one rather than the last one.
            if self.best_state is not None:
                self.embedder.load_state_dict_into(self.best_state)
            elif self.config.ema_decay > 0 and self.ema_state is not None:
                self.embedder.load_state_dict_into(self.ema_state)
            self.save_state()
        else:
            # Paused: keep the *latest* weights. Restoring the best epoch here would make
            # the next `fit` continue from a different trajectory than an uninterrupted
            # run would have taken, which is exactly the reproducibility this feature exists
            # to protect.
            LOGGER.info(
                "Paused (%s) after epoch %d of %d. Resume with `--resume`; "
                "the latest weights are kept so the optimisation continues unchanged.",
                reason, self.history.epochs[-1].epoch if self.history.epochs else 0,
                self.epochs_planned,
            )
            self.save_state()

        # ``best.pt`` is the inference artifact and is written on both paths: after a
        # completed run it holds the validated weights, and after a pause it holds the best
        # seen so far (which may be no better than a random head if nothing completed).
        if self.best_state is not None or self.history.epochs:
            self.export_best()

        self.history.pause_reason = reason if reason is not None else self.history.pause_reason
        self.history.save(Path(self.config.output_dir) / "training_history.json")
        return self.history

    def export_best(self, path: str | Path | None = None) -> Path:
        """Write the best-validation weights as an inference checkpoint.

        Kept separate from the resumable state: the state is large (optimiser moments) and
        is for continuing training, whereas this file is what an index build or a benchmark
        loads.
        """
        target = Path(path) if path is not None else self.best_path
        current = {k: v.detach().cpu().clone() for k, v in self.embedder.state_dict().items()}
        source = self.best_state if self.best_state is not None else current
        try:
            if self.best_state is not None:
                self.embedder.load_state_dict_into(source)
            self.embedder.save(
                target,
                extra={
                    "train_identities": len(self.class_names),
                    "train_images": len(self.records),
                    "best_epoch": self.history.best_epoch,
                    "best_val_hit_at_1": self.history.best_score,
                    "class_names": list(self.class_names),
                    "exported_from": "Trainer.export_best",
                },
            )
        finally:
            self.embedder.load_state_dict_into(current)
        return target

    def _compute_loss(self, labels, features):
        """Combine the margin-head loss with the auxiliary objectives.

        Args:
            labels: `(B,)` identity indices.
            features: `(B, pooled_dim)` *unprojected* backbone features. The head applies
                BatchNorm, projection and normalisation itself; passing already-projected
                embeddings here would project twice.

        Returns:
            A scalar loss tensor.
        """
        logits = self.embedder.head(features, labels)
        embeddings = self.embedder.head.embed(features)
        if logits.ndim == 2 and logits.shape[0] == labels.shape[0]:
            loss = self.criterion(logits, labels)
        else:
            loss = torch_zero(embeddings)
        if self.config.loss == "none":
            # Pure metric learning: no classification term at all.
            loss = self.triplet(embeddings, labels)
        elif self.config.triplet_weight > 0:
            loss = loss + self.config.triplet_weight * self.triplet(embeddings, labels)
        return loss + self.variance(embeddings, labels)

    def _update_ema(self) -> None:
        decay = self.config.ema_decay
        current = self.embedder.state_dict()
        if self.ema_state is None:
            self.ema_state = {k: v.detach().clone().float() for k, v in current.items()}
            return
        for key, value in current.items():
            if key not in self.ema_state:
                self.ema_state[key] = value.detach().clone().float()
            else:
                self.ema_state[key].mul_(decay).add_(value.detach().float(), alpha=1 - decay)

    @staticmethod
    def _compute_gallery(split: Split) -> tuple[np.ndarray, np.ndarray]:
        paths = [r.path for r in split.gallery_records]
        labels = np.array([r.identity for r in split.gallery_records])
        return np.array(paths, dtype=object), labels

    def validate(self) -> tuple[float, float, int]:
        """Score the current weights on held-out identities by retrieval quality.

        The headline number is the top-1 **hit rate** (``hit@1``) — the share of held-out
        identities whose query finds a same-identity reference in the top 1 — because that is
        the metric model selection should track. Reporting *full recall* here instead reads as
        roughly ``1 / references_per_identity`` and is nearly insensitive to model quality,
        which makes it useless for choosing a checkpoint.

        Returns:
            ``(hit_at_1, mean_average_precision, num_identities)``.
        """
        query_paths = [r.path for r in self.val_split.query_records]
        gallery_paths = [r.path for r in self.val_split.gallery_records]
        query = embed_records(
            self.embedder, query_paths, image_size=self.config.image_size,
            batch_size=max(8, self.config.batch_size // 2),
        )
        gallery = embed_records(
            self.embedder, gallery_paths, image_size=self.config.image_size,
            batch_size=max(8, self.config.batch_size // 2),
        )
        if query.vectors.size == 0 or gallery.vectors.size == 0:
            return 0.0, 0.0, 0
        similarity = query.vectors @ gallery.vectors.T
        self_mask = np.array(
            [[a == b for b in gallery.ids] for a in query.ids], dtype=bool
        )
        metrics = evaluate_retrieval(
            similarity,
            query_labels=self.val_split.query_labels,
            gallery_labels=self.val_split.gallery_labels,
            recall_ks=(1, 5),
            exclude_self=self_mask if self_mask.any() else None,
        )
        return (
            metrics.hit_at.get(1, 0.0),
            metrics.map_at.get(5, 0.0),
            len(set(self.val_split.query_labels.tolist())),
        )


def _warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup followed by cosine decay, as a multiplier on the base LR."""
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def torch_zero(reference):
    """A zero tensor on the same device as ``reference``, keeping the graph intact."""
    return reference.sum() * 0.0


def train_metric_learner(
    embedder: Embedder,
    train_records: Sequence[Any],
    val_split: Split,
    config: TrainConfigResolved,
    device: str = "cuda",
    resume: bool = False,
    max_seconds: float | None = None,
    handle_signals: bool = True,
) -> tuple[TrainingHistory, Trainer]:
    """Convenience wrapper: build a :class:`Trainer`, run it, persist the history.

    Args:
        resume: When ``True`` and a ``train_state.pt`` exists in the output directory,
            continue that run instead of starting a new one.
        max_seconds: Wall-clock budget; see :meth:`Trainer.fit`.
        handle_signals: Install graceful pause handlers; see :meth:`Trainer.fit`.

    Returns:
        ``(history, trainer)``. The trainer is returned because callers need it to export
        the best checkpoint, inspect whether the run paused, or resume again later.
    """
    trainer = Trainer(embedder, train_records, val_split, config, device=device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        if trainer.state_path.is_file():
            trainer.load_state(trainer.state_path)
        else:
            LOGGER.warning(
                "--resume was requested but %s does not exist; starting a new run",
                trainer.state_path,
            )
    history = trainer.fit(max_seconds=max_seconds, handle_signals=handle_signals)
    history.save(config.output_dir / "training_history.json")
    return history, trainer


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ModelError("PyTorch is required for training") from exc
    return torch


__all__ = [
    "BEST_FILENAME",
    "EpochRecord",
    "IdentityImageDataset",
    "PAUSE_FILENAME",
    "PKBatchSampler",
    "STATE_FILENAME",
    "TrainConfigResolved",
    "Trainer",
    "TrainingHistory",
    "TrainingState",
    "train_metric_learner",
]
