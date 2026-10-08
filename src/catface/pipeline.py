"""End-to-end orchestration.

The CLI is a thin shell over the functions here, so every stage is importable and
testable without spawning a process. Each function returns a JSON-serialisable report
that is written into ``artifacts/`` alongside the run config — that pairing is what
makes a result auditable later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .data.annotation import load_oiid_entries
from .data.cropping import Box
from .data.manifest import Manifest
from .data.prepare import (
    finalise_manifest,
    make_writer,
    prepare_cat_individuals,
    prepare_oiid,
    prepare_oiid_from_annotations,
)
from .data.sources import CATALOGUE, DatasetStore
from .errors import DataError
from .eval.protocols import Split, assert_identity_disjoint, build_identity_split
from .logging_utils import get_logger, new_run_id, timed
from .models.embedder import Embedder, EmbedderConfig
from .train.loop import TrainConfigResolved, train_metric_learner

LOGGER = get_logger("pipeline")

#: Corpora this project knows how to build a manifest from.
SOURCE_LAYOUTS = {
    "oiid_cat": {
        "kind": "identity",
        "description": (
            "Oxford-IIIT Pet cats: 3686 photos of 830 individuals across 12 breeds, "
            "with an annotated head box per image. Training corpus."
        ),
        "identity_labels": True,
        "face_boxes": True,
    },
    "cat_individuals": {
        "kind": "identity",
        "description": (
            "Kaggle Cat Individual Images: 13536 photos of 518 cats in uncontrolled "
            "conditions. Evaluation corpus."
        ),
        "identity_labels": True,
        "face_boxes": False,
    },
    "calfw": {
        "kind": "pairs",
        "description": "6000 labelled same/different cat-face pairs. Verification corpus.",
        "identity_labels": False,
        "face_boxes": False,
    },
    "catfaces29k": {
        "kind": "unlabelled",
        "description": "~29842 unlabelled cat-face crops. Domain statistics only.",
        "identity_labels": False,
        "face_boxes": False,
    },
}


@dataclass
class DatasetPlan:
    """What a stage intends to use, and why — printed before any long operation."""

    source: str
    root: Path
    adapter: str
    notes: list[str]

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "root": str(self.root),
            "adapter": self.adapter,
            "notes": self.notes,
            "layout": SOURCE_LAYOUTS.get(self.source, {}),
        }


def resolve_oiid_paths(config: PipelineConfig) -> DatasetPlan:
    """Locate OIID in either supported layout and choose the best adapter.

    Preference order:
      1. original ``list.txt`` + ``xmls/`` + ``images/`` (head boxes available),
      2. the HuggingFace parquet mirror (no head boxes; whole image is cropped).

    The chosen adapter is reported, because the two produce measurably different crops
    and a number reported from layout 2 is not comparable to one from layout 1.
    """
    root = config.data.root
    notes: list[str] = []

    annotations_dir = root / "oxford_annotations"
    for candidate in (root / "raw" / "oxford_iiit_pet" / "extracted" / "annotations", annotations_dir):
        if (candidate / "list.txt").is_file() and (candidate / "xmls").is_dir():
            annotations_dir = candidate
            break

    images_dir = None
    for candidate in (
        root / "oxford_images",
        root / "raw" / "oxford_iiit_pet" / "extracted" / "images",
    ):
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            images_dir = candidate
            break

    have_annotations = (annotations_dir / "list.txt").is_file() and (annotations_dir / "xmls").is_dir()
    if have_annotations and images_dir is not None:
        notes.append(f"Using annotated head boxes from {annotations_dir}")
        return DatasetPlan("oiid_cat", root, "oiid_annotations", notes)
    if have_annotations:
        notes.append(
            f"Annotations found at {annotations_dir} but no extracted image directory; "
            "falling back to the parquet mirror, which has no head boxes"
        )

    parquet_dir = root / "raw" / "oxford_iiit_pet"
    available = sorted(p.name for p in parquet_dir.glob("*.parquet"))
    if available:
        notes.append(
            "No annotated image directory; using the parquet mirror. OIID head boxes "
            "are unavailable in this layout, so crops are whole images."
        )
        return DatasetPlan("oiid_cat", root, "oiid_parquet", notes)
    raise DataError(
        "Oxford-IIIT Pet was not found. Expected either "
        f"{root / 'oxford_annotations'} + an extracted images directory, or parquet "
        f"files under {parquet_dir}."
    )


def load_oiid_head_boxes(annotations_dir: Path, species: str = "cat") -> dict[str, Box]:
    """Build ``{image_id: head Box}`` from the OIID XML annotations."""
    entries = load_oiid_entries(annotations_dir, species=species)
    boxes: dict[str, Box] = {}
    for entry in entries:
        if entry.head_box:
            boxes[entry.image_id] = Box(*entry.head_box)
    return boxes


def locate_cat_individuals(config: PipelineConfig) -> Path | None:
    """Find the extracted Kaggle *Cat Individual Images* directory, if present."""
    root = config.data.root
    candidates = [
        # `extracted` is where the archive downloader unpacks the zip.
        root / "raw" / "kaggle_cat_individuals" / "extracted" / "cat_individuals_dataset",
        root / "raw" / "kaggle_cat_individuals" / "cat_individuals_dataset",
        root / "raw" / "cat_individuals" / "extracted" / "cat_individuals_dataset",
        root / "raw" / "cat_individuals" / "cat_individuals_dataset",
        root / "cat_individuals_dataset",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.rglob("*.JPG")):
            return candidate
    return None


def locate_calfw(config: PipelineConfig) -> Path | None:
    """Find a materialised CALFW pair parquet, if present."""
    root = config.data.root
    for candidate in (
        root / "raw" / "kaggle_calfw",
        root / "raw" / "calfw",
    ):
        if candidate.is_dir():
            for parquet in sorted(candidate.glob("*.parquet")):
                return parquet
    return None


def prepare_dataset(config: PipelineConfig, source: str, _force: bool = False) -> dict[str, Any]:
    """Build the face-crop manifest for one source and split it by identity."""
    writer = make_writer(config.data)
    manifest_dir = Path(config.data.manifest)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    plan = resolve_oiid_paths(config) if source == "oiid_cat" else DatasetPlan(
        source=source, root=config.data.root, adapter="", notes=[]
    )

    with timed(LOGGER, f"prepare {source}", stage="prepare", adapter=plan.adapter or source):
        if source == "oiid_cat":
            adapter = plan.adapter
            if adapter == "oiid_annotations":
                annotations_dir = None
                for candidate in (
                    config.data.root / "oxford_annotations",
                    config.data.root / "raw" / "oxford_iiit_pet" / "extracted" / "annotations",
                ):
                    if (candidate / "list.txt").is_file():
                        annotations_dir = candidate
                        break
                images_dir = None
                for candidate in (
                    config.data.root / "oxford_images",
                    config.data.root / "raw" / "oxford_iiit_pet" / "extracted" / "images",
                ):
                    if candidate.is_dir():
                        images_dir = candidate
                        break
                assert annotations_dir is not None and images_dir is not None
                manifest = prepare_oiid_from_annotations(
                    annotations_dir=annotations_dir,
                    images_dir=images_dir,
                    writer=writer,
                    species="cat",
                )
            else:
                parquet_dir = config.data.root / "raw" / "oxford_iiit_pet"
                parquet_paths = {
                    path.stem: path for path in sorted(parquet_dir.glob("*.parquet"))
                }
                if not parquet_paths:
                    raise DataError(f"No parquet files under {parquet_dir}")
                head_boxes = None
                annotations_dir = config.data.root / "oxford_annotations"
                if (annotations_dir / "list.txt").is_file():
                    head_boxes = load_oiid_head_boxes(annotations_dir, species="cat")
                    LOGGER.info("Loaded %d annotated head boxes", len(head_boxes))
                manifest = prepare_oiid(
                    parquet_paths=parquet_paths,
                    writer=writer,
                    species="cat",
                    head_boxes=head_boxes,
                )
        elif source == "cat_individuals":
            root = locate_cat_individuals(config)
            if root is None:
                raise DataError(
                    "Cat Individual Images corpus not found. Expected "
                    f"{config.data.root / 'raw' / 'kaggle_cat_individuals' / 'cat_individuals_dataset'}"
                )
            manifest = prepare_cat_individuals(root, writer=writer)
        else:
            raise DataError(
                f"prepare_dataset does not handle source {source!r}. "
                f"Known: {sorted(SOURCE_LAYOUTS)}"
            )

    report = finalise_manifest(
        manifest,
        output_dir=manifest_dir,
        source_name=source,
        seed=config.data.seed,
    )
    report["prepare_stats"] = writer.stats.as_dict()
    report["descriptor"] = {"tile": config.data.tile, "pad_ratio": config.data.pad_ratio}
    LOGGER.info(
        "Prepared %s: %d usable tiles, %d identities",
        source, report["stats"]["total"], report["stats"]["identities"],
    )
    return report


def load_splits_from_manifest(
    manifest_path: Path,
    splits_dir: Path,
    split_name: str,
) -> list[Any]:
    """Load the records belonging to one split (``train``/``val``/``test``)."""
    manifest = Manifest.load(manifest_path)
    split_file = splits_dir / f"{split_name}.txt"
    if not split_file.is_file():
        raise DataError(f"Split file not found: {split_file}")
    ids = {line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    records = [r for r in manifest if r.image_id in ids]
    if not records:
        raise DataError(f"Split {split_name!r} selected no records from {manifest_path}")
    return records


def build_eval_split(
    _config: PipelineConfig,
    query_manifest: Path,
    gallery_manifest: Path,
    _queries_per_identity: int = 1,
    name: str = "cross_dataset",
    _require_min_identity_images: int = 2,
) -> Split:
    """Combine a query and a gallery manifest into one evaluation protocol.

    Cross-dataset evaluation (train on corpus A, test on corpus B) is the honest test
    of generalisation: within a single dataset, shared capture conditions make the task
    easier than production traffic.
    """
    query_manifest_obj = Manifest.load(query_manifest)
    gallery_manifest_obj = Manifest.load(gallery_manifest)
    query_records = list(query_manifest_obj)
    gallery_records = list(gallery_manifest_obj)
    if not query_records or not gallery_records:
        raise DataError("Query and gallery manifests must both be non-empty")

    query_labels = np.array([r.identity or "" for r in query_records])
    gallery_labels = np.array([r.identity or "" for r in gallery_records])
    self_mask = np.array(
        [[a.image_id == b.image_id for b in gallery_records] for a in query_records],
        dtype=bool,
    )
    shared = set(query_labels.tolist()) & set(gallery_labels.tolist())
    if not shared:
        raise DataError(
            "Query and gallery share no identities; cross-dataset evaluation needs at "
            "least one common individual"
        )
    LOGGER.info(
        "Cross-dataset protocol: %d queries, %d gallery images, %d shared identities",
        len(query_records), len(gallery_records), len(shared),
    )
    return Split(
        name=name,
        query_records=query_records,
        gallery_records=gallery_records,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        self_occurrences=self_mask,
    )


def build_within_split(
    records: Sequence[Any],
    queries_per_identity: int = 1,
    name: str = "within",
    seed: int = 1337,
    max_gallery_per_identity: int | None = None,
) -> Split:
    """Build a query/gallery protocol from one identity-labelled record set."""
    return build_identity_split(
        records,
        queries_per_identity=queries_per_identity,
        max_gallery_per_identity=max_gallery_per_identity,
        name=name,
        seed=seed,
    )


def prepare_calfw_pairs(
    config: PipelineConfig,
    parquet_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Materialise CALFW verification pairs as an image directory plus a CSV.

    CALFW supplies labelled same/different pairs but no gallery, so it can only drive
    *verification* metrics (ROC-AUC, EER, TAR@FAR). Its value is that it is completely
    independent of the corpus the model was trained on, and that it exposes the failure
    mode a user actually feels: at a 1 % false-accept budget, how many genuine matches
    are missed?

    Args:
        config: Pipeline config (paths).
        parquet_path: The downloaded ``calfw.parquet``.
        limit: Optional cap on the number of pairs, for a quick run.

    Returns:
        A report with the pair count, class balance and where the files were written.
    """
    from .data.annotation import load_parquet_pairs, write_pairs_csv

    output_dir = Path(config.data.root) / "calfw"
    pairs = load_parquet_pairs(parquet_path, limit=limit)
    positives = sum(pair.label for pair in pairs)
    csv_path = write_pairs_csv(pairs, output_dir / "pairs", output_dir / "pairs.csv")

    report = {
        "source": "calfw",
        "parquet": str(parquet_path),
        "pairs": len(pairs),
        "positive_pairs": positives,
        "negative_pairs": len(pairs) - positives,
        "positive_ratio": round(positives / len(pairs), 4),
        "csv": str(csv_path),
        "images_dir": str(output_dir / "pairs"),
        "license": "See dataset card (research use)",
        "protocol": "verification — AUC / EER / TAR@FAR; supplies no gallery",
    }
    LOGGER.info(
        "Prepared CALFW: %d pairs (%d same / %d different)",
        report["pairs"], report["positive_pairs"], report["negative_pairs"],
    )
    return report


def make_embedder(
    config: PipelineConfig,
    num_classes: int = 0,
    device: str | None = None,
    checkpoint: str | Path | None = None,
) -> Embedder:
    """Build an :class:`Embedder` from the pipeline config, or load a checkpoint."""
    resolved_device = device or config.resolve_device()
    if checkpoint is not None:
        embedder = Embedder.load(checkpoint, device=resolved_device)
        LOGGER.info(
            "Loaded checkpoint %s (backbone=%s, dim=%d)",
            checkpoint, embedder.config.backbone, embedder.config.embedding_dim,
        )
        return embedder

    embedder_config = EmbedderConfig(
        backbone=config.model.backbone,
        pretrained=str(config.model.pretrained) if config.model.pretrained else None,
        timm_name=config.model.timm_name,
        embedding_dim=config.model.embedding_dim,
        pooling=config.model.pooling,
        head=config.model.head,
        head_margin=config.model.head_margin,
        head_scale=config.model.head_scale,
        head_num_subcenters=config.model.head_num_subcenters,
        image_size=config.model.image_size,
        l2_normalize=config.model.l2_normalize,
        tta=tuple(config.model.tta),
        gradient_checkpointing=config.model.gradient_checkpointing,
    )
    embedder = Embedder(embedder_config, num_classes=num_classes, device=resolved_device)
    LOGGER.info(
        "Built embedder: backbone=%s pooling=%s head=%s dim=%d device=%s",
        embedder_config.backbone, embedder_config.pooling, embedder_config.head,
        embedder_config.embedding_dim, resolved_device,
    )
    return embedder


def train_pipeline(
    config: PipelineConfig,
    train_manifest: Path,
    splits_dir: Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Train a metric-learning embedder on the ``train`` identity split."""
    train_records = load_splits_from_manifest(train_manifest, splits_dir, "train")
    val_records = load_splits_from_manifest(train_manifest, splits_dir, "val")
    if len(train_records) < 8:
        raise DataError(f"Only {len(train_records)} training tiles; need more to train")
    assert_identity_disjoint(
        {"train": train_records, "val": val_records, "test": load_splits_from_manifest(train_manifest, splits_dir, "test")}
    )

    identities = sorted({r.identity for r in train_records if r.identity})
    val_split = build_within_split(val_records, queries_per_identity=1, name="val", seed=config.seed)
    resolved_device = device or config.resolve_device()

    embedder = make_embedder(config, num_classes=len(identities), device=resolved_device)
    train_config = TrainConfigResolved(
        loss=config.train.loss,
        epochs=config.train.epochs,
        batch_size=config.train.identities_per_batch * config.train.samples_per_identity,
        lr=config.train.lr,
        backbone_lr_scale=config.train.backbone_lr_scale,
        weight_decay=config.train.weight_decay,
        warmup_epochs=config.train.warmup_epochs,
        scheduler=config.train.scheduler,
        label_smoothing=config.train.label_smoothing,
        triplet_weight=config.train.triplet_weight,
        triplet_margin=config.train.triplet_margin,
        amp=config.train.amp,
        grad_clip=config.train.grad_clip,
        ema_decay=config.train.ema_decay,
        identities_per_batch=config.train.identities_per_batch,
        samples_per_identity=config.train.samples_per_identity,
        val_every=config.train.val_every,
        early_stop_patience=config.train.early_stop_patience,
        num_workers=config.train.num_workers,
        seed=config.train.seed,
        image_size=config.model.image_size,
        output_dir=config.train.output_dir,
        fingerprint=config.fingerprint(),
    )

    history, trainer = train_metric_learner(
        embedder, train_records, val_split, train_config, device=resolved_device
    )
    # ``export_best`` writes the validated weights and leaves the resumable state in place,
    # so this path is also pause/resume capable.
    checkpoint = trainer.export_best(config.train.output_dir / "best.pt")
    return {
        "checkpoint": str(checkpoint),
        "train_images": len(train_records),
        "train_identities": len(identities),
        "val_images": len(val_records),
        "best_epoch": history.best_epoch,
        "best_val_recall_at_1": history.best_score,
        "stopped_early": history.stopped_early,
        "epochs": len(history.epochs),
        "history": str(config.train.output_dir / "training_history.json"),
    }


def acquire_pipeline(
    config: PipelineConfig,
    keys: Sequence[str] | None = None,
    extract: bool = True,
) -> dict[str, Any]:
    """Download and verify the requested datasets."""
    store = DatasetStore(config.data.root / "raw")
    report: dict[str, Any] = {}
    for key in keys or list(CATALOGUE):
        spec = CATALOGUE[key]
        try:
            if extract and any(a.kind in ("tar.gz", "zip") for a in spec.artifacts):
                path = store.ensure(key)
                report[key] = {"status": "ok", "path": str(path)}
            else:
                for artifact in spec.artifacts:
                    if artifact.kind in ("parquet",):
                        from .data.sources import download

                        target = store.archive_path(spec, artifact)
                        download(artifact, target)
                report[key] = {"status": "downloaded-no-extract"}
        except Exception as exc:
            LOGGER.error("Acquisition failed for %s: %s", key, exc)
            report[key] = {"status": "failed", "error": str(exc)}
    return report


def environment_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Report what is present and usable, without changing anything.

    This is the command to run first on an unfamiliar machine: it prints the device,
    the installed libraries, and for each dataset and artifact whether it is usable.
    """
    from .logging_utils import environment_report

    report: dict[str, Any] = {
        "run_id": new_run_id(),
        "config_fingerprint": config.fingerprint(),
        "device": config.resolve_device(),
        "environment": environment_report(),
        "paths": {
            "data_root": str(config.data.root),
            "manifest_dir": str(config.data.manifest),
            "crops_dir": str(config.data.crops_dir),
            "artifacts": str(config.output_dir),
        },
        "datasets": DatasetStore(config.data.root / "raw").status(),
        "backbones": [],
        "issues": [],
    }

    try:
        from .models.backbone import available_backbones

        report["backbones"] = available_backbones()
    except Exception as exc:
        report["issues"].append(f"torch/torchvision not importable: {exc}")

    # Data readiness checks, with the exact failing path in the message.
    manifest_dir = Path(config.data.manifest)
    report["manifests"] = {
        path.stem: path.stat().st_size for path in sorted(manifest_dir.glob("*_manifest.jsonl"))
    } if manifest_dir.is_dir() else {}
    for source in SOURCE_LAYOUTS:
        try:
            if source == "oiid_cat":
                report.setdefault("plans", {})[source] = resolve_oiid_paths(config).describe()
            elif source == "cat_individuals":
                found = locate_cat_individuals(config)
                report.setdefault("plans", {})[source] = {
                    "root": str(found) if found else None,
                    "available": found is not None,
                }
        except Exception as exc:
            report.setdefault("plans", {})[source] = {"error": str(exc)}
    return report


__all__ = [
    "SOURCE_LAYOUTS",
    "DatasetPlan",
    "acquire_pipeline",
    "build_eval_split",
    "build_within_split",
    "environment_pipeline",
    "load_oiid_head_boxes",
    "load_splits_from_manifest",
    "locate_calfw",
    "locate_cat_individuals",
    "make_embedder",
    "prepare_calfw_pairs",
    "prepare_dataset",
    "resolve_oiid_paths",
    "train_pipeline",
]
