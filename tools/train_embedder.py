"""Train one metric-learning embedder and save a self-describing checkpoint.

A script rather than only a CLI flag because the exact recipe behind a reported number
should be a reviewable file. Every run writes:

``<output>/best.pt``
    Checkpoint with the embedder config, class names and training summary embedded.
``<output>/training_history.json``
    Per-epoch loss and held-out identity retrieval.
``<output>/run.json``
    Config fingerprint, protocol, environment and the resulting headline metric.

Usage::

    python -m tools.train_embedder --backbone dinov2_vits14 --epochs 20 \\
        --manifest data/manifests/cat_individuals_manifest.jsonl \\
        --splits data/manifests/cat_individuals_splits --output artifacts/train/dinov2s
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.config import load_config
from catface.data.manifest import Manifest
from catface.errors import CatFaceError
from catface.eval.protocols import assert_identity_disjoint, build_identity_split
from catface.logging_utils import (
    configure_utf8_console,
    environment_report,
    get_logger,
    new_run_id,
)
from catface.models.embedder import Embedder, EmbedderConfig, embed_records
from catface.train.loop import TrainConfigResolved, train_metric_learner

LOGGER = get_logger("tools.train")


def load_split(manifest_path: Path, splits_dir: Path, split: str) -> list:
    """Load the records belonging to one identity-disjoint split."""
    manifest = Manifest.load(manifest_path)
    split_file = splits_dir / f"{split}.txt"
    if not split_file.is_file():
        raise CatFaceError(f"split file not found: {split_file}")
    ids = {line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    records = [r for r in manifest if r.image_id in ids]
    if not records:
        raise CatFaceError(f"split {split!r} selected no records from {manifest_path}")
    return records


def main(argv: list[str] | None = None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Train a metric-learning embedder")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--timm-name", default=None)
    parser.add_argument("--manifest", default="data/manifests/cat_individuals_manifest.jsonl")
    parser.add_argument("--splits", default="data/manifests/cat_individuals_splits")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument(
        "--loss", default="arcface", choices=["arcface", "cosface", "subcenter_arcface", "triplet", "linear"]
    )
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--scale", type=float, default=64.0)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--pooling", default="auto")
    parser.add_argument("--identities-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-identity", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--triplet-weight", type=float, default=0.3)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    # -- pause / resume -------------------------------------------------------
    parser.add_argument(
        "--resume", action="store_true", help="Continue from <output>/train_state.pt if present"
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Wall-clock budget; pauses cleanly at the next epoch boundary",
    )
    parser.add_argument(
        "--ignore-signals",
        action="store_true",
        help="Do not install SIGINT/SIGTERM pause handlers (for tests)",
    )
    args = parser.parse_args(argv)

    base = load_config(args.config)
    device = args.device or base.resolve_device()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_split(Path(args.manifest), Path(args.splits), "train")
    val_records = load_split(Path(args.manifest), Path(args.splits), "val")
    test_records = load_split(Path(args.manifest), Path(args.splits), "test")
    # Re-asserted here, not merely assumed: a leaky split silently inflates every number.
    assert_identity_disjoint({"train": train_records, "val": val_records, "test": test_records})

    identities = sorted({r.identity for r in train_records if r.identity})
    LOGGER.info(
        "train: %d images / %d identities | val: %d images | test: %d images | device=%s",
        len(train_records),
        len(identities),
        len(val_records),
        len(test_records),
        device,
    )

    val_split = build_identity_split(val_records, queries_per_identity=1, seed=args.seed, name="val")

    embedder_config = EmbedderConfig(
        backbone=args.backbone,
        timm_name=args.timm_name,
        embedding_dim=args.embedding_dim,
        pooling=args.pooling,
        head=args.loss if args.loss != "triplet" else "linear",
        head_margin=args.margin,
        head_scale=args.scale,
        image_size=args.image_size,
        tta=tuple(base.model.tta),
        gradient_checkpointing=args.gradient_checkpointing,
    )
    embedder = Embedder(embedder_config, num_classes=len(identities), device=device)
    LOGGER.info(
        "embedder: backbone=%s pooling=%s backbone_dim=%d projected_dim=%d params=%.1fM",
        args.backbone,
        embedder.pooling,
        embedder.backbone.feature_dim,
        embedder.head.embedding_dim,
        sum(p.numel() for p in embedder.parameters()) / 1e6,
    )

    train_config = TrainConfigResolved(
        loss=args.loss,
        epochs=args.epochs,
        batch_size=args.identities_per_batch * args.samples_per_identity,
        lr=args.lr,
        backbone_lr_scale=args.backbone_lr_scale,
        weight_decay=base.train.weight_decay,
        warmup_epochs=args.warmup_epochs,
        scheduler=base.train.scheduler,
        label_smoothing=base.train.label_smoothing,
        triplet_weight=args.triplet_weight,
        triplet_margin=base.train.triplet_margin,
        amp=base.train.amp,
        grad_clip=base.train.grad_clip,
        ema_decay=args.ema_decay,
        identities_per_batch=args.identities_per_batch,
        samples_per_identity=args.samples_per_identity,
        val_every=1,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        output_dir=output_dir,
        # Recorded in the resumable state so a resumed run can be checked against the
        # configuration it started with.
        fingerprint=base.fingerprint(),
    )

    started = time.perf_counter()
    history, trainer = train_metric_learner(
        embedder,
        train_records,
        val_split,
        train_config,
        device=device,
        resume=args.resume,
        max_seconds=args.max_seconds,
        handle_signals=not args.ignore_signals,
    )
    elapsed = time.perf_counter() - started

    # The best-validation weights are exported either way: after a completed run they are
    # the final artifact, and after a pause they are the best seen so far. The *latest*
    # weights live in train_state.pt and are what a resume continues from.
    checkpoint = trainer.export_best()
    paused = history.pause_reason is not None

    # Score the held-out identities with the saved weights, so the report quotes a number
    # that comes from the artifact rather than from an in-memory model.
    reloaded = Embedder.load(checkpoint, device=device)
    query = embed_records(
        reloaded, [r.path for r in val_split.query_records], image_size=args.image_size, batch_size=32
    )
    gallery = embed_records(
        reloaded, [r.path for r in val_split.gallery_records], image_size=args.image_size, batch_size=32
    )
    similarity = query.vectors @ gallery.vectors.T
    from catface.eval.metrics import evaluate_retrieval

    self_mask = [[a == b for b in gallery.ids] for a in query.ids]
    import numpy as np

    mask = np.array(self_mask, dtype=bool)
    metrics = evaluate_retrieval(
        similarity,
        val_split.query_labels,
        val_split.gallery_labels,
        recall_ks=(1, 5, 10),
        exclude_self=mask if mask.any() else None,
    )

    completed_epochs = history.epochs[-1].epoch if history.epochs else 0
    run_report = {
        "run_id": new_run_id(),
        "backbone": args.backbone,
        "timm_name": args.timm_name,
        "pooling": embedder.pooling,
        "backbone_dim": int(embedder.backbone.feature_dim),
        "projected_dim": int(embedder.head.embedding_dim),
        "loss": args.loss,
        "margin": args.margin,
        "scale": args.scale,
        "epochs": args.epochs,
        "epochs_completed": completed_epochs,
        "lr": args.lr,
        "backbone_lr_scale": args.backbone_lr_scale,
        "identities_per_batch": args.identities_per_batch,
        "samples_per_identity": args.samples_per_identity,
        "batch_size": train_config.batch_size,
        "train_images": len(train_records),
        "train_identities": len(identities),
        "best_epoch": history.best_epoch,
        "best_hit_at_1": history.best_score,
        "stopped_early": history.stopped_early,
        # A paused run is explicitly NOT a finished one; reporting the reason separately
        # stops an interrupted experiment from being read as converged.
        "paused": paused,
        "pause_reason": history.pause_reason,
        "resumed_from_checkpoint": bool(args.resume),
        "wall_seconds": round(elapsed, 1),
        "checkpoint": str(checkpoint),
        "resumable_state": str(trainer.state_path),
        "validation_reloaded": metrics.to_dict(),
        "environment": environment_report(),
        "config_fingerprint": base.fingerprint(),
    }
    (output_dir / "run.json").write_text(
        json.dumps(run_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print(json.dumps(run_report, indent=2, default=str))
    if paused:
        print(
            f"\nPAUSED ({history.pause_reason}) at epoch {completed_epochs}/{args.epochs}.\n"
            f"Resume with:  python -m tools.train_embedder ... --resume --output {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
