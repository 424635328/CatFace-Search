"""``catface`` command-line interface.

One binary, one config object, explicit stages. Every command prints a JSON report and
writes it to ``artifacts/<run>/``, so a CI job or a reviewer can diff two runs.

Command map::

    catface doctor                      # environment + data readiness, changes nothing
    catface acquire [--datasets ...]    # download + verify + extract
    catface prepare --source oiid_cat    # crops + manifest + identity-disjoint split
    catface train                       # metric learning on the train identities
    catface benchmark --models ...      # the comparison table that justifies the upgrade
    catface index build|search           # production vector index
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import PipelineConfig, dump_config, load_config
from .errors import CatFaceError, DataError
from .logging_utils import configure_logging, configure_utf8_console, get_logger, new_run_id

LOGGER = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="catface",
        description="Cat-face identity retrieval: prepare, train, benchmark, index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config file (defaults are used when omitted)")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data.root, the base directory for all corpora")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output_dir for artifacts")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--json-logs", action="store_true",
                        help="Emit machine-readable JSON log lines")
    parser.add_argument("--version", action="store_true", help="Print the version and exit")

    sub = parser.add_subparsers(dest="command", required=False)

    # -- doctor ------------------------------------------------------------
    sub.add_parser("doctor", help="Report environment and data readiness")

    # -- acquire -----------------------------------------------------------
    acquire = sub.add_parser("acquire", help="Download and verify datasets")
    acquire.add_argument("--datasets", nargs="*", default=None,
                         help="Dataset keys; all known datasets when omitted")
    acquire.add_argument("--no-extract", action="store_true")

    # -- prepare -----------------------------------------------------------
    prepare = sub.add_parser("prepare", help="Crop faces and build the manifest")
    prepare.add_argument("--source", default="oiid_cat",
                         choices=["oiid_cat", "cat_individuals"])
    prepare.add_argument("--force", action="store_true",
                         help="Re-crop even when crops already exist")

    # -- verify ------------------------------------------------------------
    verify = sub.add_parser(
        "verify",
        help="Benchmark same/different pair verification (AUC, EER, TAR@FAR)",
    )
    verify.add_argument("--models", nargs="+", required=True,
                        help="Model specs as name=backbone[:checkpoint]")
    verify.add_argument("--pairs", required=True,
                        help="Pairs CSV with columns path_a,path_b,label")
    verify.add_argument("--limit", type=int, default=None,
                        help="Use only the first N pairs")
    verify.add_argument("--batch-size", type=int, default=64)
    verify.add_argument("--device", default=None)

    # -- train -------------------------------------------------------------
    train = sub.add_parser("train", help="Train the metric-learning embedder")
    train.add_argument("--backbone", default=None, help="Override model.backbone")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--loss", default=None,
                       choices=["arcface", "cosface", "subcenter_arcface", "triplet", "ce"])
    train.add_argument("--identities-per-batch", type=int, default=None)
    train.add_argument("--samples-per-identity", type=int, default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--seed", type=int, default=None)

    # -- benchmark ---------------------------------------------------------
    bench = sub.add_parser("benchmark", help="Compare model configurations")
    bench.add_argument("--models", nargs="+", required=True,
                       help=("Model specs as name=backbone[:checkpoint]. Examples: "
                             "baseline=resnet50  dino=dinov2_vitb14  "
                             "finetuned=dinov2_vitb14:artifacts/train/best.pt"))
    bench.add_argument("--protocol", default="oiid_identity",
                       choices=["oiid_identity", "cat_individuals", "cross_dataset"],
                       help="Which query/gallery protocol to evaluate")
    bench.add_argument("--queries-per-identity", type=int, default=1)
    bench.add_argument("--max-gallery-per-identity", type=int, default=None)
    bench.add_argument("--whiten", default="none", choices=["none", "pca", "pcaw"])
    bench.add_argument("--whiten-dim", type=int, default=0)
    bench.add_argument("--dba", action="store_true")
    bench.add_argument("--aqe", action="store_true")
    bench.add_argument("--batch-size", type=int, default=32)
    bench.add_argument("--image-size", type=int, default=None)
    bench.add_argument("--baseline", default=None, help="Model name to test others against")
    bench.add_argument("--tag", default=None, help="Suffix for the artifact directory")

    # -- index -------------------------------------------------------------
    index = sub.add_parser("index", help="Build or query a production vector index")
    index.add_argument("--checkpoint", required=True)
    index.add_argument("--manifest", required=True, help="Manifest whose records are indexed")
    index.add_argument("--kind", default="flat_ip",
                       choices=["flat_ip", "ivf_pq", "hnsw", "numpy"])
    index.add_argument("--query", default=None, help="Image path to search for")
    index.add_argument("--top-k", type=int, default=10)
    index.add_argument("--device", default=None)

    return parser


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Resolve CLI flags into one validated config object."""
    overrides: dict[str, Any] = {}
    if args.data_root is not None:
        overrides["data.root"] = args.data_root
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    config = load_config(args.config, overrides)

    # Command-specific flags are applied after parsing so validation still runs once.
    if args.command == "train":
        if args.backbone:
            config = _set(config, "model", "backbone", args.backbone)
        if args.loss:
            config = _set(config, "train", "loss", args.loss)
        if args.epochs is not None:
            config = _set(config, "train", "epochs", args.epochs)
        if args.lr is not None:
            config = _set(config, "train", "lr", args.lr)
        if args.identities_per_batch is not None:
            config = _set(config, "train", "identities_per_batch", args.identities_per_batch)
        if args.samples_per_identity is not None:
            config = _set(config, "train", "samples_per_identity", args.samples_per_identity)
        if args.seed is not None:
            config = _set(config, "train", "seed", args.seed)
    if args.command == "benchmark" and args.image_size is not None:
        config = _set(config, "model", "image_size", args.image_size)
    return config


def _set(config: PipelineConfig, section: str, key: str, value: Any) -> PipelineConfig:
    """Return a copy of ``config`` with one field replaced, re-validated."""
    target = getattr(config, section)
    if not hasattr(target, key):
        raise DataError(f"{section}.{key} is not a valid option")
    setattr(target, key, value)
    # Re-run __post_init__ validation on the mutated section and the root.
    target.__post_init__()
    config.__post_init__()
    return config


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_doctor(config: PipelineConfig, _args: argparse.Namespace) -> int:
    """Report environment and data readiness. Takes no command-specific arguments."""
    from .pipeline import environment_pipeline

    report = environment_pipeline(config)
    _emit(report, config, "doctor")
    issues = report.get("issues") or []
    if issues:
        LOGGER.error("Blocking issue(s) detected: %s", "; ".join(issues))
        return 1
    return 0


def cmd_acquire(config: PipelineConfig, args: argparse.Namespace) -> int:
    from .pipeline import acquire_pipeline

    report = acquire_pipeline(config, args.datasets, extract=not args.no_extract)
    _emit(report, config, "acquire")
    failed = [k for k, v in report.items() if v.get("status") == "failed"]
    if failed:
        LOGGER.error("Acquisition failed for: %s", ", ".join(failed))
        return 1
    return 0


def cmd_prepare(config: PipelineConfig, args: argparse.Namespace) -> int:
    from .pipeline import prepare_dataset

    report = prepare_dataset(config, args.source, force=args.force)
    _emit(report, config, f"prepare-{args.source}")
    return 0


def cmd_train(config: PipelineConfig, args: argparse.Namespace) -> int:
    from .pipeline import train_pipeline

    manifest = Path(config.data.manifest) / "oiid_cat_manifest.jsonl"
    splits_dir = Path(config.data.manifest) / "oiid_cat_splits"
    report = train_pipeline(config, manifest, splits_dir, device=args.device)
    _emit(report, config, "train")
    return 0


def cmd_benchmark(config: PipelineConfig, args: argparse.Namespace) -> int:
    from .benchmark_runner import run_benchmark

    report = run_benchmark(
        config,
        model_specs=args.models,
        protocol=args.protocol,
        queries_per_identity=args.queries_per_identity,
        max_gallery_per_identity=args.max_gallery_per_identity,
        whiten=args.whiten,
        whiten_dim=args.whiten_dim,
        dba=args.dba,
        aqe=args.aqe,
        batch_size=args.batch_size,
        baseline=args.baseline,
        tag=args.tag,
    )
    _emit(report, config, "benchmark")
    return 0


def cmd_verify(config: PipelineConfig, args: argparse.Namespace) -> int:
    from .benchmark_runner import run_verification

    report = run_verification(
        config,
        model_specs=args.models,
        pairs_csv=Path(args.pairs),
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
    )
    report.pop("table", None)
    _emit(report, config, "verify")
    print(report.get("table", ""))
    return 0


def cmd_index(config: PipelineConfig, args: argparse.Namespace) -> int:
    from .cli_index import build_and_search

    report = build_and_search(
        config,
        checkpoint=args.checkpoint,
        manifest_path=args.manifest,
        kind=args.kind,
        query=args.query,
        top_k=args.top_k,
        device=args.device,
    )
    _emit(report, config, "index")
    return 0


def _emit(report: dict[str, Any], config: PipelineConfig, stage: str) -> None:
    """Print the report and persist it next to the run config."""
    run_dir = Path(config.output_dir) / f"{stage}-{new_run_id()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "config": config.to_mapping(), "report": report}
    (run_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    dump_config(config, run_dir / "config.resolved.yaml")
    LOGGER.info("Report written to %s", run_dir / "report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    configure_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(__version__)
        return 0
    if not args.command:
        parser.print_help()
        return 2

    configure_logging(args.log_level, json_console=args.json_logs)

    try:
        config = build_config(args)
    except CatFaceError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 2

    commands = {
        "doctor": cmd_doctor,
        "acquire": cmd_acquire,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "benchmark": cmd_benchmark,
        "verify": cmd_verify,
        "index": cmd_index,
    }
    try:
        return commands[args.command](config, args)
    except CatFaceError as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
