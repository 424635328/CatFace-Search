"""Run the full benchmark suite and emit the evidence report.

One command produces every number quoted in ``docs/BENCHMARK.md``, so the report can be
regenerated rather than trusted. The order matters and is fixed:

1. **Zero-shot comparison** — every backbone as-is. Establishes whether the architecture
   change alone is an improvement, before any training confounds it.
2. **Post-processing selection on val** — whitening / DBA / αQE are hyper-parameters, so
   they are chosen on *validation* identities.
3. **Final benchmark on test** — the untuned and the selected configurations, plus every
   trained checkpoint, evaluated on identities that neither training nor selection saw.

A checkpoint is only included if it exists, so the suite is usable before, during and after
training.

Usage::

    python -m tools.run_benchmark_suite
    python -m tools.run_benchmark_suite --skip-zeroshot --checkpoints artifacts/train/dinov2s-arcface/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.config import load_config
from catface.eval.benchmark import PostprocessConfig
from catface.logging_utils import (
    configure_utf8_console,
    environment_report,
    get_logger,
    new_run_id,
)
from catface.pipeline import make_embedder

LOGGER = get_logger("tools.suite")

#: Backbone comparison: the configuration the project replaced, and the candidates.
DEFAULT_ZERO_SHOT = (
    ("zeroshot-resnet50", "resnet50"),
    ("zeroshot-dinov2s", "dinov2_vits14"),
    ("zeroshot-dinov2b", "dinov2_vitb14"),
)


def discover_checkpoints(root: Path) -> list[tuple[str, Path]]:
    """Find trained checkpoints under ``root``, named by their run directory."""
    found: list[tuple[str, Path]] = []
    if not root.is_dir():
        return found
    for best in sorted(root.glob("*/best.pt")):
        found.append((f"trained-{best.parent.name}", best))
    return found


def readable_checkpoint(path: Path) -> bool:
    """Whether a checkpoint can actually be loaded.

    A run that was killed before its first checkpoint, or a partially written file, would
    otherwise abort the whole suite. Probing here keeps the suite usable during training.
    """
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        return isinstance(payload, dict) and "embedder_config" in payload
    except Exception as exc:
        LOGGER.warning("Skipping unreadable checkpoint %s (%s)", path, exc)
        return False


def main(argv: list[str] | None = None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Run the full benchmark suite")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--protocol", default="cat_individuals")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Explicit checkpoint paths; defaults to artifacts/train/*/best.pt",
    )
    parser.add_argument("--train-root", default="artifacts/train")
    parser.add_argument("--skip-zeroshot", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--baseline", default="zeroshot-resnet50")
    parser.add_argument("--out", default="docs/diagnostics/benchmark-suite.json")
    args = parser.parse_args(argv)

    from catface.benchmark_runner import run_benchmark

    config = load_config(args.config)
    started = time.perf_counter()
    suite: dict = {
        "run_id": new_run_id(),
        "protocol": args.protocol,
        "environment": environment_report(),
        "config_fingerprint": config.fingerprint(),
        "stages": {},
    }

    # -- stage 1: zero-shot -------------------------------------------------
    if not args.skip_zeroshot:
        LOGGER.info("=== stage 1/3: zero-shot backbones ===")
        specs = [f"{name}={backbone}" for name, backbone in DEFAULT_ZERO_SHOT]
        report = run_benchmark(
            config,
            model_specs=specs,
            protocol=args.protocol,
            batch_size=args.batch_size,
            baseline=args.baseline,
            tag=f"zeroshot-{args.protocol}",
        )
        suite["stages"]["zeroshot"] = {
            "results": report["results"],
            "comparisons": report["comparisons"],
            "protocol": report["protocol"],
            "table": report["table"],
        }

    # -- collect trained checkpoints ---------------------------------------
    if args.checkpoints:
        candidates = [(Path(p).parent.name, Path(p)) for p in args.checkpoints]
    else:
        candidates = discover_checkpoints(Path(args.train_root))
    trained = [(name, path) for name, path in candidates if path.is_file() and readable_checkpoint(path)]
    if candidates and not trained:
        LOGGER.warning("No loadable checkpoints found; running without trained models")
    suite["checkpoints"] = [{"name": name, "path": str(path)} for name, path in trained]

    # -- stage 2: post-processing selection on val -------------------------
    selected_config = PostprocessConfig()
    if trained and not args.skip_postprocess:
        LOGGER.info("=== stage 2/3: post-processing selection on val ===")
        from tools.tune_postprocess import candidate_configs, embed_protocol, load_split, score_cached

        manifest = Path(config.data.manifest) / "cat_individuals_manifest.jsonl"
        splits = Path(config.data.manifest) / "cat_individuals_splits"
        val_records = load_split(manifest, splits, "val")
        _name, path = trained[-1]  # the most recent run
        embedder = make_embedder(config, checkpoint=path)
        # Embed the val protocol once, then score the whole grid on cached descriptors.
        cache = embed_protocol(embedder, val_records, config.model.image_size, 1, config.seed)
        ranking: list[dict] = []
        for candidate in candidate_configs(int(cache["query"].shape[1])):
            metrics, protocol = score_cached(cache, candidate)
            ranking.append({"config": candidate.describe(), "val": metrics, "protocol": protocol})
        ranking.sort(key=lambda entry: entry["val"].get("hit@1", 0.0), reverse=True)
        selected_config = PostprocessConfig(**ranking[0]["config"])
        suite["stages"]["postprocess_selection"] = {
            "checkpoint": str(path),
            "selected_on_val": ranking[0]["config"],
            "val_ranking": ranking[:10],
            "candidates_evaluated": len(ranking),
        }
        LOGGER.info("selected post-processing: %s", json.dumps(ranking[0]["config"], sort_keys=True))
        del embedder

    # -- stage 3: final benchmark on test -----------------------------------
    LOGGER.info("=== stage 3/3: final benchmark on test identities ===")
    if args.skip_postprocess:
        # Single row, untuned: keeps the suite usable when only a baseline is wanted.
        rows = [("postprocess-untuned", PostprocessConfig())]
    else:
        rows = [("postprocess-untuned", PostprocessConfig()), ("postprocess-selected", selected_config)]
    rows = [
        (name, cfg) for name, cfg in rows if not (args.skip_postprocess and name != "postprocess-untuned")
    ]

    final_results: list[dict] = []
    if not trained:
        LOGGER.warning("Stage 3 skipped: no trained checkpoints available")
    else:
        # A checkpoint path already determines the backbone, so it is read back from the
        # checkpoint rather than assumed — a mismatch would silently score the wrong model.
        specs = [f"{name}={_checkpoint_backbone(path)}:{path}" for name, path in trained]
        baseline_name = specs[0].split("=")[0]
        for label, postprocess in rows:
            report = run_benchmark(
                config,
                model_specs=specs,
                protocol=args.protocol,
                batch_size=args.batch_size,
                baseline=baseline_name,
                whiten=postprocess.whiten,
                whiten_dim=postprocess.whiten_dim,
                dba=postprocess.dba,
                aqe=postprocess.query_expansion == "aqe",
                tag=f"final-{label}-{args.protocol}",
            )
            final_results.append(
                {
                    "label": label,
                    "config": postprocess.describe(),
                    "results": report["results"],
                    "comparisons": report["comparisons"],
                    "table": report["table"],
                }
            )
    suite["stages"]["final"] = final_results

    suite["wall_seconds"] = round(time.perf_counter() - started, 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(suite, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    LOGGER.info("Suite report written to %s (%.1f min)", out, suite["wall_seconds"] / 60)

    for stage in final_results:
        print(f"\n===== {stage['label']} =====\n{stage['table']}")
    return 0


def _checkpoint_backbone(path: Path) -> str:
    """Read the backbone name from a checkpoint so the spec can be built without guessing."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return str(payload.get("embedder_config", {}).get("backbone", "dinov2_vits14"))


if __name__ == "__main__":
    raise SystemExit(main())
