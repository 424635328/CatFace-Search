"""Entry point: ``python -m catface.web``.

Kept import-light on purpose. ``uvicorn`` and ``fastapi`` are an optional dependency group, so this
module must be importable — and must fail with a usable message — on an installation that has only
the core requirements.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..logging_utils import configure_utf8_console, get_logger

LOGGER = get_logger("catface.web")

REPO_ROOT = Path(__file__).resolve().parents[3]


def _missing_dependency(error: ImportError) -> int:
    print(
        "The web interface needs FastAPI and uvicorn, which are an optional dependency group.\n"
        "\n"
        '  pip install -e ".[web]"\n'
        "\n"
        f"(import failed with: {error})",
        file=sys.stderr,
    )
    return 3


def main(argv: list[str] | None = None) -> int:
    configure_utf8_console()

    parser = argparse.ArgumentParser(
        prog="python -m catface.web",
        description="Serve the cat-face identity search UI and JSON API.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("CATFACE_CHECKPOINT", "artifacts/train/dinov2s-arcface/best.pt"),
        help="trained embedder; a model is required, the service cannot answer without one",
    )
    parser.add_argument(
        "--manifest",
        default=os.environ.get("CATFACE_MANIFEST", "data/manifests/cat_individuals_manifest.jsonl"),
        help="gallery manifest that defines the searchable identities",
    )
    parser.add_argument(
        "--device", default=os.environ.get("CATFACE_DEVICE", "cpu"), help="torch device, e.g. cpu or cuda"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=int(os.environ.get("CATFACE_IMAGE_SIZE", "224")),
        help="must match the resolution the checkpoint was trained at",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("CATFACE_HOST", "127.0.0.1"),
        help="bind address; the default is loopback, not 0.0.0.0",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("CATFACE_PORT", "8000")))
    parser.add_argument(
        "--gallery-root",
        default=os.environ.get("CATFACE_GALLERY_ROOT"),
        help="root that manifest image paths are relative to, for serving thumbnails",
    )
    parser.add_argument("--reload", action="store_true", help="auto-reload on source changes")
    args = parser.parse_args(argv)

    try:
        import uvicorn

        from .api import create_app
        from .service import SearchService
    except ImportError as error:  # pragma: no cover - depends on the installation
        return _missing_dependency(error)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = (REPO_ROOT / manifest).resolve()

    # Fail before binding the port when the inputs are obviously wrong. Starting a server that can
    # only answer 503 wastes the operator's time and looks like a service fault rather than a
    # misconfiguration.
    problems = []
    if not checkpoint.is_file():
        problems.append(f"checkpoint not found: {checkpoint}")
    if not manifest.is_file():
        problems.append(f"manifest not found: {manifest}")
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        print(
            "\nTrain a model first:  python -m tools.train_embedder --backbone dinov2_vits14 "
            "--output artifacts/train/run1\n"
            "Build the gallery:    python -m catface.cli prepare --source cat_individuals",
            file=sys.stderr,
        )
        return 2

    gallery_root = Path(args.gallery_root).resolve() if args.gallery_root else REPO_ROOT

    LOGGER.info("checkpoint: %s", checkpoint)
    LOGGER.info("manifest  : %s", manifest)
    LOGGER.info("device    : %s at %d px", args.device, args.image_size)

    service = SearchService(
        checkpoint=checkpoint,
        manifest=manifest,
        device=args.device,
        image_size=args.image_size,
        gallery_root=gallery_root,
    )
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
