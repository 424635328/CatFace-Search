"""Entry point: ``python -m catface.web``.

Kept import-light on purpose. ``uvicorn`` and ``fastapi`` are an optional dependency group, so this
module must be importable — and must fail with a usable message — on an installation that has only
the core requirements.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
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
    # 69 (EX_UNAVAILABLE) rather than 3, matching the launcher's "dependency missing" code.
    return 69


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
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CATFACE_PORT", "8000")),
        help="port to serve on; if it is busy the next free port is used unless --strict-port is set",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="fail instead of moving to another port when --port is already in use",
    )
    parser.add_argument(
        "--gallery-root",
        default=os.environ.get("CATFACE_GALLERY_ROOT"),
        help="root that manifest image paths are relative to, for serving thumbnails",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser window; use for a headless or scripted start",
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
        # 66 rather than 2: argparse already uses 2 for a usage error, and a launcher that prints
        # "2 = a path was wrong" would then mislabel every mistyped flag. Keeping these distinct is
        # what makes the codes usable by a caller instead of decorative.
        return 66

    gallery_root = Path(args.gallery_root).resolve() if args.gallery_root else REPO_ROOT

    LOGGER.info("checkpoint: %s", checkpoint)
    LOGGER.info("manifest  : %s", manifest)
    LOGGER.info("device    : %s at %d px", args.device, args.image_size)

    # Resolve the port before doing anything else, and report the one actually used. A busy default
    # port is a normal condition on a developer machine — on this one, port 8000 is held by another
    # product whose listener answers TCP with an HTTP 502, so a browser shows a gateway error and our
    # own bind fails. Refusing to start over that would make the launcher useless exactly where it is
    # meant to be convenient.
    port = _choose_port(args.host, args.port, args.strict_port)
    if port is None:
        return 66
    args.port = port
    LOGGER.info("serving   : http://%s:%d", args.host, port)

    service = SearchService(
        checkpoint=checkpoint,
        manifest=manifest,
        device=args.device,
        image_size=args.image_size,
        gallery_root=gallery_root,
    )
    app = create_app(service)

    if not args.no_browser:
        # Wait for the application to answer, then open the page. A fixed delay is wrong in both
        # directions: too short and the user gets "connection refused" before the socket is bound,
        # too long and they wait for nothing. Since the model loads in the background, the port binds
        # almost immediately and this returns as soon as /healthz responds. The URL uses the resolved
        # port, not the requested one.
        url = f"http://{args.host}:{port}"
        watcher = threading.Thread(
            target=_open_browser_when_serving, args=(args.host, port, url), daemon=True
        )
        watcher.start()

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")
    return 0


def _port_is_free(host: str, port: int) -> bool:
    """Whether ``port`` can be bound on ``host`` right now.

    Tries the real bind rather than a heuristic. A common alternative — checking whether something
    answers on the port — misses the case that actually bit here: another program (Incredibuild's
    Manager, in this environment) held 8000 and answered TCP, but with an HTTP 502, so "something
    responded" looked like success while our own bind failed with WinError 10013.
    """
    import socket

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    # getaddrinfo picks the right family so this works for IPv4 and IPv6 hosts alike.
    try:
        infos = socket.getaddrinfo(probe_host, port, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for family, socktype, proto, _canonical, address in infos:
        with socket.socket(family, socktype, proto) as probe:
            # No SO_REUSEADDR on purpose: on Windows it would let this probe succeed against a port
            # another process is already listening on, which is exactly the situation to detect.
            try:
                probe.bind(address)
            except OSError:
                continue
            return True
    return False


def _choose_port(host: str, preferred: int, strict: bool, attempts: int = 20) -> int | None:
    """Return a bindable port, preferring ``preferred``.

    Being unable to bind should not be a dead end for a one-click launcher: any other program on the
    machine can hold the default port, and making the user diagnose that is a poor trade for
    determinism that nobody asked for. ``--strict-port`` restores the strict behaviour for anything
    scripted that depends on the exact port.
    """
    for offset in range(attempts):
        candidate = preferred + offset
        if _port_is_free(host, candidate):
            if offset:
                LOGGER.warning("port %d is in use; using %d instead", preferred, candidate)
            return candidate
        if strict:
            LOGGER.error("port %d is already in use (--strict-port given); not moving", preferred)
            return None
    LOGGER.error("no free port in %d..%d", preferred, preferred + attempts - 1)
    return None


def _open_browser_when_serving(host: str, port: int, url: str, timeout_s: float = 120.0) -> None:
    """Open ``url`` once the server accepts connections.

    Polls the liveness endpoint rather than sleeping a fixed amount, because the page states its own
    readiness: opening it as soon as the socket is up shows the real banner ("not ready, model
    loading") instead of a browser error page, and no refresh is needed afterwards.
    """
    import time
    import urllib.error
    import urllib.request
    import webbrowser

    # 127.0.0.1 rather than the possibly-0.0.0.0 bind address: the latter is not a valid destination.
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{probe_host}:{port}/healthz", timeout=2) as reply:
                if reply.status == 200:
                    break
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)
    else:
        LOGGER.warning("server did not answer within %.0fs; open %s manually", timeout_s, url)
        return
    webbrowser.open(url)


if __name__ == "__main__":
    raise SystemExit(main())
