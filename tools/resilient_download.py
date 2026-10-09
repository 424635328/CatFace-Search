"""Supervisor that drives :mod:`tools.chunk_download` to completion.

Corporate proxies and flaky CDNs close connections mid-transfer. A single
``chunk_download`` invocation can exhaust its per-window retries and exit non-zero even
though progress is resumable. This wrapper re-invokes it with fresh connections until
the destination reaches the expected size.

It exits non-zero only when it has made **no progress across several consecutive
rounds**, which distinguishes "slow network" from "this will never work".

Usage::

    python -m tools.resilient_download --url <url> --out <path> \
        --expect-bytes N --basic-auth user:key --rounds 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def current_size(destination: Path) -> int:
    """Bytes already written, counting the in-progress ``.part`` file."""
    part = destination.with_suffix(destination.suffix + ".part")
    if part.is_file():
        return part.stat().st_size
    if destination.is_file():
        return destination.stat().st_size
    return 0


def snapshot(*args: str, window_mb: float, timeout_s: float) -> subprocess.CompletedProcess:
    """Run one chunked-download round, tolerating either partial or complete success."""
    command = [
        sys.executable,
        str(HERE / "chunk_download.py"),
        *args,
        "--window-mb",
        str(window_mb),
    ]
    return subprocess.run(command, capture_output=True, text=True, timeout=None, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resilient chunked download supervisor")
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect-bytes", type=int, required=True)
    parser.add_argument("--basic-auth", default="")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--window-mb", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument(
        "--stall-rounds",
        type=int,
        default=6,
        help="Give up after this many consecutive rounds with no progress",
    )
    parser.add_argument("--pause", type=float, default=4.0)
    args = parser.parse_args(argv)

    # Export the credentials through the environment as well as the proxy: a plain
    # subprocess inherits them, and it keeps the key out of the process argument list.

    destination = Path(args.out).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    rounds = 0
    stalls = 0
    best = current_size(destination)
    started = time.time()

    while rounds < args.rounds:
        rounds += 1
        size = current_size(destination)
        if size >= args.expect_bytes:
            break
        if destination.is_file() and destination.stat().st_size == args.expect_bytes:
            break

        extra = []
        for raw in args.header:
            extra += ["--header", raw]
        command = [
            sys.executable,
            str(HERE / "chunk_download.py"),
            "--url",
            args.url,
            "--out",
            str(destination),
            "--expect-bytes",
            str(args.expect_bytes),
            "--window-mb",
            str(args.window_mb),
            *extra,
        ]
        if args.basic_auth:
            command += ["--basic-auth", args.basic_auth]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        after = current_size(destination)
        pct = after / args.expect_bytes * 100
        elapsed_min = (time.time() - started) / 60
        print(
            f"[supervisor] round {rounds}: {after / 1e6:.1f} MB ({pct:.2f}%) after {elapsed_min:.1f} min",
            flush=True,
        )

        if after >= args.expect_bytes:
            break
        if after > best:
            best = after
            stalls = 0
        else:
            stalls += 1
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            print(
                f"[supervisor] no progress ({stalls}/{args.stall_rounds}); "
                f"last error: {tail[-1] if tail else 'n/a'}",
                flush=True,
            )
            if stalls >= args.stall_rounds:
                print("[supervisor] stalled; giving up", file=sys.stderr)
                return 2
        time.sleep(args.pause)

    final = current_size(destination)
    if destination.is_file() and destination.stat().st_size == args.expect_bytes:
        print(f"[supervisor] complete: {destination} ({final} bytes)")
        return 0
    print(
        f"[supervisor] incomplete: {final} of {args.expect_bytes} bytes",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
