"""Resumable HTTP download with strict size verification.

The project needs ~12 GB of third-party archives over links that truncate. This
module is the one place allowed to talk to the network, so the verification rules
live here rather than being re-derived in each downloader:

* ``Range`` resume is used, and the response code is inspected. A server that
  answers ``200`` to a ranged request is *rewinding the stream*, and continuing to
  append would silently corrupt the file — the classic way a 791 MB archive ends
  up as a 339 MB file that still un-gzips part way.
* Every transfer is checked against an expected byte count.
* Writes go to ``<name>.part`` and are renamed only after verification, so a
  crash never leaves a plausible-looking complete file.

Usage::

    python -m tools.fetch --url <url> --out <path> --expect-bytes 791918971
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1 << 20
USER_AGENT = "catface-search/2.0 (+research)"


class TruncatedDownloadError(RuntimeError):
    """Raised when fewer bytes arrive than the server declared."""


def _head(url: str, timeout: float = 30.0) -> tuple[int | None, bool]:
    """Return ``(content_length, accepts_ranges)`` for ``url``."""
    request = urllib.request.Request(url, method="HEAD")
    request.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            ranges = (response.headers.get("Accept-Ranges") or "").lower() == "bytes"
            return (int(length) if length else None), ranges
    except Exception:
        return None, False


def fetch(
    url: str,
    destination: Path,
    expect_bytes: int | None = None,
    max_attempts: int = 10,
    allow_resume: bool = True,
    timeout: float = 120.0,
) -> Path:
    """Download ``url`` to ``destination``, resuming and verifying.

    Args:
        url: Source URL.
        destination: Final path; a ``.part`` sibling is used during transfer.
        expect_bytes: Required final size. When ``None`` the server's
            ``Content-Length`` is used.
        max_attempts: Reconnect attempts before giving up.
        allow_resume: Set ``False`` to always restart from byte zero.
        timeout: Per-read socket timeout.

    Returns:
        ``destination`` once verified.

    Raises:
        TruncatedDownloadError: The transfer completed but the size does not match.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")

    declared, ranges_ok = _head(url)
    expected = expect_bytes or declared
    if ranges_ok is False:
        print(f"[fetch] server does not advertise Range support for {url}")

    if destination.is_file() and expected and destination.stat().st_size == expected:
        print(f"[fetch] already complete: {destination.name} ({expected} bytes)")
        return destination

    for attempt in range(1, max_attempts + 1):
        offset = part.stat().st_size if (allow_resume and part.is_file()) else 0
        if expected and offset == expected:
            break
        if expected and offset > expected:
            print(f"[fetch] {part.name} is larger than expected ({offset} > {expected}); restarting")
            part.unlink()
            offset = 0

        request = urllib.request.Request(url)
        request.add_header("User-Agent", USER_AGENT)
        if offset:
            request.add_header("Range", f"bytes={offset}-")

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                if offset and status != 206:
                    # The server ignored our Range header and is sending the whole
                    # file from byte 0. Appending would corrupt it — start over.
                    print(
                        f"[fetch] server answered {status} to a ranged request; "
                        "discarding partial data and restarting cleanly"
                    )
                    part.unlink(missing_ok=True)
                    offset = 0
                    continue
                total = expected or (
                    int(response.headers.get("Content-Length", "0")) + offset
                )
                mode = "ab" if offset else "wb"
                written = offset
                started = time.perf_counter()
                with part.open(mode) as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        written += len(block)
                        if total:
                            pct = written / total * 100
                            elapsed = max(time.perf_counter() - started, 1e-6)
                            speed = (written - offset) / elapsed / 1e6
                            sys.stdout.write(
                                f"\r[fetch] {destination.name}: {written / 1e6:.1f}/"
                                f"{total / 1e6:.1f} MB ({pct:5.1f}%) {speed:5.2f} MB/s"
                            )
                            sys.stdout.flush()
            sys.stdout.write("\n")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            sys.stdout.write("\n")
            print(f"[fetch] attempt {attempt}/{max_attempts} failed: {exc}")
            time.sleep(min(2 ** attempt, 20))
            continue

        size = part.stat().st_size
        if expected and size != expected:
            print(
                f"[fetch] incomplete: {size} of {expected} bytes "
                f"(short by {expected - size}); resuming"
            )
            continue
        break

    if not part.is_file():
        raise TruncatedDownloadError(f"No data was written for {url}")

    size = part.stat().st_size
    if expected and size != expected:
        raise TruncatedDownloadError(
            f"{destination.name} is {size} bytes but {expected} were expected after "
            f"{max_attempts} attempts"
        )

    part.replace(destination)
    print(f"[fetch] verified {destination.name} ({size} bytes) -> {destination}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumable verified HTTP download")
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect-bytes", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    try:
        fetch(
            args.url,
            Path(args.out).expanduser().resolve(),
            expect_bytes=args.expect_bytes or None,
            max_attempts=args.attempts,
            allow_resume=not args.no_resume,
        )
    except TruncatedDownloadError as exc:
        print(f"[fetch] FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
