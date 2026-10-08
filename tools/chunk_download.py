"""Chunked-range downloader that survives connections truncated at a fixed size.

Some hosts and corporate proxies close a large response after ~10 MB without an
error. ``curl --retry`` cannot help because the connection itself succeeded; the
body simply ended early. The workaround is to stop asking for the whole file:
request one bounded window at a time with ``Range`` and append it.

If the server ignores ``Range`` and answers ``200``, this falls back to
single-stream mode and reports the truncation honestly rather than looping.

Example::

    python -m tools.chunk_download \
        --url https://www.kaggle.com/api/v1/datasets/download/timost1234/cat-individuals \
        --out data/raw/cat-individuals.zip --header "Authorization: Basic ..." \
        --window-mb 8 --expect-bytes 11247919277
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "catface-search/2.0 (+research)"


class RangeUnsupported(RuntimeError):
    """The server ignored the Range header, so windowed transfer is impossible."""


def probe(url: str, headers: dict[str, str], timeout: float = 30.0, attempts: int = 5) -> tuple[int | None, bool]:
    """Return ``(total_size, accepts_ranges)`` using a 1-byte ranged GET.

    Retried, because a transient TLS failure must not be mistaken for "this server does
    not support Range" — that misclassification silently disables the only transfer
    mode that survives a connection-truncating proxy.

    The second element is only ``False`` when the server *responded* and the response
    itself showed Range is unavailable. An unreachable server raises instead.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url)
        for key, value in headers.items():
            request.add_header(key, value)
        request.add_header("User-Agent", USER_AGENT)
        request.add_header("Range", "bytes=0-0")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    total = content_range.split("/")[-1] if "/" in content_range else None
                    return (int(total) if total and total.isdigit() else None), True
                # A 200 to a ranged request means the server ignored Range.
                length = response.headers.get("Content-Length")
                return (int(length) if length else None), False
        except urllib.error.HTTPError as exc:
            if exc.code in (200, 416):
                # 416 means Range is understood but the range is unsatisfiable.
                return None, exc.code == 416
            last_error = exc
        except Exception as exc:
            last_error = exc
        if attempt < attempts:
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(
        f"Could not probe {url} after {attempts} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def download_window(
    url: str,
    headers: dict[str, str],
    start: int,
    end: int,
    handle,
    timeout: float = 180.0,
) -> int:
    """Fetch ``[start, end]`` inclusive, append to ``handle``, return bytes written."""
    request = urllib.request.Request(url)
    for key, value in headers.items():
        request.add_header(key, value)
    request.add_header("User-Agent", USER_AGENT)
    request.add_header("Range", f"bytes={start}-{end}")
    written = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 206:
            raise RangeUnsupported(f"expected 206, got {response.status}")
        while True:
            block = response.read(1 << 18)
            if not block:
                break
            if written + len(block) > end - start + 1:
                block = block[: end - start + 1 - written]
            handle.write(block)
            written += len(block)
            if written >= end - start + 1:
                break
    return written


def chunk_download(
    url: str,
    destination: Path,
    headers: dict[str, str] | None = None,
    window_bytes: int = 8 << 20,
    expect_bytes: int | None = None,
    max_retries_per_window: int = 12,
    timeout: float = 180.0,
) -> Path:
    """Download ``url`` in ``window_bytes`` slices with per-window retries.

    Returns the verified destination path.
    """
    headers = dict(headers or {})
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")

    total, ranges_ok = probe(url, headers)
    expected = expect_bytes or total
    print(f"[chunk] server total={total} ranges={ranges_ok} expected={expected}", flush=True)
    if not ranges_ok:
        raise RangeUnsupported(
            f"{url} does not support Range requests; use tools/fetch.py instead"
        )
    if expected is None:
        raise RangeUnsupported("Could not determine the expected size; refusing to guess")

    if destination.is_file() and destination.stat().st_size == expected:
        print(f"[chunk] already complete: {destination}")
        return destination

    if not part.is_file():
        part.write_bytes(b"")

    started = time.perf_counter()
    while True:
        have = part.stat().st_size
        if have >= expected:
            break
        end = min(have + window_bytes - 1, expected - 1)
        attempt = 0
        while attempt < max_retries_per_window:
            attempt += 1
            try:
                with open(part, "ab") as handle:
                    written = download_window(url, headers, have, end, handle, timeout=timeout)
                if written == 0:
                    raise RuntimeError("window returned no data")
                break
            except RangeUnsupported:
                raise
            except Exception as exc:
                # A partially written window leaves the file longer than ``have``;
                # truncate back so the next attempt starts at a known offset.
                current = part.stat().st_size
                if current > have:
                    with open(part, "r+b") as handle:
                        handle.truncate(have)
                delay = min(1.5 * attempt, 15.0)
                print(f"[chunk] window {have}-{end} attempt {attempt} failed: "
                      f"{type(exc).__name__}: {exc}; retrying in {delay:.1f}s")
                time.sleep(delay)
        else:
            raise RuntimeError(f"window starting at {have} failed after "
                               f"{max_retries_per_window} attempts")

        have = part.stat().st_size
        elapsed = max(time.perf_counter() - started, 1e-6)
        speed = have / elapsed / 1e6
        remaining = (expected - have) / max(speed, 1e-6) / 60
        sys.stdout.write(
            f"\r[chunk] {have / 1e6:8.1f}/{expected / 1e6:.1f} MB "
            f"({have / expected * 100:5.1f}%) {speed:5.2f} MB/s  ETA {remaining:4.1f} min"
        )
        sys.stdout.flush()

    sys.stdout.write("\n")
    size = part.stat().st_size
    if size != expected:
        raise RuntimeError(f"finished with {size} bytes, expected {expected}")
    part.replace(destination)
    print(f"[chunk] verified {destination.name} ({size} bytes)")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunked range-resume downloader")
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect-bytes", type=int, default=0)
    parser.add_argument("--window-mb", type=float, default=8.0)
    parser.add_argument("--basic-auth", default="")
    """``user:password``, encoded into an Authorization header."""
    parser.add_argument("--header", action="append", default=[],
                        help="Extra header as 'Name: value' (repeatable)")
    args = parser.parse_args(argv)

    headers: dict[str, str] = {}
    if args.basic_auth:
        token = base64.b64encode(args.basic_auth.encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    for raw in args.header:
        name, _, value = raw.partition(":")
        headers[name.strip()] = value.strip()

    try:
        chunk_download(
            args.url,
            Path(args.out).expanduser().resolve(),
            headers=headers,
            window_bytes=int(args.window_mb * (1 << 20)),
            expect_bytes=args.expect_bytes or None,
        )
    except RangeUnsupported as exc:
        print(f"[chunk] FAILED: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"[chunk] FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
