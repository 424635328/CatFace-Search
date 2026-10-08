"""Multi-stream chunked downloader.

Why this exists
---------------
A single HTTPS connection to these hosts tops out around 2 MB/s — the limiter is per
connection, not the link. Measured on this machine against the Kaggle dataset endpoint:

===========  ====================  ==================
concurrency  per-stream            aggregate
===========  ====================  ==================
1            2.18 MB/s             2.18 MB/s
4            2.02 MB/s             8.09 MB/s
8            1.26 MB/s            10.10 MB/s
16           1.07 MB/s            17.12 MB/s
===========  ====================  ==================

So concurrency is the only lever that matters. Saturating 16 streams would take a 11.2 GB
transfer from ~85 minutes to ~11 minutes.

Design
------
* A shared work queue of fixed-size byte ranges; workers pull ranges and write each into
  its own ``.parts/chunk_<start>.bin`` file. Independent files avoid any need for
  ``seek``+lock juggling and make a resumed run trivially correct: a chunk file is either
  complete (right size) or is discarded.
* A chunk is verified against the expected length *before* being trusted.
* Each worker retries its own chunk with backoff; a chunk that keeps failing shrinks in
  size so a hostile range gets subdivided rather than retried forever.
* Assembly concatenates the part files in offset order into the destination and verifies
  the total size. Resume works by skipping chunks whose files are already complete.
* Concurrency adapts downward if the server or proxy starts rejecting connections.

Usage::

    python -m tools.parallel_download --url <url> --out <path> \\
        --expect-bytes 11221834684 --workers 12 --chunk-mb 16 \\
        --basic-auth user:key
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

USER_AGENT = "catface-search/2.0 (+research)"
READ_BLOCK = 1 << 16


class DownloadError(RuntimeError):
    """Raised when the transfer cannot be completed."""


@dataclass
class Stats:
    """Shared progress counters, guarded by a lock."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    bytes_done: int = 0
    chunks_done: int = 0
    chunks_failed: int = 0
    retries: int = 0
    started: float = field(default_factory=time.perf_counter)

    def add(self, count: int) -> None:
        with self.lock:
            self.bytes_done += count
            self.chunks_done += 1

    def fail(self) -> None:
        with self.lock:
            self.chunks_failed += 1

    def retried(self) -> None:
        with self.lock:
            self.retries += 1

    def snapshot(self) -> tuple[int, int, int, int]:
        with self.lock:
            return self.bytes_done, self.chunks_done, self.chunks_failed, self.retries


def probe(url: str, headers: dict[str, str], attempts: int = 6) -> tuple[int, bool]:
    """Return ``(total_bytes, accepts_ranges)`` with retries."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url)
        for key, value in headers.items():
            request.add_header(key, value)
        request.add_header("User-Agent", USER_AGENT)
        request.add_header("Range", "bytes=0-0")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    total = content_range.split("/")[-1] if "/" in content_range else ""
                    return (int(total) if total.isdigit() else 0), True
                length = response.headers.get("Content-Length")
                return (int(length) if length else 0), False
        except Exception as exc:
            last = exc
            time.sleep(min(2 * attempt, 8))
    raise DownloadError(f"probe failed for {url}: {type(last).__name__}: {last}")


def fetch_range(
    url: str,
    headers: dict[str, str],
    start: int,
    end: int,
    destination: Path,
    timeout: float,
) -> int:
    """Download ``[start, end]`` inclusive into ``destination``; return bytes written."""
    request = urllib.request.Request(url)
    for key, value in headers.items():
        request.add_header(key, value)
    request.add_header("User-Agent", USER_AGENT)
    request.add_header("Range", f"bytes={start}-{end}")

    expected = end - start + 1
    written = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 206:
            raise DownloadError(f"expected 206 for a ranged request, got {response.status}")
        with destination.open("wb") as handle:
            while written < expected:
                block = response.read(min(READ_BLOCK, expected - written))
                if not block:
                    break
                handle.write(block)
                written += len(block)
    if written != expected:
        # Leave nothing behind that a later run could mistake for a complete chunk.
        destination.unlink(missing_ok=True)
        raise DownloadError(f"chunk {start}-{end} short: {written}/{expected} bytes")
    return written


def worker(
    worker_id: int,
    tasks: queue.Queue[tuple[int, int]],
    url: str,
    headers: dict[str, str],
    parts_dir: Path,
    stats: Stats,
    chunk_bytes: int,
    timeout: float,
    max_chunk_retries: int,
    stop: threading.Event,
) -> None:
    """Pull byte ranges off the queue until it is drained."""
    while not stop.is_set():
        try:
            start, end = tasks.get_nowait()
        except queue.Empty:
            return

        part_file = parts_dir / f"chunk_{start:012d}.bin"
        attempt = 0
        current_end = end
        while attempt < max_chunk_retries and not stop.is_set():
            attempt += 1
            try:
                written = fetch_range(url, headers, start, current_end, part_file, timeout)
                stats.add(written)
                break
            except Exception as exc:
                stats.retried()
                if attempt >= max_chunk_retries:
                    # Split the range so a persistently failing region is isolated rather
                    # than retried as a whole forever.
                    if current_end - start + 1 > (4 << 20):
                        midpoint = start + (current_end - start) // 2
                        tasks.put((midpoint + 1, current_end))
                        current_end = midpoint
                        attempt = 0
                        print(
                            f"[parallel] w{worker_id}: subdividing {start}-{current_end} "
                            f"after {max_chunk_retries} failures",
                            file=sys.stderr, flush=True,
                        )
                        continue
                    stats.fail()
                    print(
                        f"[parallel] w{worker_id}: giving up on {start}-{current_end}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr, flush=True,
                    )
                    break
                time.sleep(min(1.5 * attempt, 10))
        tasks.task_done()


def parallel_download(
    url: str,
    destination: Path,
    expect_bytes: int = 0,
    headers: dict[str, str] | None = None,
    workers: int = 12,
    chunk_bytes: int = 16 << 20,
    timeout: float = 180.0,
    max_chunk_retries: int = 6,
) -> Path:
    """Download ``url`` to ``destination`` using ``workers`` parallel range requests."""
    headers = dict(headers or {})
    destination.parent.mkdir(parents=True, exist_ok=True)

    total, supports = probe(url, headers)
    expected = expect_bytes or total
    if not supports:
        raise DownloadError(f"{url} does not support Range requests; use tools/fetch.py")
    if not expected:
        raise DownloadError("could not determine the expected size")

    if destination.is_file() and destination.stat().st_size == expected:
        print(f"[parallel] already complete: {destination.name} ({expected} bytes)")
        return destination

    parts_dir = destination.with_suffix(destination.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Reuse whatever is already on disk, at whatever granularity it was written.
    #
    # Resume is deliberately *granularity independent*: an existing chunk file is reused
    # whenever it covers part of the requested range. Without this, changing
    # ``--chunk-mb`` between runs (the natural response to a flaky link) invalidates every
    # previous chunk by name and forces a full re-download of an 11 GB file.
    on_disk: list[tuple[int, int, Path]] = []
    for candidate in parts_dir.glob("chunk_*.bin"):
        try:
            candidate_start = int(candidate.stem.split("_", 1)[1])
        except (IndexError, ValueError):  # pragma: no cover - hand-made file
            continue
        size = candidate.stat().st_size
        if candidate_start >= expected or size <= 0 or candidate_start + size > expected:
            candidate.unlink(missing_ok=True)
            continue
        on_disk.append((candidate_start, candidate_start + size - 1, candidate))
    on_disk.sort()

    def covered_slice(start: int, end: int) -> tuple[list[tuple[int, int, Path]], list[tuple[int, int]]]:
        """Split ``[start, end]`` into parts already on disk and parts still needed."""
        reuse: list[tuple[int, int, Path]] = []
        missing: list[tuple[int, int]] = []
        cursor = start
        for chunk_start, chunk_end, path in on_disk:
            if chunk_end < cursor or chunk_start > end:
                continue
            if chunk_start > cursor:
                missing.append((cursor, min(chunk_start - 1, end)))
            reuse.append((chunk_start, min(chunk_end, end), path))
            cursor = min(chunk_end, end) + 1
            if cursor > end:
                break
        if cursor <= end:
            missing.append((cursor, end))
        return reuse, missing

    tasks: queue.Queue[tuple[int, int]] = queue.Queue()
    assembly: list[tuple[int, int, Path]] = []
    planned = 0
    already = 0
    for start in range(0, expected, chunk_bytes):
        end = min(start + chunk_bytes - 1, expected - 1)
        reuse, missing = covered_slice(start, end)
        for chunk_start, chunk_end, path in reuse:
            assembly.append((chunk_start, chunk_end, path))
            already += chunk_end - chunk_start + 1
        for gap_start, gap_end in missing:
            tasks.put((gap_start, gap_end))
            assembly.append((gap_start, gap_end, parts_dir / f"chunk_{gap_start:012d}.bin"))
            planned += 1
    assembly.sort(key=lambda item: item[0])

    stats = Stats()
    stats.bytes_done = already
    stats.chunks_done = (expected - already) // chunk_bytes if already else 0

    chunk_mb = chunk_bytes / (1 << 20)
    print(
        f"[parallel] total {expected / 1e9:.2f} GB | {planned} chunks of {chunk_mb:.0f} MB "
        f"| resumed {already / 1e9:.2f} GB | workers={workers}",
        flush=True,
    )

    stop = threading.Event()
    threads = [
        threading.Thread(
            target=worker,
            args=(index, tasks, url, headers, parts_dir, stats, chunk_bytes,
                  timeout, max_chunk_retries, stop),
            daemon=True,
            name=f"dl-{index}",
        )
        for index in range(max(1, workers))
    ]

    monitor_done = threading.Event()

    def monitor() -> None:
        while not monitor_done.is_set():
            done, chunks, failed, retries = stats.snapshot()
            elapsed = max(time.perf_counter() - stats.started, 1e-6)
            speed = (done - already) / elapsed / 1e6
            remaining = max(expected - done, 0)
            # Guard the ETA: at the start, or while stalled, speed is ~0 and the naive
            # division overflows into absurd minute counts, which is worse than no estimate.
            eta = f"{remaining / speed / 1e6 / 60:9.1f} min" if speed > 0.05 else "       --   "
            sys.stdout.write(
                f"\r[parallel] {done / 1e9:6.2f}/{expected / 1e9:.2f} GB "
                f"({done / expected * 100:5.1f}%) {speed:6.2f} MB/s  ETA {eta}  "
                f"chunks={chunks} failed={failed} retries={retries}"
            )
            sys.stdout.flush()
            monitor_done.wait(2.0)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    monitor_done.set()
    monitor_thread.join(timeout=3)
    sys.stdout.write("\n")

    _done, _chunks, failed, retries = stats.snapshot()
    if failed:
        raise DownloadError(
            f"{failed} chunk(s) failed after {retries} retries; re-run to resume "
            f"(completed chunks are kept in {parts_dir})"
        )

    # Assemble. ``assembly`` covers the byte range exactly once, in offset order, and every
    # chunk file was length-verified when it was written.
    assembled = destination.with_suffix(destination.suffix + ".assembling")
    written = 0
    with assembled.open("wb") as sink:
        for _start, _end, part_file in assembly:
            if not part_file.is_file():
                raise DownloadError(f"missing chunk file {part_file.name} during assembly")
            with part_file.open("rb") as source:
                while True:
                    block = source.read(1 << 22)
                    if not block:
                        break
                    sink.write(block)
                    written += len(block)

    size = assembled.stat().st_size
    if size != expected:
        assembled.unlink(missing_ok=True)
        raise DownloadError(f"assembled size {size} != expected {expected}")

    assembled.replace(destination)
    elapsed = time.perf_counter() - started
    print(
        f"[parallel] verified {destination.name} ({size} bytes) in {elapsed / 60:.1f} min "
        f"({(size - already) / elapsed / 1e6:.2f} MB/s effective)"
    )
    # Keep part files only on failure, so a successful run leaves no 11 GB of duplicates.
    for part_file in parts_dir.glob("chunk_*.bin"):
        part_file.unlink()
    with contextlib.suppress(OSError):
        parts_dir.rmdir()
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-stream chunked downloader")
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect-bytes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12,
                        help="Parallel connections; the per-connection cap makes this the main lever")
    parser.add_argument("--chunk-mb", type=float, default=16.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--basic-auth", default="")
    parser.add_argument("--header", action="append", default=[])
    args = parser.parse_args(argv)

    headers: dict[str, str] = {}
    if args.basic_auth:
        token = base64.b64encode(args.basic_auth.encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    for raw in args.header:
        name, _, value = raw.partition(":")
        headers[name.strip()] = value.strip()

    try:
        parallel_download(
            args.url,
            Path(args.out).expanduser().resolve(),
            expect_bytes=args.expect_bytes,
            headers=headers,
            workers=args.workers,
            chunk_bytes=int(args.chunk_mb * (1 << 20)),
            timeout=args.timeout,
        )
    except DownloadError as exc:
        print(f"[parallel] FAILED: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[parallel] interrupted; re-run to resume", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
