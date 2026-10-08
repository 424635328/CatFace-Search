"""Apply the granularity-independent resume changes to tools/parallel_download.py.

Run once; kept as a script so the transformation is reviewable rather than an opaque
in-place edit.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent / "parallel_download.py"

OLD_PLAN = '''    # Build the work list, skipping chunks already downloaded completely.
    tasks: "queue.Queue[tuple[int, int]]" = queue.Queue()
    planned = 0
    already = 0
    for start in range(0, expected, chunk_bytes):
        end = min(start + chunk_bytes - 1, expected - 1)
        part_file = parts_dir / f"chunk_{start:012d}.bin"
        if part_file.is_file() and part_file.stat().st_size == end - start + 1:
            already += end - start + 1
            continue
        if part_file.is_file():
            part_file.unlink()  # partial chunk from an interrupted run
        tasks.put((start, end))
        planned += 1'''

NEW_PLAN = '''    # Reuse whatever is already on disk, at whatever granularity it was written.
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
        if candidate_start >= expected:
            continue
        span = min(chunk_bytes, expected - candidate_start)
        if candidate.stat().st_size == span:
            on_disk.append((candidate_start, candidate_start + span - 1, candidate))
        else:
            # Wrong-sized: cannot be trusted, so it must not contribute to assembly.
            candidate.unlink(missing_ok=True)
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

    tasks: "queue.Queue[tuple[int, int]]" = queue.Queue()
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
    assembly.sort(key=lambda item: item[0])'''

OLD_ASSEMBLE = '''    # Assemble. Chunks are concatenated in offset order; each was length-verified.
    assembled = destination.with_suffix(destination.suffix + ".assembling")
    written = 0
    with open(assembled, "wb") as sink:
        for start in range(0, expected, chunk_bytes):
            end = min(start + chunk_bytes - 1, expected - 1)
            part_file = parts_dir / f"chunk_{start:012d}.bin"
            if not part_file.is_file():
                raise DownloadError(f"missing chunk file {part_file.name} during assembly")'''

NEW_ASSEMBLE = '''    # Assemble. ``assembly`` covers the byte range exactly once, in offset order, and every
    # chunk file was length-verified when it was written.
    assembled = destination.with_suffix(destination.suffix + ".assembling")
    written = 0
    with open(assembled, "wb") as sink:
        for _start, _end, part_file in assembly:
            if not part_file.is_file():
                raise DownloadError(f"missing chunk file {part_file.name} during assembly")'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    for old, new, label in (
        (OLD_PLAN, NEW_PLAN, "planning"),
        (OLD_ASSEMBLE, NEW_ASSEMBLE, "assembly"),
    ):
        if old not in text:
            print(f"FAIL: {label} block not found (already patched?)")
            return 1
        text = text.replace(old, new, 1)
    TARGET.write_text(text, encoding="utf-8")
    print("patched parallel_download.py: granularity-independent resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
