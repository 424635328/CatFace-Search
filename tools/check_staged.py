"""Detect staged files that should not be committed.

The pre-commit hook is the last line of defence in this repository: it runs even when someone
pushes with ``-SkipPreflight``, which removes every other gate. Shell is a poor place for logic
that needs testing, so the detection lives here and the hook only calls it.

This check is not hypothetical. A one-line scratch file named ``tmp-message-probe.txt`` was
created to exercise the sync script, swept up by its ``git add -A``, committed four times and
pushed to the public remote before anyone looked. Nothing noticed, because ``-SkipPreflight`` had
switched off the only check that would have.

Usage::

    python -m tools.check_staged            # inspect the git index
    python -m tools.check_staged --paths a.txt b/tmp-c.txt   # inspect given paths (for tests)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Path shapes that indicate scratch, probe, or editor-leftover content rather than source.
#: Matched against each path component and the basename, never as a loose substring, so a real
#: file such as ``tools/check_staged.py`` cannot be caught by accident.
SCRATCH_NAMES = frozenset({"probe", "scratch", "tmp", "temp", "junk", "untitled"})
SCRATCH_PREFIXES = ("tmp-", "probe-", "scratch-", "debug-", "untitled-")
SCRATCH_SUFFIXES = ("~", ".orig", ".rej", ".bak", ".swp", ".tmp")


def is_scratch(relative: str) -> bool:
    """Whether ``relative`` looks like a file that was only ever meant to be temporary."""
    path = Path(relative.replace("\\", "/"))
    lowered = [part.lower() for part in path.parts]

    if path.name.lower().endswith(SCRATCH_SUFFIXES):
        return True

    # The basename with its extension removed, e.g. "tmp-message-probe" from "tmp-message-probe.txt".
    stem = path.stem.lower()
    if stem.startswith(SCRATCH_PREFIXES):
        return True
    if stem in SCRATCH_NAMES:
        return True

    # A directory literally named "tmp" or "scratch" anywhere in the path.
    return any(part in {"tmp", "temp", "scratch"} for part in lowered[:-1])


def staged_paths() -> list[str]:
    """Return the paths staged in the index, or an empty list outside a repository."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def check(paths: list[str]) -> list[str]:
    """Return the offending subset of ``paths``."""
    return [path for path in paths if is_scratch(path)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="check these paths instead of the git index (used by tests)",
    )
    args = parser.parse_args(argv)

    paths = staged_paths() if args.paths is None else list(args.paths)
    offenders = check(paths)

    if offenders:
        print("staged files that look like scratch or editor leftovers:")
        for path in offenders:
            print("   ", path)
        print("Delete them, gitignore them, or use 'git commit --no-verify' if one is real.")
        return 1

    if args.paths is None:
        print(f"no scratch files staged ({len(paths)} staged path(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
