"""Check that documentation references things that exist.

Documentation drifts silently. This project's README once documented a seven-step quick start
in which five steps could not run, and nothing failed until a reader tried them — the failure
surfaced as a user question, not as a test. A cheap existence check on the paths the README
quotes catches the common case before anyone reads it.

Scope is deliberately narrow: only repo-relative paths with a directory component are treated as
claims about the filesystem. A bare file name inside prose about an error message is not a
claim, and architecture diagrams quote paths relative to a package root.

Usage::

    python -m tools.check_docs
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Documents that make path claims, and the extensions worth checking.
DOCUMENTS = (
    "README.md",
    "docs/README.md",
    "docs/BENCHMARK.md",
    "docs/HYGIENE.md",
    "docs/PASS-CI-FIRST-TRY.md",
    "docs/archive/HISTORY.md",
    "docs/diagnostics/LABEL-COLLISIONS.md",
)
SUFFIXES = r"py|md|json|yaml|yml|jsonl|ps1|toml|txt|csv"

#: A path quoted in backticks.
QUOTED_PATH = re.compile(rf"`([A-Za-z0-9_./\\-]+\.(?:{SUFFIXES}))`")

#: A path written in prose without backticks, such as "see docs/archive/HISTORY.md".
#: This rule exists because the backtick-only rule missed a real reference: an outdated
#: ``docs/HISTORY.md`` sat in a comment inside a code block and nothing flagged it. Anchoring on
#: the repository's own top-level directories keeps ordinary prose from matching.
ROOT_DIRS = (".github", "configs", "docs", "scripts", "src", "tests", "tools")
_ROOT_ALTERNATION = "|".join(re.escape(name) for name in ROOT_DIRS)
BARE_PATH = re.compile(
    rf"(?<![\w./\\-])((?:{_ROOT_ALTERNATION})"
    rf"(?:/[A-Za-z0-9_.\-]+)*/[A-Za-z0-9_.\-]+\.(?:{SUFFIXES}))"
)

#: Directory names whose contents are documentation-relative illustrations, not path claims.
#: ``catface/`` and ``query/`` appear in architecture diagrams and sample output that show how a
#: path looked on the machine that produced the text, not where a file lives in the repository.
IGNORED_PREFIXES = ("catface/", "query/")

#: A path quoted inside ``src/catface/data/`` style prose is written relative to a package root.
PACKAGE_ROOTS = ("src", "src/catface")


def references(text: str) -> set[str]:
    """Return every path this document appears to claim exists."""
    found = set(QUOTED_PATH.findall(text))
    # A backticked path is already covered; drop the bare-pattern match nested inside it so the
    # same reference is not reported twice.
    for candidate in BARE_PATH.findall(text):
        if not any(candidate in quoted for quoted in found):
            found.add(candidate)
    return found


def check_document(path: Path) -> list[str]:
    """Return the paths in ``path`` that do not resolve from anywhere sensible.

    A path is accepted when it resolves from the repository root or from the directory holding the
    document, because both conventions appear in this project: the README links ``docs/...`` from
    the root, while a document inside ``docs/`` links its neighbours as ``diagnostics/...``.
    """
    text = path.read_text(encoding="utf-8")
    missing: list[str] = []
    for raw in sorted(references(text)):
        relative = raw.replace("\\", "/").removeprefix("./")
        if any(relative.startswith(prefix) for prefix in IGNORED_PREFIXES):
            continue
        # A bare file name is usually prose, not a path claim.
        if "/" not in relative:
            continue
        candidates = [REPO_ROOT / relative, path.parent / relative]
        candidates += [REPO_ROOT / root / relative for root in PACKAGE_ROOTS]
        if not any(candidate.exists() for candidate in candidates):
            missing.append(f"{path.name}: {relative}")
    return missing


def main() -> int:
    missing: list[str] = []
    scanned = 0
    for name in DOCUMENTS:
        document = REPO_ROOT / name
        if not document.is_file():
            continue
        scanned += 1
        missing.extend(check_document(document))

    if missing:
        print("documentation quotes paths that do not exist:")
        for item in missing:
            print("   ", item)
        print("Fix the document, or move the file it refers to.")
        return 1

    print(f"documentation path references check out ({scanned} documents scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
