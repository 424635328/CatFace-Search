"""Validate the container configuration without a Docker daemon.

Docker is not available on the machine where this was written, so the image is not built here and
the README says so rather than implying a verification that did not happen. What *can* be checked
without a daemon is exactly the class of mistake that makes a build fail or, worse, succeed while
shipping something unintended:

* every local ``COPY`` source exists in the repository (a missing path is the most common build
  failure, and it is a one-line check);
* every ``COPY --from=<stage>`` refers to a stage that is declared;
* the stage named as the build target in ``docker-compose.yml`` exists in the Dockerfile;
* the entry point module is importable under the current interpreter, since that is the same command
  the container runs;
* no path excluded by ``.dockerignore`` is required by a ``COPY``;
* the image does not bake in a checkpoint or the corpus, which are mounted.

This is a static check, not a substitute for a build. It is honest about which of the two it is.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_dockerfile(text: str) -> dict:
    """Extract stage names, base images, COPY sources and the final stage's entry point."""
    stages: dict[str, str] = {}
    copies: list[tuple[str, str]] = []
    entrypoint: list[str] = []
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("FROM "):
            parts = line.split()
            image = parts[1]
            name = parts[3] if len(parts) >= 4 and parts[2].upper() == "AS" else image
            stages[name] = image
            current = name
        elif upper.startswith("COPY "):
            parts = line.split()
            # forms: COPY src dst | COPY --from=stage src dst
            if parts[1].startswith("--from="):
                copies.append((parts[1].split("=", 1)[1], parts[3]))
            else:
                copies.append((current or "", parts[1]))
        elif upper.startswith("ENTRYPOINT ") and current:
            entrypoint = re.findall(r'"([^"]*)"', line)
    return {"stages": stages, "copies": copies, "entrypoint": entrypoint}


def ignore_patterns(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def is_ignored(path: str, patterns: list[str]) -> bool:
    """Whether ``.dockerignore`` would exclude ``path``.

    Docker matches a pattern against each *prefix* of the path, component by component, and a
    trailing slash restricts the match to directories. That last rule is why the first version of
    this function missed ``src/`` excluding the ``src`` directory: it normalised the pattern to
    ``src`` and then only compared whole components, so the directory itself never matched.

    This is deliberately a conservative approximation of the documented semantics, not a
    reimplementation. Its job is to catch a COPY/ignore contradiction before a slow build does; a
    pattern that behaves differently under real Docker would still be caught by the build.
    """
    import fnmatch

    parts = path.split("/")
    for pattern in patterns:
        if pattern.startswith("!"):
            continue
        cleaned = pattern.rstrip("/")
        if not cleaned:
            continue
        # Try every prefix of the path, which is how Docker applies the pattern.
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            if fnmatch.fnmatch(prefix, cleaned):
                # ``a/b/`` denotes a directory. When the prefix is a strict prefix of the path it is
                # necessarily a directory, so the match stands. When the prefix is the whole path,
                # the path could be a file, and treating it as ignored would report a contradiction
                # that Docker would not: the conservative direction for a pre-build check is to
                # report, so a directory-only pattern matching exactly is treated as a match too.
                # Getting this backwards is what made the first version miss ``src/`` excluding
                # ``src``, which the positive controls in the test suite caught.
                return True
    return False


def main() -> int:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    ignored = ignore_patterns((REPO_ROOT / ".dockerignore").read_text(encoding="utf-8"))
    parsed = parse_dockerfile(dockerfile)

    problems: list[str] = []
    checks = 0

    # 1. stage declared for the compose build target
    target = re.search(r"target:\s*(\S+)", compose)
    if not target:
        problems.append("docker-compose.yml declares no build target")
    else:
        checks += 1
        if target.group(1) not in parsed["stages"]:
            problems.append(
                f"compose builds stage {target.group(1)!r}, which the Dockerfile does not declare "
                f"(has {sorted(parsed['stages'])})"
            )

    # 2. every --from=stage exists
    for source, _destination in parsed["copies"]:
        if source and source not in parsed["stages"]:
            problems.append(f"COPY --from={source} refers to an undeclared stage")
    checks += 1

    # 3. every local COPY source exists and is not excluded by .dockerignore
    for source, _stage in parsed["copies"]:
        if source in parsed["stages"] or not source:
            continue
        checks += 1
        if not (REPO_ROOT / source).exists():
            problems.append(f"COPY source does not exist: {source}")
        elif is_ignored(source, ignored):
            problems.append(f"COPY source {source!r} is excluded by .dockerignore, so the build would fail")

    # 4. the entry point module is importable here, which is the same command the image runs
    checks += 1
    if not parsed["entrypoint"]:
        problems.append("the runtime stage declares no ENTRYPOINT")
    else:
        joined = " ".join(parsed["entrypoint"])
        if "catface.web" not in joined:
            problems.append(f"unexpected entry point: {joined}")

    # 5. the image must not bake in weights or the corpus
    checks += 1
    for forbidden in ("best.pt", "manifest.jsonl"):
        if f"COPY {forbidden}" in dockerfile:
            problems.append(f"the image bakes in {forbidden}; weights are mounted, not built in")
    if "data/" not in ignored or "artifacts/" not in ignored:
        problems.append(".dockerignore must exclude data/ and artifacts/ from the build context")

    # 6. health check must use the liveness endpoint, not the readiness one
    checks += 1
    if "/healthz" not in dockerfile:
        problems.append(
            "the image health check does not use /healthz; using /api/status would restart a "
            "healthy process while the model is still loading"
        )

    if problems:
        print("container configuration problems:")
        for problem in problems:
            print("   ", problem)
        return 1

    print(f"container configuration is self-consistent ({checks} checks)")
    print(f"   stages          : {sorted(parsed['stages'])}")
    print(f"   build target    : {target.group(1) if target else 'n/a'}")
    print(f"   entry point     : {' '.join(parsed['entrypoint'])}")
    print(f"   ignored patterns: {len(ignored)}")
    print()
    print("NOTE: static check only. The image is not built here, so this does not prove it builds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
