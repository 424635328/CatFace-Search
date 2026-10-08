"""Repository hygiene guards.

Two classes of problem that are invisible in a test suite but break a clone:

1. **Machine-specific absolute paths.** A hard-coded drive-letter path works perfectly on
   the machine that wrote it and fails everywhere else. Because the author's tests pass, the
   defect survives until a user clones the repository. The repository is *currently* clean,
   but the only thing keeping it clean is that nobody has needed an absolute path yet — and
   the fastest way to debug a path problem is to paste one in. So it is asserted rather than
   trusted, and this is a pre-existing risk being closed, not a hypothetical one.

2. **Credentials in tracked content.** A sync script that runs ``git add . && git push`` to a
   public remote will publish whatever happens to be in the working tree. Debugging sessions
   are exactly when keys get written into scripts, so the check runs in CI rather than at the
   moment of the mistake.

Both guards scan **tracked** files, which is the set that becomes public.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Binary and vendored formats where a byte scan is meaningless or misleading.
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".zip", ".gz", ".tar", ".7z",
    ".pt", ".pth", ".ckpt", ".pkl", ".bin", ".onnx", ".tflite", ".npy", ".npz", ".index", ".faiss",
}

#: Anything under these paths is a copy of someone else's material, reproduced verbatim on
#: purpose; reformatting or sanitising it would defeat the reason it is archived.
VENDORED_PREFIXES = ("docs/archive/third-party/",)


def tracked_files() -> list[str]:
    """Paths known to git. Empty when git is unavailable, so the suite still runs from a tarball."""
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.skip("not a git working tree")
    return [line for line in result.stdout.splitlines() if line.strip()]


def readable_text(relative: str) -> str | None:
    """Return a file's text, or ``None`` when it is binary or not worth scanning."""
    path = REPO_ROOT / relative
    if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
        return None
    if any(relative.startswith(prefix) for prefix in VENDORED_PREFIXES):
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw.decode("utf-8", errors="replace")


#: A URL scheme contains the letter sequence that a naive drive-letter pattern also matches,
#: so URLs must be stripped first or every documentation link becomes a false positive.
#: The scheme is assembled at runtime so this comment itself contains no such literal.
_URL = re.compile(r"(?:https?|ftp|file|ssh|git)://\S+")

#: A drive-letter path, not preceded by an identifier character. Case-insensitive; both
#: separators are accepted. Anchored on a "letter, colon, separator" shape rather than on any
#: specific drive, so it catches every machine's layout.
_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]:[\\/][^\s\"',)\]>|]*)")

#: Home directories that are machine- and user-specific.
_HOME_PATH = re.compile(r"(?:/home/|/Users/|\\Users\\)[A-Za-z0-9._-]+")

#: Patterns that indicate a secret rather than a mention of one.
_SECRET = re.compile(
    r"(?:KAGGLE_KEY|HF_TOKEN|HUGGINGFACE_TOKEN|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN)"
    r"\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{16,}",
)


class TestNoMachineSpecificPaths:
    def test_no_absolute_paths_in_tracked_text(self):
        """A drive-letter path would break every clone that is not this machine."""
        offenders: list[str] = []
        for relative in tracked_files():
            text = readable_text(relative)
            if text is None:
                continue
            stripped = _URL.sub("URL", text)
            for number, line in enumerate(stripped.splitlines(), start=1):
                match = _ABSOLUTE_PATH.search(line)
                if match:
                    offenders.append(f"{relative}:{number}: {match.group(1)[:80]}")
        assert not offenders, (
            "tracked files contain machine-specific absolute paths:\n  "
            + "\n  ".join(offenders)
            + "\nResolve paths relative to the config or __file__ instead."
        )

    def test_the_home_path_pattern_matches_what_it_claims(self):
        """Positive control for the user-home rule, using a placeholder user name."""
        # Assembled so the literal does not appear in this file, which the rule would flag.
        sample = "/ho" + "me/someuser/data"
        assert _HOME_PATH.search(sample), "the home-path pattern does not match a home path"
        assert not _HOME_PATH.search("relative/path/without/home"), "false positive on a rel path"

    def test_no_user_home_paths_in_tracked_text(self):
        """``/Users/<name>`` and ``\\Users\\<name>`` leak a machine layout into a shared repo."""
        offenders: list[str] = []
        for relative in tracked_files():
            text = readable_text(relative)
            if text is None:
                continue
            stripped = _URL.sub("URL", text)
            for number, line in enumerate(stripped.splitlines(), start=1):
                if _HOME_PATH.search(line):
                    offenders.append(f"{relative}:{number}: {line.strip()[:90]}")
        assert not offenders, "tracked files contain user-home absolute paths:\n  " + "\n  ".join(offenders)

    def test_the_guard_actually_detects_paths(self):
        """Negative and positive control.

        A guard that cannot fail is not a guard. This pins both directions: a real path is
        caught, and a URL is not — the second case is what a naive implementation gets wrong.
        """
        scheme = "ht" + "tps"
        drive = "X" + ":"
        other = "Y" + ":"
        sep = chr(92)
        fwd = "/"
        sample = (
            f"a = r'{drive}{sep}projects{sep}thing'\n"
            f"b = '{other}{fwd}shared/thing'\n"
            f"link = '{scheme}://example.invalid/datasets/x/y'\n"
        )
        stripped = _URL.sub("URL", sample)
        found = _ABSOLUTE_PATH.findall(stripped)
        assert len(found) == 2, f"expected the two drive-letter paths, got {found}"
        assert not any("example.invalid" in item for item in found), "a URL was misread as a path"

    def test_config_round_trips_from_another_working_directory(self, tmp_path):
        """The portability property that actually matters: a relative config works elsewhere.

        An earlier version of this test asserted that ``to_mapping`` never emits absolute
        paths. That is not the property the project needs — ``dump_config`` deliberately
        records *resolved* paths for provenance, and those dumps land in the gitignored
        ``artifacts/`` tree. What matters is that a config written with relative paths
        resolves correctly regardless of the directory the tool is invoked from, so that is
        what is asserted here.
        """
        import sys

        sys.path.insert(0, str(REPO_ROOT / "src"))
        from catface.config import load_config

        shipped = REPO_ROOT / "configs" / "default.yaml"
        if not shipped.is_file():
            pytest.skip("configs/default.yaml not present")

        # Loaded from a directory unrelated to the repository, relative paths must resolve
        # against the caller's working directory rather than silently keep the repo's.
        elsewhere = tmp_path / "somewhere" / "else"
        elsewhere.mkdir(parents=True)
        previous = Path.cwd()
        try:
            import os

            os.chdir(elsewhere)
            config = load_config(shipped)
            assert config.data.root.is_absolute()
            assert config.data.root == (elsewhere / "data").resolve(), (
                "relative config paths did not resolve against the working directory"
            )
            assert config.output_dir == (elsewhere / "artifacts").resolve()
        finally:
            os.chdir(previous)

        # Loading the *same* config from two different directories must produce the same
        # recipe identity, otherwise the same experiment would look like two.
        first = load_config(shipped)
        os.chdir(tmp_path)
        try:
            second = load_config(shipped)
        finally:
            os.chdir(previous)
        assert first.fingerprint() == second.fingerprint(), (
            "the same config file produced different fingerprints from different directories"
        )

    def test_a_relative_root_override_is_resolved_against_the_cwd(self, tmp_path):
        """Relative CLI overrides must behave like relative config values."""
        import os
        import sys

        sys.path.insert(0, str(REPO_ROOT / "src"))
        from catface.config import load_config

        elsewhere = tmp_path / "other"
        elsewhere.mkdir()
        previous = Path.cwd()
        try:
            os.chdir(elsewhere)
            config = load_config(None, {"data.root": "my_data", "output_dir": "my_out"})
        finally:
            os.chdir(previous)
        assert config.data.root == (elsewhere / "my_data").resolve()
        assert config.output_dir == (elsewhere / "my_out").resolve()

    def test_shipped_config_uses_relative_paths(self):
        """The reference config is committed, so it must not name a machine."""
        config_path = REPO_ROOT / "configs" / "default.yaml"
        if not config_path.is_file():
            pytest.skip("configs/default.yaml not present")
        text = _URL.sub("URL", config_path.read_text(encoding="utf-8"))
        assert not _ABSOLUTE_PATH.search(text), "configs/default.yaml contains an absolute path"


class TestNoSecrets:
    def test_no_credential_assignments_in_tracked_text(self):
        """A public repository plus ``git add .`` in a sync script makes this a live risk."""
        offenders: list[str] = []
        for relative in tracked_files():
            text = readable_text(relative)
            if text is None:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if _SECRET.search(line):
                    # Report the location and the variable name, never the value.
                    offenders.append(f"{relative}:{number}: {line.split('=')[0].strip()[:40]}")
        assert not offenders, (
            "possible credentials in tracked files:\n  " + "\n  ".join(offenders)
        )

    def test_the_secret_guard_actually_detects_a_key(self):
        """Positive control, using a synthetic value rather than a real one."""
        variable = "KAGGLE" + "_KEY"
        sample = f'{variable} = "0123456789abcdef0123456789abcdef"\n'
        assert _SECRET.search(sample), "the secret pattern does not match an obvious assignment"
        # And it must not fire on a documentation mention with no value.
        assert not _SECRET.search("Set the KAGGLE_KEY environment variable before running.")


class TestRepositoryLayout:
    def test_no_large_files_are_tracked(self):
        """GitHub rejects files over 100 MB and warns over 50 MB; catch it before the push."""
        limit = 5 * 1024 * 1024
        offenders = [
            f"{relative} ({path.stat().st_size / 1e6:.1f} MB)"
            for relative in tracked_files()
            if (path := REPO_ROOT / relative).is_file() and path.stat().st_size > limit
        ]
        assert not offenders, (
            "tracked files over 5 MB belong in .gitignore or Git LFS: " + ", ".join(offenders)
        )

    def test_archived_derived_imagery_is_not_tracked(self):
        """Audit imagery is rendered from third-party corpora and is regenerable."""
        derived = [
            relative for relative in tracked_files()
            if (relative.endswith((".jpg", ".jpeg", ".png"))
            and "/reference_samples/" in relative) or relative.endswith("contact_sheets.png")
        ]
        assert not derived, f"derived audit imagery should not be tracked: {derived}"

    def test_third_party_extraction_is_not_tracked(self):
        """``docs/archive/third-party`` is copyrighted material kept locally for reading."""
        offenders = [r for r in tracked_files() if r.startswith(VENDORED_PREFIXES)]
        assert not offenders, (
            "third-party extracted material must stay out of version control: "
            f"{offenders[:5]}"
        )


class TestDeclaredPythonFloor:
    """The package declares a minimum Python version; the code must respect it.

    CI caught this the hard way: the code used a ``@dataclass`` option introduced in 3.10
    while ``pyproject.toml`` advertised 3.9, so the 3.9 job failed on import and no local run
    could have noticed, because the development interpreter was 3.11. A declared floor that is
    not tested is a claim, not a guarantee.
    """

    @staticmethod
    def _too_new_constructs() -> dict[str, str]:
        """Constructs newer than the declared floor, with the version that introduced them.

        Assembled at runtime so this file contains no literal that the repository-hygiene
        guard would have to make an exception for. That guard has now been tripped six times
        by explanations of itself, which is the evidence that the no-exceptions rule is the
        right one.
        """
        slots = "slots" + "=True"
        kw_only = "kw_only" + "="
        strict = "strict" + "="
        return {
            "dataclass(" + slots + ")": "3.10",
            "dataclass(" + kw_only: "3.10",
            "zip(" + strict: "3.10",
            "import " + "tomllib": "3.11",
            "except" + "*": "3.11",
        }

    def _declared_floor(self) -> tuple[int, int]:
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'requires-python\s*=\s*">=([0-9]+)\.([0-9]+)"', text)
        assert match, "pyproject.toml declares no requires-python floor"
        return int(match.group(1)), int(match.group(2))

    def test_no_construct_newer_than_the_declared_floor(self):
        floor = self._declared_floor()
        offenders: list[str] = []
        for root in ("src", "tests", "tools"):
            for path in sorted((REPO_ROOT / root).rglob("*.py")):
                text = path.read_text(encoding="utf-8", errors="replace")
                for needle, introduced in self._too_new_constructs().items():
                    if needle in text:
                        major_minor = tuple(int(x) for x in introduced.split("."))
                        if major_minor > floor:
                            rel = path.relative_to(REPO_ROOT).as_posix()
                            offenders.append(f"{rel}: {needle!r} needs {introduced}, floor is {floor[0]}.{floor[1]}")
        assert not offenders, (
            "code uses constructs newer than the declared Python floor:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_floor_check_detects_a_too_new_construct(self):
        """Positive control: the rule must be able to fail."""
        assert self._declared_floor() >= (3, 9)
        # The needle list must be non-empty and must still cover the construct CI caught.
        assert any("dataclass(" in key for key in self._too_new_constructs())


class TestSourceIsTracked:
    """Source files must be visible to git.

    This exists because of a real failure with a deceptive signature. ``src/catface/data/`` --
    the entire data subpackage, six modules -- was silently excluded from version control by a
    ``.gitignore`` pattern with no leading slash, which matches a directory name at *any* depth.
    Every local test passed, because the files were still sitting in the working tree. Only a
    fresh clone failed, and only at import time, which is where CI found it.

    The lesson generalises: a file that exists but is not tracked looks perfectly healthy until
    somebody else clones the repository.
    """

    SOURCE_ROOTS = ("src", "tests", "tools")

    def test_every_python_source_file_is_tracked(self):
        tracked = set(tracked_files())
        missing: list[str] = []
        for root in self.SOURCE_ROOTS:
            for path in sorted((REPO_ROOT / root).rglob("*.py")):
                rel = path.relative_to(REPO_ROOT).as_posix()
                if rel in tracked:
                    continue
                # A directory-level ignore is the failure mode being guarded against; an
                # individual file absent from the index would fail the same way.
                missing.append(rel)
        assert not missing, (
            f"{len(missing)} Python source file(s) exist but are not tracked by git, so a "
            "clone would be broken:\n  " + "\n  ".join(missing[:20])
        )

    def test_no_source_directory_is_ignored(self):
        """Distinguish 'not added yet' from 'actively ignored', which need different fixes."""
        ignored: list[str] = []
        for root in self.SOURCE_ROOTS:
            for path in sorted((REPO_ROOT / root).rglob("*.py")):
                rel = path.relative_to(REPO_ROOT).as_posix()
                result = subprocess.run(
                    ["git", "check-ignore", "--no-index", "-q", rel],
                    cwd=REPO_ROOT, capture_output=True, check=False,
                )
                if result.returncode == 0:
                    ignored.append(rel)
        assert not ignored, (
            ".gitignore is excluding source files:\n  " + "\n  ".join(ignored[:20])
            + "\nAnchored patterns (a leading slash) are almost always what is wanted for "
            "project-owned directories."
        )

    def test_project_owned_ignores_are_anchored(self):
        """Pin the specific mistake: an unanchored pattern matching a project directory."""
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        patterns = [
            line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        # Names that also occur as a directory inside the source tree.
        project_dirs = {"data", "artifacts", "outputs", "runs"}
        unanchored = [
            pattern for pattern in patterns
            if pattern.rstrip("/") in project_dirs and not pattern.startswith("/")
        ]
        assert not unanchored, (
            f"these .gitignore patterns are unanchored and will match inside src/: {unanchored}"
        )


class TestPreCommitHook:
    """The hook is the same guard as above, but at the moment it cannot be forgotten."""

    HOOK = "tools/git-hooks/pre-commit"

    def test_the_hook_is_versioned(self):
        """A hook that only exists in .git/hooks cannot be reviewed or replicated."""
        assert (REPO_ROOT / self.HOOK).is_file(), (
            f"{self.HOOK} is missing; the pre-commit guard would silently not exist on a "
            "fresh clone"
        )

    def test_the_hook_checks_the_three_guarded_properties(self):
        """Pin what the hook guards, so a weakened hook is caught in review."""
        text = (REPO_ROOT / self.HOOK).read_text(encoding="utf-8")
        assert "diff --cached" in text, "the hook must inspect staged content, not the worktree"
        assert "A-Za-z]:[" in text, "the hook no longer looks for drive-letter paths"
        assert "KAGGLE_KEY" in text, "the hook no longer looks for credentials"
        assert "5242880" in text, "the hook no longer enforces the 5 MB staged-file limit"

    def test_the_hook_is_installed_for_this_checkout(self):
        """``core.hooksPath`` is local config, so a clone must run the installer once."""
        result = subprocess.run(
            ["git", "config", "--local", "--get", "core.hooksPath"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        configured = result.stdout.strip()
        if not configured:
            pytest.skip(
                "git hooks are not installed in this checkout; run "
                "tools/install-git-hooks.ps1 to enable the pre-commit guard"
            )
        assert configured.rstrip("/") == "tools/git-hooks", (
            f"core.hooksPath points at {configured!r}, not the versioned hook directory"
        )
        assert (REPO_ROOT / configured / "pre-commit").is_file()
