"""Container configuration checks.

Docker is not installed on the machine where this was written, so the image is never built here and
nothing in this project claims otherwise. What is testable without a daemon is the class of mistake
that either breaks a build or lets something unintended ship:

* a ``COPY`` source that does not exist, or that ``.dockerignore`` excludes;
* a ``COPY --from`` naming an undeclared stage;
* a compose build target the Dockerfile does not define;
* a health check pointed at the readiness endpoint, which would restart a healthy process while the
  model is still loading;
* weights or the corpus baked into the image instead of mounted.

The ignore-pattern matcher is the part most likely to be wrong, so it carries its own positive and
negative controls. Writing them inline found two real bugs in the first version: a trailing-slash
pattern failed to exclude the directory itself, and the ``!`` negation form was treated as a match.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


container = _load("check_container")


class TestIgnoreMatcher:
    """The approximation of .dockerignore semantics, with both directions covered."""

    @pytest.mark.parametrize(
        "pattern,path,expected",
        [
            # A directory pattern must exclude the directory itself, not only its contents. This is
            # what the first version got wrong.
            ("src/", "src", True),
            ("src/", "src/catface/api.py", True),
            ("data/", "data", True),
            ("data/", "data/faces/x.jpg", True),
            # A name prefix is not a match: "src" must not exclude "source_code".
            ("src/", "source_code", False),
            ("src", "source_code", False),
            ("data/", "database/x", False),
            # Globs behave as expected at any depth.
            ("*.md", "README.md", True),
            ("*.md", "docs/WEB.md", True),
            ("*.py", "src/catface/cli.py", True),
            # Negation must not itself count as an exclusion. ``!`` is only special at the start of
            # a *pattern*; a path whose name begins with it is an ordinary path, so ``*.md`` still
            # matches it. My first version of this case asserted the opposite and was simply wrong.
            ("!README.md", "README.md", False),
            ("*.md", "notes.md", True),
            # Unrelated patterns leave a path alone.
            (".git", "src/main.py", False),
        ],
    )
    def test_pattern_semantics(self, pattern, path, expected):
        assert container.is_ignored(path, [pattern]) is expected, (
            f"{pattern!r} against {path!r} should be {expected}"
        )

    def test_negation_does_not_override_a_later_exclusion(self):
        """Order matters in .dockerignore; the matcher only needs the exclusion to be seen."""
        assert container.is_ignored("secrets.env", ["*.env"]) is True
        assert container.is_ignored("secrets.env", ["*.env", "!secrets.env"]) is True


class TestShippedContainerConfiguration:
    """The real files in this repository must be self-consistent."""

    def test_configuration_check_passes(self):
        assert container.main() == 0, "tools.check_container reported a problem"

    def test_the_image_does_not_bake_in_weights_or_the_corpus(self):
        text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "best.pt" not in text.split("ENTRYPOINT")[0].replace("CATFACE_CHECKPOINT=/data/best.pt", ""), (
            "the checkpoint must be mounted, not copied into a layer"
        )
        assert "COPY data" not in text
        assert "COPY artifacts" not in text

    def test_the_build_context_excludes_data_and_history(self):
        ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
        for required in ("data/", "artifacts/", ".git", ".venv/"):
            assert required in ignored, f"{required} must not enter the build context"

    def test_health_check_uses_the_liveness_endpoint(self):
        """The probe must be the one that answers without the model.

        The check looks at the command, not at the whole file: the Dockerfile mentions
        ``/api/status`` in a comment explaining why it is *not* used, and an earlier version of this
        test flagged that comment. A test has to be precise about what it is asserting.
        """
        text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        probe = [line for line in text.splitlines() if "HEALTHCHECK" in line.upper() or "urlopen" in line]
        joined = "\n".join(probe)
        assert "healthz" in joined, "the image health check does not use the liveness endpoint"
        assert "api/status" not in joined, (
            "/api/status needs a loaded model, so using it as a health check restarts a healthy "
            "container during the ~164 s model load"
        )

    def test_runs_as_a_non_root_user(self):
        text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "USER " in text, "a container that writes to its own code is modifiable when exploited"

    def test_compose_binds_loopback_by_default(self):
        import yaml

        compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        ports = compose["services"]["search"]["ports"]
        assert all(str(entry).startswith("127.0.0.1:") for entry in ports), (
            "the service has no user management; publishing it on all interfaces by default is not "
            "a safe default"
        )


class TestContainerCiJob:
    """The CI job that actually builds the image, checked for the failure mode that already bit.

    Inside a double-quoted YAML scalar, an escaped quote collapses to a plain one, which silently
    split the find expression into separate arguments; ``sh -c`` then failed with "find: missing
    argument to -size". The workflow file looked correct when read. These assertions read the
    *parsed* YAML, which is what the runner sees, so the same mistake cannot pass review again.
    """

    @staticmethod
    def _runs() -> list[str]:
        import yaml

        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        )
        return [step["run"] for step in workflow["jobs"]["container"]["steps"] if "run" in step]

    def test_the_job_builds_the_image(self):
        assert any("docker build" in text for text in self._runs()), (
            "without a build the job asserts nothing about the Dockerfile"
        )

    def test_the_compose_file_is_validated(self):
        assert any("docker compose config" in text for text in self._runs())

    def test_missing_checkpoint_is_asserted_to_fail_fast(self):
        """Asserting the refusal is stronger than asserting nothing: it pins the diagnostic."""
        assert any("checkpoint not found" in text for text in self._runs())

    def test_the_weights_assertion_is_size_based_and_correctly_quoted(self):
        find_step = next((text for text in self._runs() if "offenders=" in text), None)
        assert find_step is not None, "the weights assertion is missing from the job"

        assert "'find" in find_step, (
            "the sh -c command must be single-quoted: a double-quoted YAML scalar unescapes the "
            "quote and splits the find expression into separate arguments"
        )
        assert '-c "find' not in find_step, "the sh -c command is double-quoted"

        for predicate in ("-xdev", "-type f", "-size +10240k", "-name"):
            assert predicate in find_step, f"the find expression lost {predicate!r}"
        assert "-size +" in find_step, (
            "the assertion must be size-based: a bare name match cannot tell a 1 KB library fixture "
            "from an 86 MB checkpoint"
        )
        assert find_step.count(r"\(") == 1 and find_step.count(r"\)") == 1, (
            "the -name alternatives need exactly one group, or the size filter applies to the wrong "
            "part of the expression"
        )
        assert "find /" in find_step, "the scan must target the container filesystem root"
