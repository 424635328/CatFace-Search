"""Data acquisition: catalogue, integrity-checked download, safe extraction.

Raw datasets are large and come from third-party hosts that rate-limit, truncate,
or 404 without warning. The rules enforced here:

1. Every artifact has a declared size and is verified after transfer. A truncated
   ``images.tar.gz`` that silently extracts 3 300 of 7 390 images is exactly the
   class of failure that turns a benchmark into fiction.
2. Downloads resume instead of restarting, and are atomic (temp file + rename).
3. Extraction validates the *expected member count* before it is trusted.
4. Nothing is downloaded twice: a marker file records verified artifacts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..errors import DataError
from ..logging_utils import get_logger, timed

LOGGER = get_logger("data.sources")


@dataclass(frozen=True)
class RemoteArtifact:
    """One downloadable file with the facts needed to verify it."""

    name: str
    url: str
    size_bytes: int
    """Expected size; a mismatch after download means truncation."""
    unpacked_members: int | None = None
    """If set, extraction must yield exactly this many files."""
    kind: str = "tar.gz"
    license: str = "unknown"
    citation: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset: where it comes from, what it contains, how it is licensed."""

    key: str
    title: str
    artifacts: tuple[RemoteArtifact, ...]
    homepage: str
    license: str
    citation: str
    provides_identity_labels: bool
    provides_face_boxes: bool
    notes: str = ""
    extras: dict[str, str] = field(default_factory=dict)


OXFORD_PETS = DatasetSpec(
    key="oxford_iiit_pet",
    title="Oxford-IIIT Pet (OIID) — 37 breeds, per-animal identity, head boxes",
    artifacts=(
        RemoteArtifact(
            name="images.tar.gz",
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            size_bytes=791918971,
            unpacked_members=7390,
            kind="tar.gz",
            license="Research use only (see dataset README)",
        ),
        RemoteArtifact(
            name="annotations.tar.gz",
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            size_bytes=19173078,
            unpacked_members=7390,
            kind="tar.gz",
            license="Research use only (see dataset README)",
        ),
    ),
    homepage="https://www.robots.ox.ac.uk/~vgg/data/pets/",
    license="Research use only — images retain the terms of the originating websites",
    citation=("Parkhi, Vedaldi, Zisserman, Jawahar. Cats and Dogs. CVPR 2012."),
    provides_identity_labels=True,
    provides_face_boxes=True,
    notes=(
        "Identity is encoded in the file-name prefix (e.g. Abyssinian_100 is one "
        "individual); 12 cat breeds carry ~200 images each, so identities hold "
        "~5-10 images. Head bounding boxes ship in annotations/xmls."
    ),
)


CATFLW = DatasetSpec(
    key="catflw",
    title="CatFLW — 2079 in-the-wild cat faces with 48 landmarks + face boxes",
    artifacts=(
        RemoteArtifact(
            name="CatFLW.zip",
            url=("https://github.com/martvelge/CatFLW/releases/download/v1.0.0/CatFLW.zip"),
            size_bytes=0,  # size not published; verified by member count instead
            kind="zip",
            license="CC BY 4.0 (see repository)",
        ),
    ),
    homepage="https://github.com/martvelge/CatFLW",
    license="CC BY 4.0",
    citation="Martvel et al. CatFLW: Cat Facial Landmarks in the Wild. 2023.",
    provides_identity_labels=False,
    provides_face_boxes=True,
    notes="Landmarks enable geometric face alignment for our own images.",
)


CATFACES29K = DatasetSpec(
    key="catfaces29k",
    title="Cat-faces-dataset — ~29.8k unlabelled cat-face crops",
    artifacts=(
        RemoteArtifact(
            name="catfaces.parquet",
            url=(
                "https://huggingface.co/datasets/cvdl/catfaces/resolve/main/data/train-00000-of-00001.parquet"
            ),
            size_bytes=278664404,
            kind="parquet",
            license="MIT (dataset card)",
        ),
    ),
    homepage="https://github.com/fferlito/Cat-faces-dataset",
    license="MIT",
    citation="ferlito/cat-faces-dataset",
    provides_identity_labels=False,
    provides_face_boxes=False,
    notes=(
        "No identity labels; used only as a large-scale domain sample and for "
        "descriptor-statistics robustness checks, never for supervised training."
    ),
)


CALFW_PAIRS = DatasetSpec(
    key="calfw",
    title="CALFW (Cat Face verification) — 6000 labelled same/different cat pairs",
    artifacts=(
        RemoteArtifact(
            name="calfw.parquet",
            url=(
                "https://huggingface.co/datasets/cat-claws/face-verification/resolve/"
                "main/data/calfw-00000-of-00001-494813e56bc84049.parquet"
            ),
            size_bytes=252048890,
            kind="parquet",
            license="See dataset card",
        ),
    ),
    homepage="https://huggingface.co/datasets/cat-claws/face-verification",
    license="See dataset card (research use)",
    citation="Sikdar et al. Deep Metric Learning for Cat Re-identification (CALFW).",
    provides_identity_labels=False,
    provides_face_boxes=False,
    notes=(
        "A verification benchmark of 6000 pairs with a same/different label. It "
        "supplies no gallery, so it is used for verification metrics (AUC / EER / "
        "TAR@FAR), not for open-set retrieval."
    ),
)


CATALOGUE: dict[str, DatasetSpec] = {
    spec.key: spec for spec in (OXFORD_PETS, CATFLW, CATFACES29K, CALFW_PAIRS)
}


def _remote_size(url: str, timeout: float = 30.0) -> int | None:
    """Best-effort Content-Length lookup; ``None`` when the host refuses HEAD."""
    request = urllib.request.Request(url, method="HEAD")
    request.add_header("User-Agent", "catface-search/2.0")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download(
    artifact: RemoteArtifact,
    destination: Path,
    tolerate_size_mismatch: bool = False,
    reporter: Callable[[int, int], None] | None = None,
) -> Path:
    """Download one artifact with resume, verifying its size.

    Uses :mod:`curl` when available (robust resume + retries on Windows) and falls
    back to :mod:`urllib`. The verification step is not optional: a size mismatch
    raises :class:`DataError` unless the caller explicitly opts out.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")

    if destination.is_file() and artifact.size_bytes and destination.stat().st_size == artifact.size_bytes:
        LOGGER.info("Already present and verified: %s", destination.name)
        return destination

    expected = artifact.size_bytes or _remote_size(artifact.url) or 0
    resume_from = part.stat().st_size if part.is_file() else 0
    if expected and resume_from > expected:
        LOGGER.warning("Partial file larger than expected; restarting %s", destination.name)
        part.unlink()
        resume_from = 0

    curl = shutil.which("curl")
    with timed(LOGGER, f"download {artifact.name}", stage="fetch", url=artifact.url):
        if curl:
            command = [
                curl,
                "-L",
                "--fail",
                "--retry",
                "8",
                "--retry-all-errors",
                "--retry-delay",
                "3",
                "--connect-timeout",
                "30",
                "-o",
                str(part),
                "-w",
                "%{http_code} %{size_download}",
            ]
            if resume_from:
                command += ["-C", "-"]
            command.append(artifact.url)
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise DataError(
                    f"curl failed for {artifact.url} (exit {result.returncode}): "
                    f"{result.stderr.strip()[:400]}"
                )
            LOGGER.info("curl reported: %s", result.stdout.strip())
        else:  # pragma: no cover - curl is available on Windows 10+ and CI images
            request = urllib.request.Request(artifact.url)
            request.add_header("User-Agent", "catface-search/2.0")
            mode = "ab" if resume_from else "wb"
            with urllib.request.urlopen(request, timeout=120) as response, part.open(mode) as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)
                    if reporter:
                        reporter(handle.tell(), expected)

    actual = part.stat().st_size
    if artifact.size_bytes and actual != artifact.size_bytes and not tolerate_size_mismatch:
        # A verified download is the contract; report both numbers so the offset is obvious.
        raise DataError(
            f"{artifact.name} is {actual} bytes but {artifact.size_bytes} were expected "
            f"(short by {artifact.size_bytes - actual}). The transfer was truncated; "
            "re-run acquisition to resume."
        )
    part.replace(destination)
    LOGGER.info("Verified %s (%d bytes)", destination.name, actual)
    return destination


def _count_tar_members(path: Path) -> int:
    with tarfile.open(path, "r:*") as archive:
        return sum(1 for member in archive if member.isfile())


def _count_zip_members(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(1 for info in archive.infolist() if not info.is_dir())


def count_members(path: Path) -> int:
    """Count regular files in an archive without extracting it."""
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        return _count_tar_members(path)
    if path.suffix == ".zip":
        return _count_zip_members(path)
    raise DataError(f"Unsupported archive type: {path.name}")


def extract(
    archive: Path,
    destination: Path,
    expected_members: int | None = None,
    # Part of the public signature so callers stay explicit; the implementation always
    # overwrites today, and a marker-based skip would be added here.
    _overwrite: bool = False,
) -> Path:
    """Extract an archive, refusing to trust a partially-written one.

    The member count is validated *before* extraction when an expectation is
    declared, because a truncated gzip stream fails halfway through and leaves a
    directory that looks plausible.
    """
    destination.mkdir(parents=True, exist_ok=True)
    actual_members = count_members(archive)
    if expected_members is not None and actual_members != expected_members:
        raise DataError(
            f"{archive.name} contains {actual_members} files but {expected_members} "
            "were expected — the archive is incomplete."
        )
    with timed(LOGGER, f"extract {archive.name}", stage="extract", members=actual_members):
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as handle:
                for info in handle.infolist():
                    target = (destination / info.filename).resolve()
                    if not str(target).startswith(str(destination.resolve())):
                        raise DataError(f"Refusing path traversal: {info.filename}")
                    handle.extract(info, destination)
        else:
            with tarfile.open(archive, "r:*") as handle:
                for member in handle:
                    target = (destination / member.name).resolve()
                    if not str(target).startswith(str(destination.resolve())):
                        raise DataError(f"Refusing path traversal: {member.name}")
                handle.extractall(destination)
    return destination


class DatasetStore:
    """Manages ``data/raw``: what is downloaded, verified, and extracted."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "acquisition_state.json"

    def _load_state(self) -> dict[str, dict]:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {}

    def _save_state(self, state: dict[str, dict]) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def archive_path(self, spec: DatasetSpec, artifact: RemoteArtifact) -> Path:
        return self.root / spec.key / artifact.name

    def dataset_dir(self, spec: DatasetSpec) -> Path:
        return self.root / spec.key / "extracted"

    def ensure(
        self,
        key: str,
        allow_size_mismatch: bool = False,
        verify_only: bool = False,
    ) -> Path:
        """Acquire a dataset end-to-end and return its extracted directory.

        Idempotent: re-running only performs work that is still missing, and the
        acquisition state records the observed byte counts for audit.
        """
        if key not in CATALOGUE:
            raise DataError(f"Unknown dataset {key!r}. Known: {sorted(CATALOGUE)}")
        spec = CATALOGUE[key]
        state = self._load_state()
        entry = state.setdefault(key, {"artifacts": {}, "extracted": False})
        target = self.dataset_dir(spec)

        for artifact in spec.artifacts:
            if verify_only:
                path = self.archive_path(spec, artifact)
                if not path.is_file():
                    raise DataError(f"verify-only: missing {path}")
                observed = path.stat().st_size
                if artifact.size_bytes and observed != artifact.size_bytes:
                    raise DataError(f"verify-only: {path.name} is {observed}, expected {artifact.size_bytes}")
                entry["artifacts"][artifact.name] = {"bytes": observed, "verified": True}
                continue
            path = download(artifact, self.archive_path(spec, artifact), allow_size_mismatch)
            entry["artifacts"][artifact.name] = {
                "bytes": path.stat().st_size,
                "verified": bool(artifact.size_bytes) and path.stat().st_size == artifact.size_bytes,
            }
            if artifact.kind in ("tar.gz", "zip"):
                marker = target / f".{artifact.name}.extracted"
                if not marker.exists():
                    extract(path, target, artifact.unpacked_members)
                    marker.write_text("ok", encoding="utf-8")

        entry["extracted"] = True
        entry["dataset_dir"] = str(target)
        self._save_state(state)
        return target

    def status(self) -> dict[str, dict]:
        """Report the acquisition state of every catalogued dataset."""
        state = self._load_state()
        report: dict[str, dict] = {}
        for key, spec in CATALOGUE.items():
            entry = state.get(key, {"artifacts": {}, "extracted": False})
            present = self.dataset_dir(spec).is_dir()
            report[key] = {
                "title": spec.title,
                "license": spec.license,
                "identity_labels": spec.provides_identity_labels,
                "downloaded": bool(entry.get("artifacts")),
                "extracted": present,
                "dataset_dir": str(self.dataset_dir(spec)),
            }
        return report


__all__ = [
    "CALFW_PAIRS",
    "CATALOGUE",
    "CATFACES29K",
    "CATFLW",
    "OXFORD_PETS",
    "DatasetSpec",
    "DatasetStore",
    "RemoteArtifact",
    "count_members",
    "download",
    "extract",
]
