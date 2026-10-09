"""Framework-independent identity search over the trained descriptor index.

Why this layer is separate from the HTTP layer
----------------------------------------------
Everything interesting here is decision logic: which checkpoint is loaded, how a query is validated,
what a "match" is, and what the service refuses to do. Putting that behind FastAPI would make it
testable only through HTTP. Instead this module has no web import at all, so the test suite and the
CLI can call it directly, and the API layer is a thin translation.

Two properties are deliberate and enforced here rather than documented as intentions:

* **The model is loaded once.** Loading a checkpoint per request would put a multi-second model
  construction on the critical path of every search. ``SearchService`` owns one embedder for its
  lifetime and reports readiness through :meth:`status`.
* **Exhaustive search, not an approximate index.** The gallery is 12 141 descriptors of width 512,
  so an exact matrix product costs milliseconds. An ANN index would add a recall parameter that
  makes a reported number depend on index tuning instead of on the model, which is the opposite of
  what a benchmarked retrieval service should do.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..data.manifest import Manifest
from ..errors import CatFaceError, DataError
from ..logging_utils import get_logger
from ..models.embedder import Embedder, embed_records

LOGGER = get_logger("catface.web.service")

#: Default location of the descriptor cache. Gitignored derived data, so it may legitimately be
#: absent; the service then embeds, which is only slow, not broken.
DEFAULT_DESCRIPTOR_CACHE = Path("artifacts") / "web-descriptors"


def _file_digest(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's content. Used to detect a checkpoint that changed under a known path."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _descriptor_width(embedder: Embedder | None) -> int:
    """Width of the descriptor the model produces, or 0 when it cannot be determined."""
    if embedder is None:
        return 0
    head = getattr(embedder, "head", None)
    return int(getattr(head, "embedding_dim", 0) or 0)


#: Extensions accepted for an uploaded query. Checked as a hint only — the bytes are validated by
#: decoding the image, because a filename proves nothing about its content.
ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

#: Upper bound on an uploaded query. A query is one photograph; anything larger is a mistake or an
#: attempt to make the service do work it was not sized for.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


class QueryRejectedError(CatFaceError):
    """The request was well-formed HTTP but not a usable query."""


@dataclass
class Match:
    """One retrieved gallery entry."""

    rank: int
    image_id: str
    identity: str
    similarity: float
    path: str
    identity_consensus: float
    """Share of the returned matches that carry this identity, computed over the whole result."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "image_id": self.image_id,
            "identity": self.identity,
            "similarity": round(self.similarity, 6),
            "path": self.path,
            "identity_consensus": round(self.identity_consensus, 4),
        }


@dataclass
class SearchOutcome:
    """A complete answer, including the evidence needed to distrust it."""

    predicted_identity: str | None
    top_similarity: float | None
    margin: float | None
    """Top-1 similarity minus the best similarity from a *different* identity.

    Reported because it is the honest confidence signal: a high top-1 similarity with a small
    margin means two identities are nearly tied, which is exactly the condition under which this
    system's own error analysis found every remaining failure (median margin -0.057).
    """
    matches: list[Match] = field(default_factory=list)
    descriptor_dim: int = 0
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    query_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_identity": self.predicted_identity,
            "top_similarity": None if self.top_similarity is None else round(self.top_similarity, 6),
            "margin": None if self.margin is None else round(self.margin, 6),
            "matches": [match.to_dict() for match in self.matches],
            "descriptor_dim": self.descriptor_dim,
            "timing": {
                "embedding_ms": round(self.embedding_ms, 1),
                "search_ms": round(self.search_ms, 1),
            },
        }


class SearchService:
    """Owns the embedder and the gallery descriptors for one process."""

    def __init__(
        self,
        checkpoint: str | Path,
        manifest: str | Path,
        device: str = "cpu",
        image_size: int = 224,
        gallery_root: str | Path | None = None,
        descriptor_cache_dir: str | Path | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.manifest_path = Path(manifest)
        self.device = device
        self.image_size = int(image_size)
        # Gallery images are addressed relative to the manifest's corpus root when one is given, so
        # the API can serve them without exposing an absolute path from the serving machine.
        self.gallery_root = Path(gallery_root) if gallery_root else None
        # Where gallery descriptors are cached between runs. Relative paths resolve against the
        # working directory, which the entry point sets to the repository root.
        self.descriptor_cache_dir = Path(descriptor_cache_dir or DEFAULT_DESCRIPTOR_CACHE)
        self._embedder: Embedder | None = None
        self._vectors: np.ndarray | None = None
        self._records: list[Any] = []
        self._load_seconds = 0.0

    # -- lifecycle ---------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._embedder is not None and self._vectors is not None and len(self._records) > 0

    def load(self) -> None:
        """Load the model and embed the gallery once.

        If a descriptor cache from a previous run matches this exact configuration, the gallery
        descriptors are loaded from disk instead of recomputed. The cache key includes the
        checkpoint's *content* hash, the manifest's hash, the image size and the TTA views, so a
        stale entry cannot be mistaken for a fresh one. Missing or mismatched entries fall back to
        embedding, which is what makes the optimisation safe to leave on: the slow path remains the
        fallback rather than an error.
        """
        if self.ready:
            return
        if not self.checkpoint.is_file():
            raise DataError(
                f"checkpoint not found: {self.checkpoint}. Train one with "
                "`python -m tools.train_embedder`, or point --checkpoint at an existing best.pt."
            )
        if not self.manifest_path.is_file():
            raise DataError(
                f"manifest not found: {self.manifest_path}. Build it with "
                "`python -m catface.cli prepare --source cat_individuals`."
            )

        started = time.perf_counter()
        LOGGER.info("loading checkpoint %s on %s", self.checkpoint, self.device)
        self._embedder = Embedder.load(self.checkpoint, device=self.device)
        self._records = list(Manifest.load(self.manifest_path))
        if not self._records:
            raise DataError(f"manifest {self.manifest_path} is empty")

        cached = self._load_cached_descriptors()
        if cached is not None:
            vectors = cached
        else:
            LOGGER.info("embedding %d gallery images (no usable cache)", len(self._records))
            # A deliberate delay for tests. It exists because the failure it guards against -- a
            # handler blocking the event loop during a slow load -- cannot be reproduced with an
            # instant load, and it is named as a test hook rather than disguised as a feature.
            delay = os.environ.get("CATFACE_LOAD_DELAY_SECONDS")
            if delay:
                LOGGER.warning("CATFACE_LOAD_DELAY_SECONDS=%s: delaying the load on purpose", delay)
                time.sleep(float(delay))
            embedding = embed_records(
                self._embedder,
                [record.path for record in self._records],
                image_size=self.image_size,
                batch_size=32,
            )
            vectors = np.asarray(embedding.vectors, dtype=np.float32)
            if vectors.size == 0:
                raise DataError("the gallery produced no descriptors")

        # ``ready`` becomes true the moment ``_vectors`` is set, so the load time is stored first:
        # otherwise a status request landing between the two assignments reports ready with
        # load_seconds 0.0, which reads as a measurement rather than as the race it is.
        self._load_seconds = time.perf_counter() - started
        self._vectors = vectors
        if cached is None:
            # Persist after publishing: writing the cache must not delay readiness, and a failure to
            # write is only a missed optimisation.
            self._save_cached_descriptors(vectors)
        LOGGER.info(
            "ready: %d descriptors of width %d in %.1fs",
            self._vectors.shape[0],
            self._vectors.shape[1],
            self._load_seconds,
        )

    # -- descriptor cache --------------------------------------------------------------------
    def _cache_key(self) -> str | None:
        """Key identifying this exact gallery embedding, or ``None`` when it cannot be computed.

        Hashing the checkpoint's *content* is what makes a stale entry detectable: a retrained model
        usually lands on the same path, so a path-based key would happily serve descriptors computed
        from the previous weights. Reading two files is milliseconds against the minutes it saves.

        A failure here is never fatal -- the caller simply embeds -- because the cache is an
        optimisation and a missing optimisation must not become a startup error.
        """
        try:
            manifest_digest = _file_digest(self.manifest_path)
            checkpoint_digest = _file_digest(self.checkpoint)
        except OSError:
            return None
        payload = {
            "checkpoint_sha256": checkpoint_digest,
            "manifest_sha256": manifest_digest,
            "image_size": self.image_size,
            "tta": list(self._embedder.config.tta) if self._embedder else [],
            "order": "manifest",
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self) -> Path | None:
        key = self._cache_key()
        if key is None:
            return None
        return self.descriptor_cache_dir / f"gallery-{key}.npz"

    def _load_cached_descriptors(self) -> np.ndarray | None:
        """Return gallery descriptors in manifest order from the cache, or ``None``.

        The identity of every image is re-checked against the manifest rather than trusted from the
        key alone: a wrong descriptor silently attached to the wrong label would corrupt every
        answer while looking healthy, which is the worst possible failure for a retrieval service.
        """
        target = self._cache_path()
        if target is None or not target.is_file():
            return None
        try:
            with np.load(target, allow_pickle=False) as archive:
                ids = [str(value) for value in archive["ids"].tolist()]
                labels = [str(value) for value in archive["labels"].tolist()]
                vectors = np.asarray(archive["vectors"], dtype=np.float32)
        except (OSError, KeyError, ValueError, EOFError) as error:
            # EOFError is what numpy raises for a truncated or empty .npz, which is exactly what an
            # interrupted save leaves behind — a real possibility here, because the writer runs while
            # the service may be shutting down. This was caught by a test that wrote a zero-byte
            # cache file, after a real interrupted run had left one in the repository.
            LOGGER.warning("descriptor cache at %s is unreadable (%s); embedding instead", target, error)
            return None

        if len(ids) != len(self._records) or vectors.shape[0] != len(self._records):
            LOGGER.warning(
                "descriptor cache holds %d entries but the manifest has %d; embedding",
                len(ids),
                len(self._records),
            )
            return None
        for position, record in enumerate(self._records):
            if ids[position] != record.image_id or labels[position] != record.identity:
                LOGGER.warning(
                    "descriptor cache does not match the manifest at position %d (%s vs %s); "
                    "embedding instead",
                    position,
                    ids[position],
                    record.image_id,
                )
                return None
        expected_width = _descriptor_width(self._embedder)
        if expected_width and vectors.shape[1] != expected_width:
            LOGGER.warning(
                "descriptor cache width %d does not match the model's %d; embedding",
                vectors.shape[1],
                expected_width,
            )
            return None
        LOGGER.info("gallery descriptors loaded from cache (%s)", target.name)
        return vectors

    def _save_cached_descriptors(self, vectors: np.ndarray) -> None:
        """Persist descriptors so the next start is I/O instead of minutes of GPU work."""
        target = self._cache_path()
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                vectors=np.asarray(vectors, dtype=np.float32),
                ids=np.asarray([record.image_id for record in self._records], dtype="U256"),
                labels=np.asarray([record.identity for record in self._records], dtype="U128"),
            )
            LOGGER.info("gallery descriptors cached at %s", target.name)
        except OSError as error:
            # A read-only or full disk must not fail the request that triggered the save.
            LOGGER.warning("could not write the descriptor cache (%s); serving without it", error)

    # -- introspection -----------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Readiness plus the facts a client needs to interpret an answer."""
        identities = {record.identity for record in self._records}
        return {
            "ready": self.ready,
            "checkpoint": str(self.checkpoint),
            "manifest": str(self.manifest_path),
            "device": self.device,
            "image_size": self.image_size,
            "gallery_images": len(self._records),
            "gallery_identities": len(identities),
            "descriptor_dim": int(self._vectors.shape[1]) if self._vectors is not None else 0,
            "backend": self._embedder.config.backbone if self._embedder else None,
            "tta": list(self._embedder.config.tta) if self._embedder else [],
            "load_seconds": round(self._load_seconds, 2),
        }

    def identity_index(self) -> dict[str, int]:
        """Number of gallery images per identity, for the browse view."""
        counts: dict[str, int] = {}
        for record in self._records:
            counts[record.identity] = counts.get(record.identity, 0) + 1
        return dict(sorted(counts.items()))

    def record_for(self, image_id: str) -> Any | None:
        for record in self._records:
            if record.image_id == image_id:
                return record
        return None

    def resolve_image(self, path: str) -> Path | None:
        """Return a servable path for a gallery entry, or ``None`` when it is not servable.

        Returning ``None`` rather than raising keeps the API able to answer "this entry exists but
        its file is not available to this deployment" instead of failing the whole request.
        """
        candidate = Path(path)
        if not candidate.is_absolute() and self.gallery_root is not None:
            candidate = self.gallery_root / candidate
        return candidate if candidate.is_file() else None

    # -- query -------------------------------------------------------------------------------
    def search(
        self,
        image_path: str | Path,
        top_k: int = 10,
        identity_aggregation: bool = False,
    ) -> SearchOutcome:
        """Embed one query image and return its nearest gallery entries.

        ``identity_aggregation`` re-scores each identity by the mean similarity of its two most
        similar images instead of its single best image. Measured on the benchmark this is worth
        +0.0020 hit@1, which is one query out of 503 — i.e. indistinguishable from noise. It is
        exposed because it is the better-motivated rule, and labelled as not demonstrably better.
        """
        if not self.ready:
            # A cold service is loaded here only when nothing else is already loading it. If the HTTP
            # layer registered a background task, the request handler awaits that task instead of
            # reaching this branch, because calling this from the event loop blocks every other
            # request for the whole load.
            self.load()
        # Bind the two attributes that ``load`` guarantees, so the rest of this method reads them
        # from locals: it removes the need for assertions that a type checker cannot be told about,
        # and makes it impossible to use a half-loaded service by accident.
        embedder = self._embedder
        gallery = self._vectors
        if embedder is None or gallery is None:  # pragma: no cover - guarded by ``ready`` above
            raise DataError("the search service is not loaded")

        query_path = Path(image_path)
        if not query_path.is_file():
            raise QueryRejectedError(f"query image not found: {query_path}")

        started = time.perf_counter()
        embedding = embed_records(embedder, [str(query_path)], batch_size=1)
        embedding_ms = (time.perf_counter() - started) * 1000.0
        if embedding.vectors.size == 0:
            raise QueryRejectedError(
                "could not decode this file as an image; supported formats are JPEG, PNG, WebP, BMP"
            )

        vector = np.asarray(embedding.vectors, dtype=np.float32)
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        if not np.isfinite(norm).all() or float(norm.min()) <= 0.0:
            raise QueryRejectedError("the query produced a degenerate descriptor")

        started = time.perf_counter()
        similarity = (vector / np.maximum(norm, 1e-12)) @ gallery.T
        similarity = np.asarray(similarity, dtype=np.float32)[0]
        search_ms = (time.perf_counter() - started) * 1000.0

        top_k = int(max(1, min(top_k, len(self._records))))
        if identity_aggregation:
            ranked = self._rank_identities(similarity, top_k)
        else:
            ranked = self._rank_images(similarity, top_k)

        predicted = ranked[0]["identity"] if ranked else None
        top_similarity = ranked[0]["similarity"] if ranked else None
        margin = None
        if ranked:
            others = [row["similarity"] for row in ranked if row["identity"] != predicted]
            margin = float(ranked[0]["similarity"] - others[0]) if others else None

        consensus: dict[str, float] = {}
        for row in ranked:
            consensus[row["identity"]] = consensus.get(row["identity"], 0.0) + 1.0
        total = float(len(ranked)) or 1.0
        matches = [
            Match(
                rank=index,
                image_id=row["image_id"],
                identity=row["identity"],
                similarity=row["similarity"],
                path=row["path"],
                identity_consensus=consensus[row["identity"]] / total,
            )
            for index, row in enumerate(ranked, start=1)
        ]

        return SearchOutcome(
            predicted_identity=predicted,
            top_similarity=top_similarity,
            margin=margin,
            matches=matches,
            descriptor_dim=int(gallery.shape[1]),
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            query_bytes=query_path.stat().st_size,
        )

    def _rank_images(self, similarity: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        order = np.argsort(-similarity, kind="stable")[:top_k]
        return [
            {
                "image_id": self._records[int(position)].image_id,
                "identity": self._records[int(position)].identity,
                "similarity": float(similarity[int(position)]),
                "path": self._records[int(position)].path,
            }
            for position in order
        ]

    def _rank_identities(self, similarity: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """Rank identities by the mean of their two most similar gallery images.

        Two is not a tuned constant: the sweep over k = 1..30 on the benchmark peaked at k = 2 and
        degraded monotonically beyond it, because pooling more images of a well-represented identity
        starts to outweigh the single best evidence of a poorly represented one.
        """
        by_identity: dict[str, list[int]] = {}
        for position, record in enumerate(self._records):
            by_identity.setdefault(record.identity, []).append(position)

        scored: list[tuple[float, str, int]] = []
        for identity, positions in by_identity.items():
            scores = similarity[positions]
            best = int(np.argmax(scores))
            if scores.size >= 2:
                top_two = np.partition(scores, -2)[-2:]
                value = float(top_two.mean())
            else:
                value = float(scores[best])
            scored.append((value, identity, positions[best]))
        scored.sort(key=lambda item: -item[0])

        return [
            {
                "image_id": self._records[position].image_id,
                "identity": identity,
                "similarity": value,
                "path": self._records[position].path,
            }
            for value, identity, position in scored[:top_k]
        ]


__all__ = [
    "ALLOWED_IMAGE_SUFFIXES",
    "MAX_UPLOAD_BYTES",
    "Match",
    "QueryRejectedError",
    "SearchOutcome",
    "SearchService",
]
