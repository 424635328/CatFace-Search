"""Vector index and search.

FAISS is used when available and a NumPy brute-force path is always kept, for three
reasons: results must be reproducible on a machine without FAISS; the brute-force
path is the ground truth against which an approximate index is validated; and the
``IndexFlatIP`` path is exactly equivalent to cosine similarity on unit vectors, so
switching backends cannot silently change scores.

The critical correctness point for a *re-identification* index is that the query's
own entry must be retrievable but must not be counted as its own match. The index
therefore returns ids (not positions), and the consumers exclude self-matches by id.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ArtifactError
from ..eval.postprocess import l2_normalize
from ..logging_utils import get_logger, timed

LOGGER = get_logger("index.faiss")

INDEX_FORMAT_VERSION = 2


@dataclass
class SearchResult:
    """Top-k neighbours for a batch of queries."""

    scores: np.ndarray
    """``(Q, K)`` similarity scores, descending."""
    ids: list[list[str]]
    """``(Q, K)`` neighbour identifiers."""

    def top1(self) -> list[tuple[str, float]]:
        return [(row[0], float(score[0])) for row, score in zip(self.ids, self.scores)]


@dataclass
class VectorIndex:
    """An index over descriptors, keyed by caller-supplied string ids.

    Args:
        dim: Descriptor width.
        kind: ``flat_ip`` | ``ivf_pq`` | ``hnsw`` | ``numpy``.
        ids: Identifier per indexed vector, positionally aligned.
        nlist/nprobe/m_pq/nbits/hnsw_m/ef_search: Backend-specific parameters.
    """

    dim: int
    kind: str = "flat_ip"
    ids: list[str] = field(default_factory=list)
    nlist: int = 1024
    nprobe: int = 32
    m_pq: int = 32
    nbits: int = 8
    hnsw_m: int = 32
    ef_search: int = 64
    metric: str = "cosine"

    def __post_init__(self) -> None:
        self._vectors: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._faiss_index: Any | None = None
        self._faiss = None
        if self.metric not in ("cosine", "ip", "l2"):
            raise ArtifactError(f"Unsupported metric: {self.metric}")
        self.vectors = np.zeros((0, self.dim), dtype=np.float32)

    # -- properties ---------------------------------------------------------
    @property
    def size(self) -> int:
        return int(self._vectors.shape[0])

    @property
    def backend(self) -> str:
        if self.kind == "numpy":
            return "numpy"
        return "faiss" if self._load_faiss() is not None else "numpy"

    def __len__(self) -> int:
        return self.size

    # -- construction -------------------------------------------------------
    @staticmethod
    def _normalise(vectors: np.ndarray, metric: str) -> np.ndarray:
        """Normalise for cosine metrics; inner-product and L2 are taken as given."""
        matrix = np.asarray(vectors, dtype=np.float32)
        return l2_normalize(matrix) if metric == "cosine" else matrix

    def add(self, vectors: np.ndarray, ids: Sequence[str]) -> VectorIndex:
        """Append vectors with their ids; both must be positionally aligned."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ArtifactError(f"vectors must be 2-D, got {matrix.shape}")
        if matrix.shape[0] != len(ids):
            raise ArtifactError(
                f"{matrix.shape[0]} vectors but {len(ids)} ids — they must be aligned"
            )
        if matrix.shape[1] != self.dim:
            raise ArtifactError(
                f"vector width {matrix.shape[1]} does not match index dim {self.dim}"
            )
        self._vectors = np.vstack([self._vectors, self._normalise(matrix, self.metric)])
        self.ids.extend(str(i) for i in ids)
        return self

    def build(self) -> VectorIndex:
        """Build the ANN structure (no-op for the flat backend)."""
        if self.kind == "numpy":
            return self
        faiss = self._load_faiss()
        if faiss is None:
            LOGGER.warning("FAISS unavailable; index %s will use the NumPy backend", self.kind)
            return self
        if self.size == 0:
            raise ArtifactError("Cannot build an index with no vectors")

        with timed(LOGGER, f"build {self.kind} index", stage="index", vectors=self.size):
            if self.kind == "flat_ip":
                self._faiss_index = faiss.IndexFlatIP(self.dim)
                self._faiss_index.add(self._vectors)
            elif self.kind == "ivf_pq":
                # Product quantisation must divide the descriptor dimension; the
                # common case (768) is not divisible by 32, so pick a compatible m.
                m = self.m_pq
                while m > 1 and self.dim % m != 0:
                    m //= 2
                if m != self.m_pq:
                    LOGGER.warning(
                        "Adjusted PQ segments from %d to %d so they divide dim=%d",
                        self.m_pq, m, self.dim,
                    )
                quantiser = faiss.IndexFlatIP(self.dim)
                index = faiss.IndexIVFPQ(quantiser, self.dim, self.nlist, m, self.nbits,
                                         faiss.METRIC_INNER_PRODUCT)
                index.train(self._vectors)
                index.add(self._vectors)
                index.nprobe = self.nprobe
                self._faiss_index = index
            elif self.kind == "hnsw":
                index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = max(self.ef_search, 40)
                index.add(self._vectors)
                index.hnsw.efSearch = self.ef_search
                self._faiss_index = index
            else:  # pragma: no cover - guarded by config validation
                raise ArtifactError(f"Unknown index kind: {self.kind}")
        return self

    # -- search -------------------------------------------------------------
    def search(self, queries: np.ndarray, top_k: int = 10) -> SearchResult:
        """Return the ``top_k`` most similar indexed entries per query row."""
        if self.size == 0:
            raise ArtifactError("Index is empty; add vectors before searching")
        matrix = self._normalise(np.asarray(queries, dtype=np.float32), self.metric)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape[1] != self.dim:
            raise ArtifactError(
                f"query width {matrix.shape[1]} does not match index dim {self.dim}"
            )
        k = int(min(top_k, self.size))

        if self._faiss_index is not None:
            scores, positions = self._faiss_index.search(matrix, k)
        else:
            similarity = matrix @ self._vectors.T
            positions = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
            scores = np.take_along_axis(similarity, positions, axis=1)
            order = np.argsort(-scores, axis=1, kind="stable")
            positions = np.take_along_axis(positions, order, axis=1)
            scores = np.take_along_axis(scores, order, axis=1)

        ids = [[self.ids[int(p)] for p in row if 0 <= int(p) < len(self.ids)] for row in positions]
        return SearchResult(scores=scores.astype(np.float32), ids=ids)

    def score_all(self, queries: np.ndarray) -> np.ndarray:
        """Exhaustive similarity against every indexed vector (no top-k pruning).

        Used for benchmarking, where an approximate index would make the reported
        metric depend on the index parameters instead of on the descriptor.
        """
        matrix = self._normalise(np.asarray(queries, dtype=np.float32), self.metric)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if self.size == 0:
            raise ArtifactError("Index is empty")
        return (matrix @ self._vectors.T).astype(np.float32)

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | Path, embeddings: np.ndarray | None = None) -> Path:
        """Persist ids + vectors (+ the optional raw descriptors) to a directory."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": INDEX_FORMAT_VERSION,
            "dim": self.dim,
            "kind": self.kind,
            "metric": self.metric,
            "size": self.size,
            "backend": self.backend,
            "params": {
                "nlist": self.nlist, "nprobe": self.nprobe, "m_pq": self.m_pq,
                "nbits": self.nbits, "hnsw_m": self.hnsw_m, "ef_search": self.ef_search,
            },
        }
        (target / "index_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        (target / "ids.txt").write_text("\n".join(self.ids) + "\n", encoding="utf-8")
        with (target / "vectors.pkl").open("wb") as handle:
            pickle.dump({"vectors": self._vectors, "ids": self.ids}, handle, protocol=4)
        if self._faiss_index is not None:
            faiss = self._load_faiss()
            if faiss is not None:
                faiss.write_index(self._faiss_index, str(target / "faiss.index"))
        if embeddings is not None:
            np.save(target / "raw_embeddings.npy", np.asarray(embeddings, dtype=np.float32))
        LOGGER.info("Saved index (%d vectors, %s) to %s", self.size, self.backend, target)
        return target

    @classmethod
    def load(cls, directory: str | Path) -> VectorIndex:
        """Restore an index written by :meth:`save`."""
        target = Path(directory)
        meta_path = target / "index_meta.json"
        if not meta_path.is_file():
            raise ArtifactError(f"No index_meta.json in {target}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != INDEX_FORMAT_VERSION:
            raise ArtifactError(
                f"Index at {target} has version {meta.get('version')}, "
                f"this build expects {INDEX_FORMAT_VERSION}"
            )
        index = cls(
            dim=int(meta["dim"]),
            kind=str(meta["kind"]),
            metric=str(meta.get("metric", "cosine")),
            **dict(meta.get("params", {}).items()),
        )
        with (target / "vectors.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        index._vectors = np.asarray(payload["vectors"], dtype=np.float32)
        index.ids = list(payload["ids"])
        if index.kind != "numpy":
            faiss_path = target / "faiss.index"
            faiss = index._load_faiss()
            if faiss is not None and faiss_path.is_file():
                index._faiss_index = faiss.read_index(str(faiss_path))
        return index

    # -- helpers ------------------------------------------------------------
    def _load_faiss(self):
        if self._faiss is None and self.kind != "numpy":
            try:
                import faiss

                self._faiss = faiss
            except ImportError:
                self._faiss = False
        return self._faiss or None


__all__ = ["INDEX_FORMAT_VERSION", "SearchResult", "VectorIndex"]
