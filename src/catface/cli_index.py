"""Index build/search command implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .data.manifest import Manifest
from .errors import DataError
from .index.faiss_index import VectorIndex
from .logging_utils import get_logger, timed
from .models.embedder import embed_records
from .pipeline import make_embedder

LOGGER = get_logger("cli.index")


def build_and_search(
    config: PipelineConfig,
    checkpoint: str | Path,
    manifest_path: str | Path,
    kind: str = "flat_ip",
    query: str | Path | None = None,
    top_k: int = 10,
    device: str | None = None,
) -> dict[str, Any]:
    """Embed a manifest into a vector index, then optionally run one query.

    Both halves are deliberately in one function: the index and the query must use the
    same embedder, and building a query path that loads a *different* model is the most
    common way to ship a search engine that quietly returns nonsense.
    """
    manifest_file = Path(manifest_path)
    manifest = Manifest.load(manifest_file)
    records = list(manifest)
    if not records:
        raise DataError(f"Manifest {manifest_file} is empty")

    embedder = make_embedder(config, device=device, checkpoint=checkpoint)
    with timed(LOGGER, "embed manifest", stage="index", images=len(records)):
        embedding = embed_records(
            embedder,
            [r.path for r in records],
            image_size=embedder.config.image_size,
            batch_size=32,
        )
    if embedding.vectors.size == 0:
        raise DataError("Embedding produced no vectors")

    ids = [record.image_id for record in records]
    labels = [record.identity for record in records]
    index = VectorIndex(dim=int(embedding.vectors.shape[1]), kind=kind)
    index.add(embedding.vectors, ids)
    index.build()

    output_dir = Path(config.index.output_dir) / f"{manifest_file.stem}-{kind}"
    index.save(output_dir, embeddings=embedding.vectors)
    (output_dir / "labels.json").write_text(
        json.dumps(dict(zip(ids, labels)), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report: dict[str, Any] = {
        "index_dir": str(output_dir),
        "backend": index.backend,
        "kind": kind,
        "vectors": index.size,
        "dim": index.dim,
        "labeled": sum(1 for label in labels if label),
        "embedder": embedder.describe_config(),
        "query": None,
    }

    if query is not None:
        query_path = Path(query)
        if not query_path.is_file():
            raise DataError(f"Query image not found: {query_path}")
        result = embed_records(embedder, [str(query_path)], batch_size=1)
        if result.vectors.size == 0:
            raise DataError(f"Could not embed query image {query_path}")
        # Map index ids back to positions so the self-match can be reported, not removed:
        # a production caller usually wants to see "this image is already in the index".
        id_to_position = {identifier: position for position, identifier in enumerate(ids)}
        search = index.search(result.vectors, top_k=top_k)
        neighbours = []
        for identifier, score in zip(search.ids[0], search.scores[0]):
            neighbours.append({
                "image_id": identifier,
                "identity": labels[id_to_position[identifier]] if identifier in id_to_position else None,
                "path": records[id_to_position[identifier]].path if identifier in id_to_position else None,
                "similarity": float(score),
            })
        report["query"] = {
            "path": str(query_path),
            "embedding_dim": int(result.vectors.shape[1]),
            "top_k": top_k,
            "neighbours": neighbours,
        }
        LOGGER.info("Query returned %d neighbours (best %.4f)",
                    len(neighbours), neighbours[0]["similarity"] if neighbours else float("nan"))
    return report


__all__ = ["build_and_search"]
