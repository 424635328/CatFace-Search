"""CatFace Search — production-grade cat-face identity retrieval engine.

The package is organised as a pipeline of small, independently testable stages:

``catface.config``
    Typed, validated configuration objects (single source of truth for every run).
``catface.data``
    Dataset ingestion: annotation parsing, face cropping, alignment, splits, manifests.
``catface.models``
    Backbone registry, embedding heads (metric learning), and the ``Embedder`` facade.
``catface.index``
    Vector index (FAISS / NumPy) with normalisation, retrieval and post-processing.
``catface.eval``
    Identity-retrieval and verification metrics plus the benchmark runner.
``catface.cli``
    ``catface`` console entry point wiring the stages together.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "2.0.0"
