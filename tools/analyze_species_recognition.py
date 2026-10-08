"""All-species recognition statistics for the embedding architecture.

What this measures, and why it is not the same as the identity benchmark
-----------------------------------------------------------------------
The identity benchmark asks "is this the same cat?". This asks a different question: does the
descriptor space still *organise* when the population contains several species? That matters
because the architecture is used as a general-purpose retrieval backbone, and because a
descriptor that only separates cat individuals might have collapsed the rest of the visual
world in the process. A metric-learning run is capable of exactly that kind of forgetting, so
it is worth measuring rather than assuming.

Reported per model:

* **Species separation** — 1-NN accuracy and a linear probe over cat/dog labels, plus a
  confusion breakdown. Requires a corpus with more than one species.
* **Inter/intra-species distance ratio** — how much further apart the two species sit than
  images within one species. Scale-free, so it compares across models of different widths.
* **Descriptor health** — per-dimension variance and mean pairwise cosine. Both near-degenerate
  values indicate collapse, which is the failure this audit exists to catch.
* **Untrained-baseline slot** — pass ``--include-untrained`` to compare against a randomly
  initialised head on a pretrained backbone, which separates "the backbone was already good at
  this" from "training preserved it".

Labelling uses the Oxford-IIIT Pet convention: a capitalised file name is a cat. That is the
dataset's own documented rule and is verified against the per-image XML annotation elsewhere
in the pipeline, not trusted blindly.

Usage::

    python -m tools.analyze_species_recognition \
        --checkpoints artifacts/train/dinov2s-arcface/best.pt \
        --out docs/diagnostics/species-recognition.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.errors import CatFaceError
from catface.logging_utils import configure_utf8_console, get_logger
from catface.models.embedder import Embedder, EmbedderConfig, embed_records

LOGGER = get_logger("tools.species")


def oxford_records(image_dir: Path, limit_per_species: int | None = None) -> list[tuple[str, str, str]]:
    """Return ``(path, species, breed)`` for the Oxford corpus.

    Species comes from the dataset's documented convention: the file name begins with a
    capital letter for cats and a lower-case letter for dogs. The breed is the name prefix,
    which is what this dataset actually labels (see ``docs/data-findings.json``).
    """
    if not image_dir.is_dir():
        raise CatFaceError(f"Oxford image directory not found: {image_dir}")
    rows: list[tuple[str, str, str]] = []
    for path in sorted(image_dir.glob("*.jpg")):
        stem = path.stem
        species = "cat" if stem[:1].isupper() else "dog"
        rows.append((str(path), species, stem.rsplit("_", 1)[0]))
    if not rows:
        raise CatFaceError(f"No .jpg files under {image_dir}")

    if limit_per_species:
        kept: list[tuple[str, str, str]] = []
        counts: Counter = Counter()
        for row in rows:
            if counts[row[1]] < limit_per_species:
                kept.append(row)
                counts[row[1]] += 1
        rows = kept
    LOGGER.info(
        "Oxford corpus: %d images (%s)",
        len(rows), ", ".join(f"{k}={v}" for k, v in sorted(Counter(r[1] for r in rows).items())),
    )
    return rows


def embed_all(embedder: Embedder, rows: list[tuple[str, str, str]], batch_size: int) -> tuple[np.ndarray, list[str], list[str]]:
    """Embed every image, preserving order and dropping unreadable files consistently."""
    result = embed_records(embedder, [r[0] for r in rows], batch_size=batch_size)
    if result.vectors.size == 0:
        raise CatFaceError("embedding produced no vectors")
    kept_index = {path: i for i, path in enumerate(result.ids)}
    # ``embed_records`` may skip unreadable files, so re-align the labels to what it returned.
    aligned = [row for row in rows if row[0] in kept_index]
    if len(aligned) != len(result.ids):
        raise CatFaceError(
            f"label/vector mismatch: {len(aligned)} labels for {len(result.ids)} vectors"
        )
    return (
        result.vectors,
        [row[1] for row in aligned],
        [row[2] for row in aligned],
    )


def nearest_neighbour_accuracy(
    vectors: np.ndarray, labels: list[str], exclude_self: bool = True
) -> dict:
    """Leave-one-out 1-NN accuracy over the given label set."""
    similarity = vectors @ vectors.T
    if exclude_self:
        np.fill_diagonal(similarity, -np.inf)
    predicted = np.array(labels)[similarity.argmax(axis=1)]
    truth = np.array(labels)
    correct = predicted == truth
    confusion: Counter = Counter(zip(truth.tolist(), predicted.tolist()))
    label_values = sorted(set(labels))
    matrix = {
        actual: {pred: int(confusion.get((actual, pred), 0)) for pred in label_values}
        for actual in label_values
    }
    return {
        "accuracy": round(float(correct.mean()), 4),
        "num_items": len(labels),
        "classes": label_values,
        "confusion": matrix,
    }


def linear_probe(vectors: np.ndarray, labels: list[str], folds: int = 5) -> dict:
    """Cross-validated logistic-regression accuracy: is the label linearly decodable?

    A linear probe is the standard way to ask whether a representation *contains* a property,
    independently of how a nearest-neighbour lookup happens to behave.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError:  # pragma: no cover - benchmark extra
        return {"error": "scikit-learn not installed"}

    values = np.array(labels)
    counts = Counter(labels)
    if min(counts.values()) < folds:
        return {"skipped": f"smallest class has {min(counts.values())} items, need >= {folds}"}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1337)
    scores = cross_val_score(
        LogisticRegression(max_iter=2000, C=1.0), vectors, values, cv=splitter, n_jobs=1
    )
    return {
        "accuracy_mean": round(float(scores.mean()), 4),
        "accuracy_std": round(float(scores.std()), 4),
        "folds": folds,
        "chance_level": round(1.0 / len(counts), 4),
    }


def distance_structure(vectors: np.ndarray, labels: list[str]) -> dict:
    """Inter-species versus intra-species cosine distance, and descriptor health."""
    values = np.array(labels)
    similarity = vectors @ vectors.T
    same = values[:, None] == values[None, :]
    np.fill_diagonal(same, False)
    off_diagonal = ~np.eye(len(values), dtype=bool)

    def mean_similarity(mask: np.ndarray) -> float:
        return float(similarity[mask].mean()) if mask.any() else float("nan")

    inter = mean_similarity(off_diagonal & ~same)
    intra = mean_similarity(off_diagonal & same)
    return {
        "mean_similarity_inter_species": round(inter, 4),
        "mean_similarity_intra_species": round(intra, 4),
        "separation_gap": round(intra - inter, 4),
        # Scale-free: comparable across models with different descriptor widths.
        "ratio_intra_over_inter": round(intra / inter, 4) if inter else None,
        "descriptor_variance_mean": round(float(vectors.var(axis=0).mean()), 6),
        "descriptor_std_median": round(float(np.median(vectors.std(axis=0))), 6),
        "interpretation": (
            "separation_gap near 0 means the descriptor does not distinguish species at all; "
            "descriptor_variance_mean near 0 indicates a collapsed descriptor"
        ),
    }


def evaluate_model(
    embedder: Embedder,
    rows: list[tuple[str, str, str]],
    batch_size: int,
    label_key: int,
    include_probe: bool,
) -> dict:
    """Run every statistic for one model over one labelling."""
    vectors, species, breed = embed_all(embedder, rows, batch_size)
    labels = species if label_key == 1 else breed
    axis = "species" if label_key == 1 else "breed"
    report = {
        "descriptor_dim": int(vectors.shape[1]),
        "labelling": axis,
        "1nn_accuracy": nearest_neighbour_accuracy(vectors, labels)["accuracy"],
        "num_items": len(labels),
        "num_classes": len(set(labels)),
        "chance_level": round(1.0 / len(set(labels)), 4),
    }
    if label_key == 1:
        # Species-level statistics only: breed comparison is confounded when a model was
        # trained on one species, and species is what this audit is about.
        report["distance_structure"] = distance_structure(vectors, labels)
        nn = nearest_neighbour_accuracy(vectors, labels)
        report["confusion"] = nn["confusion"]
    if include_probe:
        report["linear_probe"] = linear_probe(vectors, labels)
    return report


def main(argv: list[str] | None = None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="All-species recognition statistics")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint paths. An entry may be 'label=path'.")
    parser.add_argument("--backbone", default="dinov2_vits14",
                        help="Backbone used when --include-untrained is set")
    parser.add_argument("--include-untrained", action="store_true",
                        help="Also measure a freshly initialised head, to separate "
                             "'the backbone already did this' from 'training preserved it'")
    parser.add_argument("--limit-per-species", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--breed-sample", type=int, default=3000,
                        help="Images used for the breed-level statistic (it is not the focus)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rows = oxford_records(Path(args.data_root) / "oxford_images", args.limit_per_species)

    models: list[tuple[str, Embedder]] = []
    for spec in args.checkpoints:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(label).parent.name, label
        embedder = Embedder.load(path, device=args.device or "cuda")
        models.append((label, embedder))
        LOGGER.info("loaded %s from %s", label, path)

    if args.include_untrained:
        reference = models[0][1]
        untrained = Embedder(
            EmbedderConfig(
                backbone=args.backbone, embedding_dim=reference.config.embedding_dim,
                pooling="auto", head="linear", image_size=reference.config.image_size,
                tta=reference.config.tta,
            ),
            num_classes=0, device=args.device or "cuda",
        )
        untrained.eval()
        models.append((f"untrained-head-{args.backbone}", untrained))
        LOGGER.info("added an untrained-head reference on %s", args.backbone)

    report: dict = {
        "corpus": {
            "name": "Oxford-IIIT Pet (all 37 breeds, cats and dogs)",
            "images": len(rows),
            "species_counts": dict(Counter(r[1] for r in rows)),
            "breeds": len({r[2] for r in rows}),
            "species_label_rule": "file name capitalised => cat (the dataset's documented rule)",
            "note": "identity here is the breed, which is what this corpus labels",
        },
        "models": {},
    }
    for label, embedder in models:
        started = time.perf_counter()
        LOGGER.info("=== %s: species-level statistics ===", label)
        species_report = evaluate_model(
            embedder, rows, args.batch_size, label_key=1, include_probe=True
        )
        # Breed level uses a bounded sample: it is a secondary statistic and the full corpus
        # would triple the cost for a number nobody acts on.
        breed_rows = rows[: args.breed_sample] if args.breed_sample else rows
        LOGGER.info("=== %s: breed-level statistics (%d images) ===", label, len(breed_rows))
        breed_report = evaluate_model(
            embedder, breed_rows, args.batch_size, label_key=0, include_probe=False
        )
        report["models"][label] = {
            "species": species_report,
            "breed": breed_report,
            "seconds": round(time.perf_counter() - started, 1),
            "embedder": embedder.describe_config(),
        }
        LOGGER.info(
            "%s: species 1-NN=%.4f gap=%.4f | breed 1-NN=%.4f",
            label, species_report["1nn_accuracy"],
            species_report["distance_structure"]["separation_gap"],
            breed_report["1nn_accuracy"],
        )

    # A verdict stated in words, so the report cannot be misread as a table of numbers.
    report["verdict"] = _verdict(report)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"written to {path}")
    return 0


def _verdict(report: dict) -> dict:
    """Summarise whether training preserved cross-species structure."""
    trained = {k: v for k, v in report["models"].items() if not k.startswith("untrained")}
    untrained = {k: v for k, v in report["models"].items() if k.startswith("untrained")}
    out: dict = {}
    for label, entry in trained.items():
        species = entry["species"]
        out[label] = {
            "species_1nn": species["1nn_accuracy"],
            "species_chance": species["chance_level"],
            "separation_gap": species["distance_structure"]["separation_gap"],
            "collapsed": species["distance_structure"]["descriptor_variance_mean"] < 1e-6,
        }
    if untrained:
        reference = next(iter(untrained.values()))["species"]
        for entry in out.values():
            delta = entry["species_1nn"] - reference["1nn_accuracy"]
            entry["species_1nn_vs_untrained_head"] = round(delta, 4)
            entry["verdict"] = (
                "training preserved cross-species structure"
                if delta > -0.02
                else "training REDUCED cross-species separation — the identity objective "
                     "partly overwrote the backbone's general organisation"
            )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
