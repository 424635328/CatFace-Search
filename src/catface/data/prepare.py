"""Data preparation: raw corpora -> face crops -> a versioned manifest.

This module is the only place that decides *what a usable face image is*. Its output
(the manifest) is the input to every downstream stage, so its decisions are recorded
per row rather than applied silently.

Three source adapters are implemented:

``oiid``
    Oxford-IIIT Pet. Ships per-animal identity (file-name prefix), 12 cat breeds, and
    an annotated head box per image — the only corpus here with *both* identity labels
    and face boxes, which makes it the training corpus.
``cat_individuals``
    Kaggle *Cat Individual Images*: 13 536 photos of 518 cats. Real, uncontrolled cat
    photos with many images per individual — the evaluation corpus that actually
    resembles production traffic.
``calfw_pairs``
    Labelled same/different cat pairs, for verification metrics.
``folder``
    Any directory tree, inferred layout (``root/identity/*.jpg`` or
    ``root/identity_index.jpg``), so a user's own collection can be evaluated without
    writing code.

For an identity-labelled source the split is assigned **by identity**, never by image;
:func:`catface.eval.protocols.assert_identity_disjoint` re-checks it later.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..errors import DataError
from ..logging_utils import get_logger
from .cropping import (
    Box,
    crop_face,
    image_sha1,
    measure_quality,
    quality_flags,
)
from .manifest import (
    FaceRecord,
    Manifest,
    assign_identity_splits,
    write_split_files,
)

LOGGER = get_logger("data.prepare")

#: The 12 cat breeds in OIID, keyed by breed id from ``list.txt``.
OIID_CAT_BREEDS = {
    1: "Abyssinian", 2: "Bengal", 3: "Birman", 4: "Bombay", 5: "British_Shorthair",
    6: "Egyptian_Mau", 7: "Maine_Coon", 8: "Persian", 9: "Ragdoll",
    10: "Russian_Blue", 11: "Siamese", 12: "Sphynx",
}


@dataclass
class PrepareStats:
    """Counters describing what preparation accepted and rejected."""

    considered: int = 0
    written: int = 0
    rejected: dict[str, int] = None  # type: ignore[assignment]
    flags: dict[str, int] = None  # type: ignore[assignment]
    duplicates: int = 0

    def __post_init__(self) -> None:
        self.rejected = defaultdict(int)
        self.flags = defaultdict(int)

    def as_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "written": self.written,
            "rejected": dict(sorted(self.rejected.items())),
            "flags": dict(sorted(self.flags.items())),
            "duplicates_removed": self.duplicates,
        }


class _CropWriter:
    """Encodes and writes face tiles, tracking duplicates and quality flags."""

    def __init__(
        self,
        output_dir: Path,
        tile: int,
        pad_ratio: float,
        min_face_px: int,
        blur_var_threshold: float,
        luminance_range: tuple[int, int],
        dedupe: bool,
        quality: int = 95,
    ) -> None:
        import cv2

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tile = tile
        self.pad_ratio = pad_ratio
        self.min_face_px = min_face_px
        self.blur_var_threshold = blur_var_threshold
        self.luminance_range = luminance_range
        self.dedupe = dedupe
        self.quality = quality
        self._seen: dict[str, str] = {}
        self.stats = PrepareStats()
        self._cv2 = cv2

    def write(
        self,
        image: np.ndarray,
        box: Box,
        image_id: str,
        source: str,
        identity: str | None,
        source_image: str,
        detector: str,
        alignment: str = "none",
        breed: str | None = None,
    ) -> FaceRecord | None:
        """Crop, quality-check, encode and register one face tile.

        Returns ``None`` when the tile is unusable, after recording *why*.
        """
        self.stats.considered += 1
        tile, used_box, reason = crop_face(
            image, box, tile=self.tile, pad_ratio=self.pad_ratio, min_face_px=self.min_face_px
        )
        if reason is not None:
            self.stats.rejected[reason] += 1
            return None

        digest = image_sha1(tile)
        if self.dedupe and digest in self._seen:
            self.stats.duplicates += 1
            return None

        signals = measure_quality(tile)
        flags = list(quality_flags(signals, self.blur_var_threshold, self.luminance_range))
        for flag in flags:
            self.stats.flags[flag] += 1

        filename = f"{source}__{image_id.replace('/', '_')}.jpg"
        path = self.output_dir / filename
        # Encode deterministically so re-running preparation is idempotent.
        ok = self._cv2.imwrite(
            str(path), tile, [int(self._cv2.IMWRITE_JPEG_QUALITY), self.quality]
        )
        if not ok:
            self.stats.rejected["encode_failed"] += 1
            return None
        self._seen[digest] = str(path)
        self.stats.written += 1

        return FaceRecord(
            image_id=f"{source}:{image_id}",
            path=str(path),
            source=source,
            identity=identity,
            breed=breed,
            source_image=source_image,
            bbox_xyxy=(used_box.x1, used_box.y1, used_box.x2, used_box.y2),
            detector=detector,
            alignment=alignment,
            quality=signals.as_dict(),
            flags=tuple(flags),
            sha1=digest,
            width=self.tile,
            height=self.tile,
        )


# ---------------------------------------------------------------------------
# Oxford-IIIT Pet
# ---------------------------------------------------------------------------
def _iter_parquet_rows(path: Path, columns: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
    """Stream parquet rows in bounded batches."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise DataError("pyarrow is required to read parquet corpora") from exc
    handle = pq.ParquetFile(path)
    available = set(handle.schema_arrow.names)
    requested = [c for c in (columns or available) if c in available]
    for batch in handle.iter_batches(batch_size=64, columns=requested):
        yield from batch.to_pylist()


def prepare_oiid(
    parquet_paths: Mapping[str, Path],
    writer: _CropWriter,
    species: str = "cat",
    head_boxes: Mapping[str, Box] | None = None,
) -> Manifest:
    """Build a manifest from OIID parquet splits.

    Args:
        parquet_paths: ``{split_name: parquet_path}``.
        writer: Crop writer.
        species: ``cat`` or ``dog``. In the HuggingFace mirror ``label_cat_dog`` is
            ``0`` for cats and ``1`` for dogs — verified against ``image_id`` prefixes.
        head_boxes: Optional ``{image_id: Box}`` from the dataset's own XML head
            annotations. When supplied (strongly preferred) the crop uses the annotated
            head ROI, so a model comparison is not contaminated by detector error. When
            omitted the whole image is used and the limitation is recorded in the
            manifest as ``detector='oiid_full_image'``.
    """
    import cv2

    species_code = {"cat": 0, "dog": 1}[species]
    records: list[FaceRecord] = []
    head_boxes = head_boxes or {}
    if not head_boxes:
        LOGGER.warning(
            "No head boxes supplied for OIID; faces will be the full image, which "
            "makes this corpus a weaker proxy for face-only retrieval"
        )

    for split_name, path in parquet_paths.items():
        if not path.is_file():
            LOGGER.warning("Skipping missing OIID split %s (%s)", split_name, path)
            continue
        kept = 0
        for row in _iter_parquet_rows(path, ("image", "image_id", "label", "label_cat_dog")):
            if int(row.get("label_cat_dog", -1)) != species_code:
                continue
            image_id = str(row["image_id"])
            payload = row["image"].get("bytes") if isinstance(row["image"], Mapping) else None
            if payload is None:
                writer.stats.rejected["missing_image_bytes"] += 1
                continue
            buffer = np.frombuffer(payload, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                writer.stats.rejected["decode_failed"] += 1
                continue

            height, width = image.shape[:2]
            annotated = head_boxes.get(image_id)
            if annotated is not None:
                box = annotated.clip(width, height)
                detector = "oiid_head_box"
            else:
                box = Box(0, 0, width, height)
                detector = "oiid_full_image"
            record = writer.write(
                image=image,
                box=box,
                image_id=image_id,
                source="oiid_cat",
                identity=image_id.rsplit("_", 1)[0],
                source_image=f"oiid:{split_name}:{image_id}",
                detector=detector,
                breed=OIID_CAT_BREEDS.get(int(row.get("label", -1)) + 1),
            )
            if record is not None:
                records.append(record)
                kept += 1
        LOGGER.info("OIID split %s: wrote %d face tiles", split_name, kept)

    return Manifest(records)


def prepare_oiid_from_annotations(
    annotations_dir: Path,
    images_dir: Path,
    writer: _CropWriter,
    species: str = "cat",
) -> Manifest:
    """Build a manifest from the *original* OIID layout (list.txt + xmls + images).

    Preferred over the parquet mirror because the XML head box is the dataset's own
    face annotation, which is exactly the crop strategy this pipeline wants.
    """
    import cv2

    from .annotation import load_oiid_entries, read_split_ids

    entries = load_oiid_entries(annotations_dir, species=species)
    official_trainval = read_split_ids(annotations_dir / "trainval.txt") or set()
    official_test = read_split_ids(annotations_dir / "test.txt") or set()

    records: list[FaceRecord] = []
    missing_files = 0
    for entry in entries:
        image_path = images_dir / entry.filename
        if not image_path.is_file():
            missing_files += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            writer.stats.rejected["decode_failed"] += 1
            continue
        height, width = image.shape[:2]
        if entry.head_box:
            box = Box(*entry.head_box).clip(width, height)
            detector = "oiid_head_box"
        else:
            box = Box(0, 0, width, height)
            detector = "oiid_full_image"
        if entry.truncated:
            writer.stats.flags["truncated_subject"] += 1
        if entry.occluded:
            writer.stats.flags["occluded_subject"] += 1

        record = writer.write(
            image=image,
            box=box,
            image_id=entry.image_id,
            source="oiid_cat",
            identity=entry.identity,
            source_image=str(image_path),
            detector=detector,
            breed=OIID_CAT_BREEDS.get(entry.breed_id),
        )
        if record is not None:
            records.append(record)
    if missing_files:
        LOGGER.warning("%d OIID entries have no image on disk", missing_files)

    # Record the official split in a side file; the benchmark re-splits by identity.
    if official_trainval or official_test:
        (writer.output_dir.parent / "oiid_official_split.json").write_text(
            json.dumps(
                {
                    "trainval_count": len(official_trainval),
                    "test_count": len(official_test),
                    "note": (
                        "OIID ships a breed-stratified split in which the same animal can "
                        "appear in both parts; this project re-splits by identity."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return Manifest(records)


# ---------------------------------------------------------------------------
# Kaggle Cat Individual Images
# ---------------------------------------------------------------------------
def infer_folder_identities(root: Path, extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> list[tuple[Path, str]]:
    """Infer ``(path, identity)`` from a directory tree.

    Two layouts are recognised, and the decision is made per file so a mixed tree still
    works:

    * ``root/<identity>/<anything>.jpg`` — the parent directory is the identity.
    * ``root/<identity>_<index>.jpg`` — the file-name prefix is the identity.
    * ``root/anything.jpg`` — no identity; returned with an empty label.

    Returns:
        Sorted ``(path, identity)`` pairs for a deterministic manifest.
    """
    if not root.is_dir():
        raise DataError(f"Directory not found: {root}")
    lowered = {ext.lower() for ext in extensions}
    pairs: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in lowered:
            continue
        parent_name = path.parent.name
        if path.parent.resolve() != root.resolve() and parent_name:
            identity = parent_name
        else:
            stem = path.stem
            identity = stem.rsplit("_", 1)[0] if "_" in stem else ""
        pairs.append((path, identity))
    if not pairs:
        raise DataError(f"No images with extensions {sorted(lowered)} under {root}")
    return pairs


def prepare_cat_individuals(
    root: Path,
    writer: _CropWriter,
    source: str = "cat_individuals",
) -> Manifest:
    """Build a manifest from ``cat_individuals_dataset/<id>/<id>_<n>.JPG``.

    These photos are unannotated: there is no face box. The whole image is therefore
    used, and the manifest records ``detector='whole_image'`` so any weakness in the
    resulting scores is attributable to framing rather than to the model. Detecting a
    face box first would be better, and the crop strategy is a configurable axis for
    exactly this reason.
    """
    import cv2

    pairs = infer_folder_identities(root)
    counts = defaultdict(int)
    for _, identity in pairs:
        counts[identity] += 1
    LOGGER.info(
        "Cat-individuals: %d images across %d identities (median %d images/identity)",
        len(pairs), len(counts), int(np.median(list(counts.values()))),
    )

    records: list[FaceRecord] = []
    for path, identity in pairs:
        image = cv2.imread(str(path))
        if image is None:
            writer.stats.rejected["decode_failed"] += 1
            continue
        height, width = image.shape[:2]
        record = writer.write(
            image=image,
            box=Box(0, 0, width, height),
            image_id=path.relative_to(root).as_posix(),
            source=source,
            identity=identity or None,
            source_image=str(path),
            detector="whole_image",
        )
        if record is not None:
            records.append(record)
    return Manifest(records)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
IDENTITY_SOURCE_ADAPTERS: dict[str, Callable[..., Manifest]] = {
    "oiid_annotations": prepare_oiid_from_annotations,
    "oiid_parquet": prepare_oiid,
    "cat_individuals": prepare_cat_individuals,
}


def build_manifest(
    source: str,
    writer: _CropWriter,
    **kwargs: Any,
) -> Manifest:
    """Dispatch to a source adapter by name."""
    if source not in IDENTITY_SOURCE_ADAPTERS:
        raise DataError(
            f"Unknown source adapter {source!r}. Known: {sorted(IDENTITY_SOURCE_ADAPTERS)}"
        )
    return IDENTITY_SOURCE_ADAPTERS[source](writer=writer, **kwargs)


def make_writer(config: Any) -> _CropWriter:
    """Create a crop writer from a :class:`catface.config.DataConfig`."""
    return _CropWriter(
        output_dir=Path(config.crops_dir),
        tile=int(config.tile),
        pad_ratio=float(config.pad_ratio),
        min_face_px=int(config.min_face_px),
        blur_var_threshold=float(config.blur_var_threshold),
        luminance_range=tuple(config.luminance_range),  # type: ignore[arg-type]
        dedupe=bool(config.dedupe),
    )


def finalise_manifest(
    manifest: Manifest,
    output_dir: Path,
    source_name: str,
    seed: int = 1337,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    min_images_per_identity: int = 2,
    stats: PrepareStats | None = None,
) -> dict[str, Any]:
    """Split by identity, persist the manifest + split files, and return a report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest.save(output_dir / f"{source_name}_manifest.jsonl")

    summary = manifest.stats().summary()
    report: dict[str, Any] = {
        "source": source_name,
        "manifest": str(manifest_path),
        "stats": summary,
        "prepare": stats.as_dict() if stats is not None else None,
    }

    labeled = manifest.filter(require_label=True, min_identity_images=min_images_per_identity)
    if len(labeled) < 2:
        report["split"] = {"error": "not enough labelled images to split"}
        return report

    try:
        assignment = assign_identity_splits(
            labeled, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )
    except DataError as exc:
        report["split"] = {"error": str(exc)}
        return report

    id_to_split = {r.image_id: assignment.get(r.identity, "unassigned") for r in labeled}
    write_split_files(id_to_split, output_dir / f"{source_name}_splits")

    counts: dict[str, int] = defaultdict(int)
    identity_counts: dict[str, int] = defaultdict(int)
    for record in labeled:
        split = assignment.get(record.identity, "unassigned")
        counts[split] += 1
        if split == "train":
            identity_counts[record.identity] += 1

    report["split"] = {
        "policy": "identity-disjoint",
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "images_per_split": dict(sorted(counts.items())),
        "identities_per_split": {
            name: sum(1 for identity, split in assignment.items() if split == name)
            for name in ("train", "val", "test")
        },
        "splits_dir": str(output_dir / f"{source_name}_splits"),
    }
    (output_dir / f"{source_name}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def import_split_manifest(
    manifest_path: Path,
    split_file: Path,
    identity_to_split: Mapping[str, str] | None = None,
) -> dict[str, Manifest]:
    """Load a manifest and group it into ``{'train': ..., 'val': ..., 'test': ...}``."""
    manifest = Manifest.load(manifest_path)
    if not split_file.is_file():
        raise DataError(f"Split file not found: {split_file}")
    desired = {line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    selected = [r for r in manifest if r.image_id in desired]
    if not selected:
        raise DataError(f"No manifest rows match the ids in {split_file}")
    if identity_to_split is None:
        return {"all": Manifest(selected)}
    grouped: dict[str, list[FaceRecord]] = defaultdict(list)
    for record in selected:
        grouped[identity_to_split.get(record.identity or "", "unassigned")].append(record)
    return {name: Manifest(rows) for name, rows in grouped.items()}


__all__ = [
    "IDENTITY_SOURCE_ADAPTERS",
    "OIID_CAT_BREEDS",
    "PrepareStats",
    "build_manifest",
    "finalise_manifest",
    "import_split_manifest",
    "infer_folder_identities",
    "make_writer",
    "prepare_cat_individuals",
    "prepare_oiid",
    "prepare_oiid_from_annotations",
]
