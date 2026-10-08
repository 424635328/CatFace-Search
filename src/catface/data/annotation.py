"""Annotation parsers: raw dataset metadata -> normalised records.

Each parser is pure (files in, records out) so it can be unit-tested against small
fixtures without downloading anything. Parsers never trust a field they can derive:
for example OIID's species is taken from ``list.txt`` rather than inferred from the
file-name capitalisation that the dataset README mentions, because the explicit
column is the documented contract and the capitalisation rule is a convention.
"""

from __future__ import annotations

import csv
import io
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import DataError
from ..logging_utils import get_logger

LOGGER = get_logger("data.annotation")

#: OIID species codes as documented in annotations/README.
OIID_SPECIES = {1: "cat", 2: "dog"}

#: File-name prefix that identifies one individual, e.g. ``Abyssinian_100`` -> ``Abyssinian``.
_IDENTITY_SUFFIX = re.compile(r"^(?P<identity>.+?)_(?P<index>\d+)$")


@dataclass(frozen=True)
class OiidEntry:
    """One row of OIID ``list.txt`` joined with its XML head-box annotation."""

    image_id: str
    """``Abyssinian_100`` — also the identity-bearing key."""
    filename: str
    class_id: int
    species: str
    breed_id: int
    identity: str
    head_box: tuple[int, int, int, int] | None
    image_width: int | None
    image_height: int | None
    pose: str | None
    truncated: bool
    occluded: bool

    @property
    def is_cat(self) -> bool:
        return self.species == "cat"


def parse_identity(image_id: str) -> str:
    """Extract the individual-animal label from an OIID image id.

    ``Abyssinian_100`` -> ``Abyssinian``. Images without the ``_<n>`` suffix keep
    their whole name, which guarantees a non-empty label rather than silently
    merging unrelated images under ``None``.
    """
    match = _IDENTITY_SUFFIX.match(image_id)
    return match.group("identity") if match else image_id


def parse_oiid_list(path: str | Path) -> list[tuple[str, int, str, int]]:
    """Parse ``annotations/list.txt``.

    The file's own header is misleading: it documents an ``ID SPECIES BREED`` layout,
    but the data rows carry **four** columns (``ID CLASS-ID SPECIES BREED``) where the
    ``SPECIES`` column repeats the class id. Species is therefore *not* taken from here
    — it is read from each image's XML annotation, where the object name is explicit.

    Returns:
        ``[(image_id, class_id, declared_species_field, breed_id), ...]``

    Raises:
        DataError: if the file is missing or malformed.
    """
    source = Path(path)
    if not source.is_file():
        raise DataError(f"OIID list.txt not found: {source}")
    rows: list[tuple[str, int, str, int]] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise DataError(f"{source}:{line_number} expected 4 columns, got {len(parts)}: {line!r}")
        image_id, class_id, declared_species, breed_id = parts
        try:
            int(class_id)
            int(breed_id)
        except ValueError as exc:
            raise DataError(f"{source}:{line_number} has a non-integer column: {line!r}") from exc
        rows.append((image_id, int(class_id), declared_species, int(breed_id)))
    if not rows:
        raise DataError(f"{source} contains no data rows")
    return rows


def parse_oiid_xml(path: str | Path) -> dict[str, object]:
    """Parse one OIID PASCAL-VOC annotation file."""
    source = Path(path)
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise DataError(f"Malformed XML {source}: {exc}") from exc

    size = root.find("size")
    width = int(size.findtext("width", "0")) if size is not None else 0
    height = int(size.findtext("height", "0")) if size is not None else 0

    objects = root.findall("object")
    if not objects:
        raise DataError(f"{source} declares no object")
    obj = objects[0]
    box = obj.find("bndbox")
    head: tuple[int, int, int, int] | None = None
    if box is not None:
        head = (
            int(float(box.findtext("xmin", "0"))),
            int(float(box.findtext("ymin", "0"))),
            int(float(box.findtext("xmax", "0"))),
            int(float(box.findtext("ymax", "0"))),
        )
    return {
        "name": (obj.findtext("name") or "").strip().lower(),
        "pose": (obj.findtext("pose") or "").strip() or None,
        "truncated": (obj.findtext("truncated") or "0").strip() == "1",
        "occluded": (obj.findtext("occluded") or "0").strip() == "1",
        "head_box": head,
        "width": width,
        "height": height,
    }


def load_oiid_entries(annotations_dir: str | Path, species: str = "cat") -> list[OiidEntry]:
    """Join ``list.txt`` with the per-image XML head boxes.

    Species is resolved from the ``object/name`` field inside each XML, which is the
    dataset's own machine-readable label (``cat`` / ``dog``). The README also documents
    a file-name capitalisation convention, but that is an informal hint and the XML is
    authoritative — relying on capitalisation silently produced a manifest of 37 breed
    classes instead of 7349 individuals during development.

    Args:
        annotations_dir: Directory containing ``list.txt`` and ``xmls/``.
        species: ``cat``, ``dog``, or ``any``.

    Returns:
        Entries sorted by ``image_id`` for reproducible iteration order.
    """
    root = Path(annotations_dir)
    xml_dir = root / "xmls"
    rows = parse_oiid_list(root / "list.txt")

    entries: list[OiidEntry] = []
    missing_xml = 0
    mismatched = 0
    for image_id, class_id, _declared_species, breed_id in rows:
        xml_path = xml_dir / f"{image_id}.xml"
        if not xml_path.is_file():
            missing_xml += 1
            continue
        meta = parse_oiid_xml(xml_path)
        xml_species = str(meta["name"] or "").strip().lower()
        if species != "any" and xml_species != species:
            continue
        if xml_species not in OIID_SPECIES.values() and xml_species:
            mismatched += 1
        entries.append(
            OiidEntry(
                image_id=image_id,
                filename=f"{image_id}.jpg",
                class_id=class_id,
                species=xml_species or "unknown",
                breed_id=breed_id,
                identity=parse_identity(image_id),
                head_box=meta["head_box"],  # type: ignore[arg-type]
                image_width=meta["width"],  # type: ignore[arg-type]
                image_height=meta["height"],  # type: ignore[arg-type]
                pose=meta["pose"],  # type: ignore[arg-type]
                truncated=bool(meta["truncated"]),
                occluded=bool(meta["occluded"]),
            )
        )
    if not entries:
        raise DataError(f"No OIID entries found for species={species!r} in {root}")
    if missing_xml:
        LOGGER.warning("Skipped %d images with no XML annotation", missing_xml)
    if mismatched:
        LOGGER.warning("%d XML files declared an unexpected object name", mismatched)
    distinct = len({entry.identity for entry in entries})
    LOGGER.info(
        "OIID %s: %d images, %d distinct identities", species, len(entries), distinct
    )
    return sorted(entries, key=lambda e: e.image_id)


def read_split_ids(path: str | Path) -> set[str] | None:
    """Read an OIID ``trainval.txt``/``test.txt`` file into a set of image ids.

    Returns ``None`` when the file does not exist, so callers can distinguish
    "official split unavailable" from "official split is empty".
    """
    source = Path(path)
    if not source.is_file():
        return None
    ids: set[str] = set()
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ids.add(line.split()[0])
    return ids


@dataclass(frozen=True)
class PairRecord:
    """A labelled same/different pair from a verification benchmark."""

    pair_id: str
    image_a: bytes
    image_b: bytes
    label: int
    """``1`` = same individual, ``0`` = different individuals."""


def load_parquet_pairs(path: str | Path, limit: int | None = None) -> list[PairRecord]:
    """Load a pair-verification parquet file (``image1``/``image2``/``target``).

    Images are kept as encoded bytes; decoding is deferred to the caller so the
    loader stays cheap and free of image-library coupling.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DataError(
            "pyarrow is required to read verification pairs; install it with "
            "`pip install pyarrow`"
        ) from exc

    source = Path(path)
    if not source.is_file():
        raise DataError(f"Pair file not found: {source}")

    table = pq.read_table(source)
    columns = set(table.column_names)
    required = {"image1", "image2", "target"}
    if not required <= columns:
        raise DataError(f"{source} must contain {sorted(required)}, found {sorted(columns)}")

    images_a = table.column("image1").to_pylist()
    images_b = table.column("image2").to_pylist()
    labels = table.column("target").to_pylist()
    total = len(labels) if limit is None else min(limit, len(labels))

    pairs: list[PairRecord] = []
    for index in range(total):
        blob_a, blob_b = images_a[index], images_b[index]
        if not blob_a or not blob_b:
            continue
        payload_a = blob_a.get("bytes") if isinstance(blob_a, Mapping) else blob_a
        payload_b = blob_b.get("bytes") if isinstance(blob_b, Mapping) else blob_b
        if payload_a is None or payload_b is None:
            continue
        pairs.append(
            PairRecord(
                pair_id=f"{source.stem}:{index}",
                image_a=payload_a,
                image_b=payload_b,
                label=int(labels[index]),
            )
        )
    if not pairs:
        raise DataError(f"{source} yielded no usable pairs")
    return pairs


def stream_parquet_images(path: str | Path, column: str = "img") -> Iterator[tuple[int, bytes]]:
    """Yield ``(row_index, encoded_bytes)`` from an image-only parquet file.

    Row-group streaming keeps peak memory bounded when materialising a 30k-image
    corpus to disk.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise DataError("pyarrow is required to read parquet images") from exc

    source = Path(path)
    if not source.is_file():
        raise DataError(f"Parquet file not found: {source}")
    handle = pq.ParquetFile(source)
    if column not in handle.schema_arrow.names:
        raise DataError(f"{source} has no column {column!r}")
    row = 0
    for batch in handle.iter_batches(batch_size=256, columns=[column]):
        for value in batch.column(0).to_pylist():
            payload = value.get("bytes") if isinstance(value, Mapping) else value
            if payload:
                yield row, payload
            row += 1


def write_pairs_csv(pairs: Sequence[PairRecord], images_dir: Path, csv_path: Path) -> Path:
    """Materialise verification pairs as image files plus a ``path_a,path_b,label`` CSV."""
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "path_a", "path_b", "label"])
        for pair in pairs:
            stem = pair.pair_id.replace(":", "_")
            path_a = images_dir / f"{stem}_a.png"
            path_b = images_dir / f"{stem}_b.png"
            path_a.write_bytes(pair.image_a)
            path_b.write_bytes(pair.image_b)
            writer.writerow([pair.pair_id, str(path_a), str(path_b), pair.label])
    return csv_path


def parse_calfw_landmarks(text: str) -> np.ndarray:
    """Parse CatFLW's ``keypoints.csv`` payload into an ``(n_landmarks, 2)`` array.

    Handles the two shapes found in the wild: a wide row of comma-separated
    ``x,y`` pairs, and an ``index, x, y`` long format.
    """
    rows = list(csv.reader(io.StringIO(text.strip())))
    if not rows:
        raise DataError("Empty landmark payload")

    wide: list[tuple[float, float]] = []
    for row in rows:
        cells = [c.strip() for c in row if c.strip() != ""]
        if not cells:
            continue
        try:
            numbers = [float(c) for c in cells]
        except ValueError:
            continue  # header row
        if len(numbers) == 2:
            wide.append((numbers[0], numbers[1]))
        elif len(numbers) >= 3:
            wide.append((numbers[1], numbers[2]))
    if len(wide) < 3:
        raise DataError(f"Could not parse at least 3 landmarks from payload of {len(rows)} rows")
    return np.asarray(wide, dtype=np.float64)


__all__ = [
    "OIID_SPECIES",
    "OiidEntry",
    "PairRecord",
    "load_oiid_entries",
    "load_parquet_pairs",
    "parse_calfw_landmarks",
    "parse_identity",
    "parse_oiid_list",
    "parse_oiid_xml",
    "read_split_ids",
    "stream_parquet_images",
    "write_pairs_csv",
]
