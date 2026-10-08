"""Data-layer exports."""

from __future__ import annotations

from .annotation import (
    OiidEntry,
    PairRecord,
    load_oiid_entries,
    load_parquet_pairs,
    parse_identity,
    parse_oiid_list,
    parse_oiid_xml,
    read_split_ids,
    stream_parquet_images,
    write_pairs_csv,
)
from .manifest import (
    FaceRecord,
    Manifest,
    ManifestStats,
    assign_identity_splits,
    sha1_file,
)
from .sources import CALFW_PAIRS, CATALOGUE, CATFACES29K, CATFLW, OXFORD_PETS, DatasetStore

__all__ = [
    "CALFW_PAIRS",
    "CATALOGUE",
    "CATFACES29K",
    "CATFLW",
    "OXFORD_PETS",
    "DatasetStore",
    "FaceRecord",
    "Manifest",
    "ManifestStats",
    "OiidEntry",
    "PairRecord",
    "assign_identity_splits",
    "load_oiid_entries",
    "load_parquet_pairs",
    "parse_identity",
    "parse_oiid_list",
    "parse_oiid_xml",
    "read_split_ids",
    "sha1_file",
    "stream_parquet_images",
    "write_pairs_csv",
]
