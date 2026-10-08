"""Manifest, splitting and leakage-prevention tests.

Split correctness is a benchmarking concern, not a housekeeping one: an identity that
appears in both train and test turns a model comparison into a memorisation contest.
"""

from __future__ import annotations

import json

import pytest

from catface.data.manifest import (
    MANIFEST_VERSION,
    FaceRecord,
    Manifest,
    assign_identity_splits,
    sha1_file,
    write_split_files,
)
from catface.errors import DataError
from catface.eval.protocols import assert_identity_disjoint, build_identity_split


def make_record(
    image_id: str,
    identity: str | None,
    path: str = "/tmp/x.jpg",
    source: str = "test",
    **kwargs,
) -> FaceRecord:
    return FaceRecord(
        image_id=image_id, path=path, source=source, identity=identity, **kwargs
    )


class TestFaceRecord:
    def test_json_round_trip_preserves_types(self):
        original = FaceRecord(
            image_id="a:1",
            path="/tmp/a.jpg",
            source="a",
            identity="cat_1",
            bbox_xyxy=(1, 2, 3, 4),
            flags=("blurry",),
            quality={"sharpness": 12.5},
        )
        restored = FaceRecord.from_json(json.loads(json.dumps(original.as_json())))
        assert restored.bbox_xyxy == (1, 2, 3, 4)
        assert restored.flags == ("blurry",)
        assert restored.quality["sharpness"] == pytest.approx(12.5)

    def test_unknown_field_is_rejected(self):
        payload = make_record("a:1", "cat").as_json()
        payload["surprise"] = 1
        with pytest.raises(DataError, match="unknown field"):
            FaceRecord.from_json(payload)

    def test_is_labeled_requires_a_truthy_identity(self):
        assert make_record("a:1", "cat").is_labeled
        assert not make_record("a:1", None).is_labeled
        assert not make_record("a:1", "").is_labeled


class TestManifest:
    def test_duplicate_ids_are_rejected_at_construction(self):
        with pytest.raises(DataError, match="Duplicate image_id"):
            Manifest([make_record("a:1", "c"), make_record("a:1", "c")])

    def test_by_id_raises_for_unknown(self):
        with pytest.raises(DataError, match="Unknown image_id"):
            Manifest([make_record("a:1", "c")]).by_id("nope")

    def test_filter_by_source_and_identity_size(self):
        records = [
            make_record("a:1", "cat_a", source="a"),
            make_record("a:2", "cat_a", source="a"),
            make_record("b:1", "cat_b", source="b"),
            make_record("a:3", "cat_c", source="a"),
        ]
        manifest = Manifest(records)
        assert len(manifest.filter(sources=["a"])) == 3
        assert len(manifest.filter(min_identity_images=2)) == 2
        assert len(manifest.filter(require_label=True)) == 4
        assert len(manifest.filter(identities=["cat_b"])) == 1
        # Filters compose: source 'a' and identity size >= 2 leaves only cat_a.
        assert len(manifest.filter(sources=["a"], min_identity_images=2)) == 2

    def test_filter_excludes_flagged_rows_by_default(self):
        records = [
            make_record("a:1", "cat_a"),
            make_record("a:2", "cat_a", flags=("duplicate",)),
        ]
        manifest = Manifest(records)
        assert len(manifest.filter()) == 1
        assert len(manifest.filter(exclude_flags=())) == 2

    def test_stats_report_identity_shape(self):
        records = [
            make_record("a:1", "cat_a"),
            make_record("a:2", "cat_a"),
            make_record("a:3", "cat_b"),
            make_record("a:4", None),
        ]
        stats = Manifest(records).stats()
        assert stats.total == 4
        assert stats.labeled == 3
        assert stats.unlabeled == 1
        assert stats.identities == 2
        assert stats.singleton_identities == 1
        summary = stats.summary()
        assert summary["identities_with_ge2"] == 1

    def test_save_and_load_round_trip(self, tmp_path, identity_records):
        manifest = Manifest(identity_records)
        path = manifest.save(tmp_path / "m.jsonl")
        restored = Manifest.load(path)
        assert len(restored) == len(manifest)
        assert restored[0].image_id == manifest[0].image_id

    def test_load_rejects_a_future_version(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text(
            json.dumps({"_manifest_version": MANIFEST_VERSION + 1}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(DataError, match="version"):
            Manifest.load(path)

    def test_load_reports_the_offending_line(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text(
            json.dumps({"_manifest_version": MANIFEST_VERSION}) + "\n" + "{not json}\n",
            encoding="utf-8",
        )
        with pytest.raises((DataError, json.JSONDecodeError)):
            Manifest.load(path)

    def test_load_missing_file_is_reported(self, tmp_path):
        with pytest.raises(DataError, match="not found"):
            Manifest.load(tmp_path / "absent.jsonl")

    def test_sha1_file_is_content_addressed(self, tmp_path):
        first = tmp_path / "a.bin"
        second = tmp_path / "b.bin"
        first.write_bytes(b"hello")
        second.write_bytes(b"hello")
        third = tmp_path / "c.bin"
        third.write_bytes(b"world")
        assert sha1_file(first) == sha1_file(second)
        assert sha1_file(first) != sha1_file(third)


class TestIdentitySplits:
    def _manifest(self, n_identities: int = 20, per_identity: int = 4) -> Manifest:
        records = [
            make_record(f"{identity}:{index}", identity)
            for identity in (f"id{i:03d}" for i in range(n_identities))
            for index in range(per_identity)
        ]
        return Manifest(records)

    def test_split_is_disjoint_by_identity(self):
        """No identity may appear in more than one split. Images may repeat; identities may not."""
        manifest = self._manifest()
        assignment = assign_identity_splits(manifest, seed=1)
        identities_per_split: dict[str, set[str]] = {}
        for identity, split in assignment.items():
            identities_per_split.setdefault(split, set()).add(identity)

        assert set(identities_per_split) == {"train", "val", "test"}
        assert not (identities_per_split["train"] & identities_per_split["test"])
        assert not (identities_per_split["train"] & identities_per_split["val"])
        assert not (identities_per_split["val"] & identities_per_split["test"])
        assert sum(len(ids) for ids in identities_per_split.values()) == len(assignment)

    def test_split_is_deterministic_for_a_seed(self):
        manifest = self._manifest()
        a = assign_identity_splits(manifest, seed=42)
        b = assign_identity_splits(manifest, seed=42)
        c = assign_identity_splits(manifest, seed=43)
        assert a == b
        assert a != c

    def test_all_three_splits_are_non_empty(self):
        assignment = assign_identity_splits(self._manifest(), seed=2)
        assert {"train", "val", "test"} <= set(assignment.values())

    def test_too_few_identities_raises(self):
        with pytest.raises(DataError, match="No identity"):
            assign_identity_splits(Manifest([make_record("a:1", "only")]))

    def test_write_split_files_lists_ids(self, tmp_path):
        assignment = {"id_a": "train", "id_b": "test"}
        written = write_split_files(assignment, tmp_path)
        assert set(written) == {"train", "test"}
        assert (tmp_path / "train.txt").read_text(encoding="utf-8").strip() == "id_a"
        assert (tmp_path / "test.txt").read_text(encoding="utf-8").strip() == "id_b"


class TestLeakageDetection:
    def test_disjoint_split_passes(self):
        train = [make_record("a:1", "cat_a")]
        test = [make_record("a:2", "cat_b")]
        assert_identity_disjoint({"train": train, "test": test})

    def test_leak_raises_by_default(self):
        train = [make_record("a:1", "cat_a")]
        test = [make_record("a:2", "cat_a")]
        with pytest.raises(Exception, match="leakage"):
            assert_identity_disjoint({"train": train, "test": test})

    def test_tolerance_allows_explicitly_accepted_overlap(self):
        train = [make_record("a:1", "cat_a")]
        test = [make_record("a:2", "cat_a")]
        assert_identity_disjoint({"train": train, "test": test}, tolerance=1)


class TestProtocol:
    def _records(self, n_identities: int = 10, per_identity: int = 4) -> list[FaceRecord]:
        return [
            make_record(f"id{i:02d}:{index}", f"id{i:02d}", path=f"/tmp/{i}_{index}.jpg")
            for i in range(n_identities)
            for index in range(per_identity)
        ]

    def test_every_identity_is_queryable_and_has_a_gallery_entry(self):
        split = build_identity_split(self._records(), queries_per_identity=1, seed=5)
        assert set(split.query_labels.tolist()) == set(split.gallery_labels.tolist())
        assert split.num_queries == 10
        assert split.num_gallery == 30

    def test_self_mask_never_contains_the_query_itself(self):
        split = build_identity_split(self._records(), queries_per_identity=1, seed=5)
        assert not split.self_occurrences.any()

    def test_gallery_cap_is_respected(self):
        split = build_identity_split(
            self._records(per_identity=10), queries_per_identity=1,
            max_gallery_per_identity=3, seed=5,
        )
        assert split.num_gallery == 30

    def test_queries_per_identity_is_honoured(self):
        split = build_identity_split(self._records(), queries_per_identity=2, seed=5)
        assert split.num_queries == 20
        assert split.num_gallery == 20

    def test_identities_with_a_single_image_are_dropped(self):
        records = [*self._records(n_identities=3, per_identity=2), make_record("solo:0", "solo")]
        split = build_identity_split(records, queries_per_identity=1, seed=5)
        assert "solo" not in set(split.query_labels.tolist())

    def test_single_identity_still_yields_a_valid_protocol(self):
        """One identity cannot be split, but it can still be query/gallery evaluated.

        This documents the actual behaviour: with 4 images of one individual and
        ``queries_per_identity=1``, one image is the query and three are references, and
        the protocol is evaluable. Splitting that identity across train/test is what
        must be prevented — that is ``assign_identity_splits``'s job, tested above.
        """
        records = [make_record(f"a:{i}", "only") for i in range(4)]
        split = build_identity_split(records, queries_per_identity=1)
        assert split.num_queries == 1
        assert split.num_gallery == 3
        assert split.query_labels.tolist() == ["only"]

    def test_identity_with_too_few_images_cannot_be_queried(self):
        records = [*self._records(n_identities=3, per_identity=3), make_record("solo:0", "solo")]
        split = build_identity_split(records, queries_per_identity=3)
        assert "solo" not in set(split.query_labels.tolist())

    def test_describe_reports_gallery_shape(self):
        split = build_identity_split(self._records(), queries_per_identity=1, seed=5)
        described = split.describe()
        assert described["queries"] == 10
        assert described["gallery"] == 30
        assert described["identities_in_both"] == 10
        assert described["gallery_images_per_identity_mean"] == pytest.approx(3.0)
