"""Local file system, cropping and annotation-parser tests.

The cropping tests matter because crop geometry is a silent accuracy lever: a crop that
lands on a cat's chest still produces a unit-norm descriptor and a plausible score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catface.data.annotation import (
    parse_calfw_landmarks,
    parse_identity,
    parse_oiid_list,
    parse_oiid_xml,
)
from catface.data.cropping import (
    Box,
    alignment_affine,
    apply_affine,
    crop_face,
    heuristic_head_box,
    image_sha1,
    measure_quality,
    quality_flags,
)
from catface.data.prepare import infer_folder_identities
from catface.errors import DataError


class TestBox:
    def test_geometry(self):
        box = Box(10, 20, 40, 60)
        assert box.width == 30
        assert box.height == 40
        assert box.area == 1200

    def test_clip_keeps_the_box_inside_the_image(self):
        clipped = Box(-10, -5, 500, 400).clip(100, 80)
        assert (clipped.x1, clipped.y1) == (0, 0)
        assert clipped.x2 <= 100
        assert clipped.y2 <= 80

    def test_clip_produces_a_non_empty_box_for_marginal_input(self):
        clipped = Box(200, 200, 300, 300).clip(100, 100)
        assert clipped.area > 0

    def test_expand_to_square_makes_both_sides_equal(self):
        square = Box(0, 0, 100, 50).expand_to_square(0.0)
        assert square.width == square.height == 100

    def test_expand_to_square_honours_the_pad_ratio(self):
        square = Box(0, 0, 100, 100).expand_to_square(0.5)
        assert square.width == pytest.approx(200, abs=1)
        # The centre is preserved.
        assert (square.x1 + square.x2) / 2 == pytest.approx(50, abs=1)

    def test_zero_size_box_reports_zero_area(self):
        assert Box(10, 10, 10, 10).area == 0


class TestCropping:
    def _image(self, height: int = 200, width: int = 300) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[40:120, 120:200] = 255  # a bright "face"
        return image

    def test_crop_returns_a_square_tile_of_the_requested_size(self):
        tile, used, reason = crop_face(self._image(), Box(120, 40, 200, 120), tile=128)
        assert reason is None
        assert tile.shape == (128, 128, 3)
        assert used.width == used.height

    def test_crop_is_not_empty_for_a_box_at_the_image_edge(self):
        tile, _, reason = crop_face(self._image(), Box(0, 0, 60, 60), tile=64, pad_ratio=0.5)
        assert reason is None
        assert tile.shape == (64, 64, 3)

    def test_small_face_is_rejected_with_a_reason(self):
        tile, _, reason = crop_face(self._image(), Box(10, 10, 20, 20), min_face_px=48)
        assert reason == "face_too_small"
        assert tile.size == 0

    def test_unreadable_image_is_rejected(self):
        _, _, reason = crop_face(np.empty((0, 0, 3), np.uint8), Box(0, 0, 10, 10))
        assert reason == "unreadable_image"

    def test_crop_of_a_flat_image_preserves_the_flat_colour(self):
        image = np.full((100, 100, 3), 77, dtype=np.uint8)
        tile, _, reason = crop_face(image, Box(20, 20, 80, 80), tile=32)
        assert reason is None
        assert int(tile.mean()) == pytest.approx(77, abs=3)

    def test_upscale_path_is_used_when_the_crop_is_smaller_than_the_tile(self):
        # A 60 px face with pad_ratio 0.5 becomes a 120 px crop, still below the 256 px
        # tile, so the tile must be produced by upscaling rather than downscaling.
        tile, used, reason = crop_face(self._image(), Box(120, 40, 180, 100), tile=256, pad_ratio=0.5)
        assert reason is None
        assert tile.shape == (256, 256, 3)
        assert used.width < 256

    def test_face_below_the_minimum_size_is_rejected(self):
        tile, _, reason = crop_face(self._image(), Box(120, 40, 160, 80), min_face_px=48)
        assert reason == "face_too_small"
        assert tile.size == 0


class TestHeuristicHeadBox:
    def test_legacy_top_third_rule_is_reproducible(self):
        box = heuristic_head_box(Box(0, 0, 100, 300), head_ratio=0.34)
        # Face centre one third down, side length equal to the full body width.
        assert box.width == 100
        assert box.y2 - box.y1 == 100
        assert (box.y1 + box.y2) / 2 == pytest.approx(102, abs=1)

    def test_zero_size_box_does_not_produce_a_zero_width(self):
        box = heuristic_head_box(Box(10, 10, 10, 40))
        assert box.width >= 1


class TestQuality:
    def test_sharp_image_scores_higher_than_blurred(self):
        import cv2

        rng = np.random.default_rng(0)
        sharp = (rng.random((64, 64)) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
        assert measure_quality(sharp).sharpness > measure_quality(blurred).sharpness

    def test_flags_reflect_luminance_and_blur(self):

        flat = np.full((32, 32), 128, dtype=np.uint8)
        signals = measure_quality(flat)
        flags = quality_flags(signals)
        assert "blurry" in flags          # a flat image has ~zero Laplacian variance
        assert "low_contrast" in flags
        assert "underexposed" not in flags

    def test_dark_and_bright_images_are_flagged(self):
        dark = np.full((16, 16), 5, dtype=np.uint8)
        bright = np.full((16, 16), 250, dtype=np.uint8)
        assert "underexposed" in quality_flags(measure_quality(dark))
        assert "overexposed" in quality_flags(measure_quality(bright))

    def test_empty_tile_is_handled(self):
        signals = measure_quality(np.empty((0, 0), np.uint8))
        assert signals.sharpness == 0.0
        assert not np.isnan(signals.mean_luminance)

    def test_image_sha1_is_content_addressed(self):
        a = np.zeros((4, 4, 3), np.uint8)
        b = np.zeros((4, 4, 3), np.uint8)
        c = np.ones((4, 4, 3), np.uint8)
        assert image_sha1(a) == image_sha1(b)
        assert image_sha1(a) != image_sha1(c)


class TestAlignment:
    def test_affine_recovers_a_known_translation(self):
        source = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        target = source + np.array([5.0, -3.0])
        matrix = alignment_affine(source, target)
        moved = (matrix @ np.vstack([source.T, np.ones(4)]))
        assert np.allclose(moved, target.T, atol=1e-3)

    def test_landmark_count_mismatch_is_rejected(self):
        with pytest.raises(DataError, match="differ in shape"):
            alignment_affine(np.zeros((4, 2)), np.zeros((3, 2)))

    def test_too_few_landmarks_is_rejected(self):
        with pytest.raises(DataError, match="[Aa]t least 3"):
            alignment_affine(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_mostly_consistent_landmarks_are_accepted(self):
        """A couple of mistimed landmarks must not block an otherwise good alignment."""
        rng = np.random.default_rng(3)
        source = rng.uniform(0, 100, size=(20, 2)).astype(np.float32)
        target = source + np.array([12.0, -7.0], dtype=np.float32)
        target[5] += np.array([40.0, 40.0], dtype=np.float32)   # 1 bad landmark
        target[13] += np.array([-30.0, 25.0], dtype=np.float32)  # another bad one
        matrix = alignment_affine(source, target)
        # The transform must describe the translation the inliers share.
        assert matrix[0, 2] == pytest.approx(12.0, abs=2.0)
        assert matrix[1, 2] == pytest.approx(-7.0, abs=2.0)

    def test_incoherent_landmarks_are_rejected(self):
        """When almost no landmark pair agrees, the detection is wrong and must be refused.

        Sixteen of twenty correspondences are inconsistent, so no consensus transform
        exists and the caller has to know rather than silently aligning to garbage.
        """
        rng = np.random.default_rng(11)
        source = rng.uniform(0, 100, size=(20, 2)).astype(np.float32)
        target = rng.uniform(0, 100, size=(20, 2)).astype(np.float32)
        with pytest.raises(DataError, match="agree"):
            alignment_affine(source, target, ransac_threshold=0.5)

    def test_apply_affine_returns_the_requested_size(self):
        image = np.zeros((100, 100, 3), np.uint8)
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert apply_affine(image, matrix, 112).shape == (112, 112, 3)


class TestAnnotationParsers:
    def test_identity_is_the_filename_prefix(self, tmp_path):
        assert parse_identity("Abyssinian_100") == "Abyssinian"
        assert parse_identity("British_Shorthair_7") == "British_Shorthair"
        assert parse_identity("noNumericSuffix") == "noNumericSuffix"

    def test_list_parser_reads_four_columns(self, tmp_path):
        path = tmp_path / "list.txt"
        path.write_text(
            "#Image CLASS-ID SPECIES BREED ID\n"
            "Abyssinian_1 1 1 1\n"
            "newfoundland_1 27 2 12\n",
            encoding="utf-8",
        )
        rows = parse_oiid_list(path)
        assert rows[0] == ("Abyssinian_1", 1, "1", 1)
        assert rows[1][:2] == ("newfoundland_1", 27)

    def test_list_parser_rejects_wrong_column_count(self, tmp_path):
        path = tmp_path / "list.txt"
        path.write_text("Abyssinian_1 1 1\n", encoding="utf-8")
        with pytest.raises(DataError, match="4 columns"):
            parse_oiid_list(path)

    def test_list_parser_reports_an_empty_file(self, tmp_path):
        path = tmp_path / "list.txt"
        path.write_text("# only a comment\n", encoding="utf-8")
        with pytest.raises(DataError, match="no data rows"):
            parse_oiid_list(path)

    def test_list_parser_reports_a_missing_file(self, tmp_path):
        with pytest.raises(DataError, match="not found"):
            parse_oiid_list(tmp_path / "absent.txt")

    def test_xml_parser_reads_the_head_box(self, tmp_path):
        path = tmp_path / "Abyssinian_1.xml"
        path.write_text(
            "<annotation><size><width>600</width><height>400</height></size>"
            "<object><name>cat</name><pose>Frontal</pose><truncated>0</truncated>"
            "<occluded>1</occluded>"
            "<bndbox><xmin>333</xmin><ymin>72</ymin><xmax>425</xmax><ymax>158</ymax></bndbox>"
            "</object></annotation>",
            encoding="utf-8",
        )
        meta = parse_oiid_xml(path)
        assert meta["name"] == "cat"
        assert meta["head_box"] == (333, 72, 425, 158)
        assert meta["width"] == 600 and meta["height"] == 400
        assert meta["occluded"] is True
        assert meta["truncated"] is False

    def test_xml_parser_rejects_malformed_input(self, tmp_path):
        path = tmp_path / "broken.xml"
        path.write_text("<annotation><object>", encoding="utf-8")
        with pytest.raises((DataError, Exception)):
            parse_oiid_xml(path)

    def test_landmark_parser_handles_wide_format(self):
        payload = "1,2\n3,4\n5,6\n"
        parsed = parse_calfw_landmarks(payload)
        assert parsed.shape == (3, 2)
        assert parsed[1].tolist() == [3.0, 4.0]

    def test_landmark_parser_handles_long_format_with_header(self):
        payload = "index,x,y\n0,10,20\n1,30,40\n2,50,60\n"
        parsed = parse_calfw_landmarks(payload)
        assert parsed.shape == (3, 2)
        assert parsed[0].tolist() == [10.0, 20.0]

    def test_landmark_parser_rejects_too_few_points(self):
        with pytest.raises(DataError, match="at least 3"):
            parse_calfw_landmarks("1,2\n")


class TestFolderLayoutInference:
    def test_identity_from_parent_directory(self, tmp_path):
        import cv2

        image = np.zeros((8, 8, 3), np.uint8)
        for name in ("cat_a", "cat_b"):
            directory = tmp_path / name
            directory.mkdir()
            cv2.imwrite(str(directory / "img.jpg"), image)
        pairs = dict(infer_folder_identities(tmp_path))
        assert set(pairs.values()) == {"cat_a", "cat_b"}

    def test_identity_from_filename_prefix_when_flat(self, tmp_path):
        import cv2

        image = np.zeros((8, 8, 3), np.uint8)
        cv2.imwrite(str(tmp_path / "0001_000.jpg"), image)
        cv2.imwrite(str(tmp_path / "0002_003.jpg"), image)
        pairs = dict(infer_folder_identities(tmp_path))
        assert set(pairs.values()) == {"0001", "0002"}

    def test_unknown_layout_yields_an_empty_identity(self, tmp_path):
        import cv2

        cv2.imwrite(str(tmp_path / "lonely.jpg"), np.zeros((8, 8, 3), np.uint8))
        pairs = dict(infer_folder_identities(tmp_path))
        assert pairs[Path(tmp_path / "lonely.jpg")] == ""

    def test_missing_directory_is_reported(self, tmp_path):
        with pytest.raises(DataError, match="Directory not found"):
            infer_folder_identities(tmp_path / "absent")

    def test_empty_directory_is_reported(self, tmp_path):
        with pytest.raises(DataError, match="No images"):
            infer_folder_identities(tmp_path)

    def test_non_image_files_are_ignored(self, tmp_path):
        import cv2

        cv2.imwrite(str(tmp_path / "0001_0.jpg"), np.zeros((8, 8, 3), np.uint8))
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
        pairs = infer_folder_identities(tmp_path)
        assert len(pairs) == 1

    def test_result_order_is_deterministic(self, tmp_path):
        import cv2

        for index in range(5):
            cv2.imwrite(str(tmp_path / f"{index:04d}_0.jpg"), np.zeros((8, 8, 3), np.uint8))
        first = [str(p) for p, _ in infer_folder_identities(tmp_path)]
        second = [str(p) for p, _ in infer_folder_identities(tmp_path)]
        assert first == second
