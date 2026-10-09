"""Face cropping: from a source image to a square, quality-checked face tile.

The original pipeline cropped the top third of a whole-body cat detection box and
called it a face. That is a *heuristic*, and its error mode is silent: on a sitting
cat it lands on the chest, and the resulting "face" descriptor is dominated by fur
pattern and background. This module makes the crop strategy explicit and measurable:

``annotation_box``
    Use a supplied face/head box (OIID head ROI, CatFLW face box). Most accurate,
    and the only strategy used for the benchmark so that detector quality cannot
    pollute a model comparison.
``heuristic_head``
    Reproduce the legacy top-third rule, so its effect can be quantified and compared
    against the alternatives instead of being assumed.
``whole_image``
    No cropping — the honest control condition for "does face cropping help at all".

Every crop records its own quality signals (sharpness, luminance, face area share) so
unusable tiles can be excluded by data rather than by guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..errors import DataError
from ..logging_utils import get_logger

LOGGER = get_logger("data.cropping")


class Box(NamedTuple):
    """An axis-aligned box in source-image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def clip(self, image_width: int, image_height: int) -> Box:
        """Clamp to image bounds so slicing can never produce an empty array."""
        return Box(
            max(0, min(self.x1, image_width - 1)),
            max(0, min(self.y1, image_height - 1)),
            max(1, min(self.x2, image_width)),
            max(1, min(self.y2, image_height)),
        )

    def expand_to_square(self, pad_ratio: float) -> Box:
        """Grow the box to a square around its centre, adding a context margin.

        Squareness is not cosmetic: a non-square crop that is later squashed to a
        square input distorts the face geometry, which measurably hurts landmark-free
        matching.
        """
        side = max(self.width, self.height)
        side = round(side * (1.0 + 2.0 * pad_ratio))
        centre_x = (self.x1 + self.x2) / 2.0
        centre_y = (self.y1 + self.y2) / 2.0
        half = side / 2.0
        return Box(
            round(centre_x - half),
            round(centre_y - half),
            round(centre_x + half),
            round(centre_y + half),
        )


@dataclass(frozen=True)
class QualitySignals:
    """Numeric quality measures computed on the *cropped tile*."""

    sharpness: float
    """Variance of the Laplacian; low values indicate motion/defocus blur."""
    mean_luminance: float
    """Mean brightness in ``[0, 255]``."""
    contrast: float
    """Standard deviation of luminance."""
    face_fraction: float
    """Share of the tile occupied by the face region, in ``[0, 1]``."""

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpness": float(self.sharpness),
            "mean_luminance": float(self.mean_luminance),
            "contrast": float(self.contrast),
            "face_fraction": float(self.face_fraction),
        }


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Focus measure (Pech-Pacheco et al.). Higher is sharper."""
    import cv2

    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_quality(tile: np.ndarray, face_fraction: float = 1.0) -> QualitySignals:
    """Compute quality signals for a BGR or colour tile."""
    import cv2

    if tile.size == 0:
        return QualitySignals(0.0, 0.0, 0.0, 0.0)
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.ndim == 3 else tile
    return QualitySignals(
        sharpness=variance_of_laplacian(gray),
        mean_luminance=float(gray.mean()),
        contrast=float(gray.std()),
        face_fraction=float(face_fraction),
    )


def quality_flags(
    signals: QualitySignals,
    blur_var_threshold: float = 18.0,
    luminance_range: tuple[int, int] = (25, 235),
) -> tuple[str, ...]:
    """Turn quality signals into explicit, inspectable flags."""
    flags: list[str] = []
    if signals.sharpness < blur_var_threshold:
        flags.append("blurry")
    low, high = luminance_range
    if signals.mean_luminance < low:
        flags.append("underexposed")
    elif signals.mean_luminance > high:
        flags.append("overexposed")
    if signals.contrast < 12.0:
        flags.append("low_contrast")
    return tuple(flags)


def crop_face(
    image: np.ndarray,
    box: Box,
    tile: int = 256,
    pad_ratio: float = 0.35,
    min_face_px: int = 48,
) -> tuple[np.ndarray, Box, str | None]:
    """Crop and resize one face tile.

    Args:
        image: Source image (BGR as read by OpenCV).
        box: Face/head box in source coordinates.
        tile: Output side length in pixels.
        pad_ratio: Context margin around the box.
        min_face_px: Reject boxes whose shorter side is below this.

    Returns:
        ``(tile_image, used_box, reject_reason)``. ``reject_reason`` is ``None`` on
        success; otherwise ``tile_image`` is an empty array.
    """
    import cv2

    if image is None or image.size == 0:
        return np.empty((0, 0, 3), np.uint8), box, "unreadable_image"
    height, width = image.shape[:2]
    if box.width < min_face_px or box.height < min_face_px:
        return np.empty((0, 0, 3), np.uint8), box, "face_too_small"

    square = box.expand_to_square(pad_ratio).clip(width, height)
    if square.area < min_face_px * min_face_px:
        return np.empty((0, 0, 3), np.uint8), box, "crop_too_small"

    patch = image[square.y1 : square.y2, square.x1 : square.x2]
    if patch.size == 0:
        return np.empty((0, 0, 3), np.uint8), square, "empty_crop"

    interpolation = cv2.INTER_AREA if patch.shape[0] > tile else cv2.INTER_CUBIC
    resized = cv2.resize(patch, (tile, tile), interpolation=interpolation)
    return resized, square, None


def heuristic_head_box(body_box: Box, head_ratio: float = 0.34) -> Box:
    """Reproduce the legacy "top third of the body box" face approximation.

    Kept deliberately, so the cost of that approximation can be *measured* against
    annotated boxes rather than argued about.
    """
    centre_x = body_box.x1 + body_box.width // 2
    head_centre_y = body_box.y1 + int(body_box.height * head_ratio)
    half = max(body_box.width // 2, 1)
    return Box(centre_x - half, head_centre_y - half, centre_x + half, head_centre_y + half)


def alignment_affine(
    source_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    ransac_threshold: float = 3.0,
) -> np.ndarray:
    """Similarity transform (rotation + uniform scale + translation) between landmark sets.

    A similarity transform is used rather than a full affine because a face that is
    stretched horizontally is no longer the same shape — the anisotropic degrees of
    freedom actively hurt matching.

    RANSAC is used rather than a least-median estimator because the criterion that
    matters is "do *most* landmarks agree to within a few pixels", which cannot be
    expressed as a median-of-residuals on a handful of points. The inlier ratio is then
    checked explicitly: a transform fitted to a minority of landmarks means the detector
    locked onto something that is not a face, and returning it silently would poison every
    downstream descriptor.

    Args:
        source_landmarks: ``(n, 2)`` landmarks on the input image.
        target_landmarks: ``(n, 2)`` reference landmarks, same count and order.
        ransac_threshold: Max residual (px) for a pair to count as an inlier.

    Raises:
        DataError: If the sets differ in size, have fewer than 3 points, no transform can
            be estimated, or fewer than half the landmarks agree.
    """
    import cv2

    source = np.asarray(source_landmarks, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target_landmarks, dtype=np.float32).reshape(-1, 2)
    if source.shape != target.shape:
        raise DataError(f"Landmark sets differ in shape: {source.shape} vs {target.shape}")
    if source.shape[0] < 3:
        raise DataError("At least 3 landmark pairs are required for a similarity transform")

    # ``estimateAffinePartial2D`` returns a 2x3 matrix for a similarity transform.
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold),
        maxIters=2000,
        confidence=0.995,
        refineIters=50,
    )
    if matrix is None:
        raise DataError("cv2.estimateAffinePartial2D failed to find a transform")
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else source.shape[0]
    if inliers < source.shape[0] * 0.5:
        raise DataError(
            f"Only {inliers}/{source.shape[0]} landmarks agree within "
            f"{ransac_threshold:.1f} px; the detection is probably not a face"
        )
    return matrix


def apply_affine(image: np.ndarray, matrix: np.ndarray, size: int) -> np.ndarray:
    """Warp an image by the 2x3 affine matrix and resize to ``size`` square."""
    import cv2

    return cv2.warpAffine(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def image_sha1(array: np.ndarray) -> str:
    """Content hash of a decoded tile, for duplicate detection before encoding."""
    import hashlib

    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()


__all__ = [
    "Box",
    "QualitySignals",
    "alignment_affine",
    "apply_affine",
    "crop_face",
    "heuristic_head_box",
    "image_sha1",
    "measure_quality",
    "quality_flags",
    "variance_of_laplacian",
]
