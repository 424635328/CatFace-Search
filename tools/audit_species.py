"""Test whether a corpus is human faces or cat faces.

An image-only audit cannot answer this: an ImageNet classifier assigns low cat probability
to *both* an unusual cat face and a human face. A face *detector* can, because detectors are
species-specific — a human-face detector firing on most crops is close to conclusive.

This script runs OpenCV's YuNet human-face detector over a sample of a corpus and reports the
detection rate. Interpretation:

* detection rate near 1.0 → the images are human faces, and any "cat face" benchmark built
  on them is measuring the wrong population;
* detection rate near 0.0 → the images are not human faces (consistent with cats, though
  this alone does not prove they are cats).

Usage::

    python -m tools.audit_species --csv data/calfw/pairs.csv --limit 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def human_face_rate(
    paths: list[str],
    model_path: Path,
    score_threshold: float = 0.7,
) -> dict:
    """Fraction of images in which a human face is detected."""
    import cv2

    detector = cv2.FaceDetectorYN.create(
        str(model_path), "", (320, 320), score_threshold, 0.3, 5000
    )
    detected = 0
    probabilities: list[float] = []
    failures = 0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            failures += 1
            continue
        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
        if faces is not None and len(faces) > 0:
            detected += 1
            probabilities.append(float(faces[:, -1].max()))
        else:
            probabilities.append(0.0)
    total = len(paths) - failures
    return {
        "images": total,
        "unreadable": failures,
        "images_with_human_face": detected,
        "human_face_rate": round(detected / total, 4) if total else None,
        "mean_face_confidence": round(float(np.mean(probabilities)), 4) if probabilities else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Human-vs-animal face audit")
    parser.add_argument("--csv", default=None, help="Pairs CSV, or any CSV with a path column")
    parser.add_argument("--directory", default=None, help="Audit a directory of images instead")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--model", default="data/raw/models/yunet.onnx")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"face detector not found: {model_path}", file=sys.stderr)
        return 2
    if not args.directory and not args.csv:
        print("supply either --csv or --directory", file=sys.stderr)
        return 2

    if args.directory:
        root = Path(args.directory)
        paths = sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    else:
        with open(args.csv, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        paths = []
        for row in rows:
            for key in ("path_a", "path_b", "path"):
                if key in row:
                    paths.append(row[key])
                    break
    paths = paths[: args.limit]
    if not paths:
        print("no images selected", file=sys.stderr)
        return 2

    result = human_face_rate(paths, model_path)
    report = {
        "source": args.csv or args.directory,
        "sample_size": len(paths),
        "detector": "OpenCV YuNet (human faces)",
        "result": result,
        "verdict": (
            "HUMAN faces — do not use as a cat-face benchmark"
            if (result["human_face_rate"] or 0) > 0.5
            else "no evidence of human faces"
        ),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
