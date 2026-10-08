"""Test whether a corpus's images actually depict the thing the benchmark assumes.

A verification benchmark's numbers are meaningless if the images are not what the label
says they are. This script uses one off-the-shelf ImageNet classifier as an independent,
label-free observer and reports, per corpus:

* the top-1 predicted class and its probability,
* the probability mass assigned to cat-related classes,
* how many images are confidently *not* cats.

It is deliberately a coarse instrument: the question is "are these cat faces?", where a
confident non-cat verdict is meaningful and an inconclusive one is not.

Usage::

    python -m tools.audit_content --csv data/calfw/pairs.csv --limit 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: ImageNet-1k class indices that correspond to cats. Kept explicit rather than
#: substring-matched so the audit's judgement can be reviewed.
CAT_CLASS_INDICES = {
    281: "tabby",
    282: "tiger_cat",
    283: "Persian_cat",
    284: "Siamese_cat",
    285: "Egyptian_cat",
}
DOG_CLASS_RANGE = range(151, 269)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit image content against ImageNet classes")
    parser.add_argument("--csv", required=True, help="Pairs CSV (path_a/path_b/label)")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    import torch
    import torchvision.models as tvm
    import torchvision.transforms as transforms
    from PIL import Image

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(args.csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[: args.limit]

    paths: list[str] = []
    for row in rows:
        paths.append(row["path_a"])
        paths.append(row["path_b"])

    images = []
    for path in paths:
        with Image.open(path) as handle:
            images.append(preprocess(handle.convert("RGB")))

    top1: list[int] = []
    top1_prob: list[float] = []
    cat_mass: list[float] = []
    with torch.no_grad():
        for start in range(0, len(images), 32):
            batch = torch.stack(images[start : start + 32]).to(device)
            probabilities = torch.softmax(model(batch), dim=1).cpu()
            cat_mass.extend(probabilities[:, list(CAT_CLASS_INDICES)].sum(dim=1).tolist())
            best = probabilities.max(dim=1)
            top1.extend(best.values.tolist() and best.indices.tolist())
            top1_prob.extend(best.values.tolist())

    top1 = np.array(top1)
    top1_prob = np.array(top1_prob)
    cat_mass = np.array(cat_mass)

    def describe(indices: np.ndarray) -> dict:
        cats = np.isin(indices, list(CAT_CLASS_INDICES))
        dogs = np.isin(indices, list(DOG_CLASS_RANGE))
        return {
            "count": int(indices.size),
            "image_is_top1_cat": int(cats.sum()),
            "image_is_top1_dog": int(dogs.sum()),
            "image_is_top1_neither": int((~cats & ~dogs).sum()),
            "top1_probability_mean": float(top1_prob.mean()),
        }

    report = {
        "csv": args.csv,
        "images_audited": len(paths),
        "cat_class_indices": CAT_CLASS_INDICES,
        "overall": describe(top1),
        "top1_histogram": {
            f"{index}:{CAT_CLASS_INDICES.get(index, 'other')}": count
            for index, count in Counter(top1.tolist()).most_common(12)
        },
        "cat_probability_mass": {
            "mean": float(cat_mass.mean()),
            "median": float(np.median(cat_mass)),
            "share_with_mass_above_0.5": float((cat_mass > 0.5).mean()),
            "share_with_mass_below_0.05": float((cat_mass < 0.05).mean()),
        },
        "verdict_hint": (
            "if share_with_mass_below_0.05 is high, the images are mostly NOT cat faces, "
            "which invalidates any cat-face benchmark built on them"
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
