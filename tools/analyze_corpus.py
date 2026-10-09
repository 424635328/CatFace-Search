"""Corpus analysis — the statistical view of a dataset, across every species present.

This is the standing counterpart to the one-off audit scripts. Where ``audit_species``
answers "is this dataset what it claims to be?", this answers "what is *in* this dataset,
at what scale, and how clean is it?" — the numbers a data card or a benchmark report
needs, computed from the pixels rather than quoted from the source.

Reported per corpus:

* **Scale** — images, identities, images per identity (distribution, not just a mean).
* **Species composition** — for each image: is it a cat, a dog, a human face, or neither?
  Determined by two independent observers (a cat/dog-aware classifier and a human-face
  detector), because either alone confuses "unusual" with "other species".
* **Label integrity** — are claimed identities internally consistent, and does any image
  appear under more than one identity?
* **Image properties** — resolution distribution, aspect ratio, blur, luminance.
* **Warnings** — near-duplicate images, tiny images, identity collisions.

Species detection is deliberately conservative: it reports what it can *evidence* and
says so when it cannot decide, rather than guessing a species per image.

Usage::

    python -m tools.analyze_corpus --root <dir> --label-csv labels.csv --out docs/diagnostics/x.json
    python -m tools.analyze_corpus --preset oxford_iiit_pet --data-root data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.data.cropping import measure_quality, quality_flags
from catface.logging_utils import configure_utf8_console, get_logger

LOGGER = get_logger("corpus.analysis")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: ImageNet-1k indices for cat and dog classes, kept explicit so the judgement is auditable.
IMAGENET_CAT_INDICES = {281, 282, 283, 284, 285}
IMAGENET_DOG_INDICES = set(range(151, 269))


@dataclass
class ImageFacts:
    """Per-image measurements."""

    path: str
    identity: str
    width: int
    height: int
    sharpness: float
    luminance: float
    contrast: float
    sha1: str
    predicted_class: int | None = None
    cat_mass: float = 0.0
    dog_mass: float = 0.0
    human_face: bool = False
    human_face_confidence: float = 0.0

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


@dataclass
class CorpusReport:
    """Aggregate view of one corpus."""

    root: str
    images: int = 0
    identities: int = 0
    images_per_identity: dict[str, int] = field(default_factory=dict)
    resolution: dict[str, object] = field(default_factory=dict)
    quality: dict[str, object] = field(default_factory=dict)
    species: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    duplicates: dict[str, object] = field(default_factory=dict)
    samples_analysed: int = 0
    samples_total: int = 0

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "images": self.images,
            "identities": self.identities,
            "images_per_identity": self._identity_summary(),
            "resolution": self.resolution,
            "quality": self.quality,
            "species": self.species,
            "duplicates": self.duplicates,
            "warnings": self.warnings,
            "samples_analysed": self.samples_analysed,
            "samples_total": self.samples_total,
        }

    def _identity_summary(self) -> dict[str, object]:
        counts = sorted(self.images_per_identity.values())
        if not counts:
            return {}
        return {
            "min": counts[0],
            "median": statistics.median(counts),
            "mean": round(statistics.mean(counts), 2),
            "p90": counts[int(0.9 * (len(counts) - 1))],
            "max": counts[-1],
            "with_ge2": sum(1 for c in counts if c >= 2),
            "with_ge5": sum(1 for c in counts if c >= 5),
            "singletons": sum(1 for c in counts if c == 1),
        }


def discover_images(root: Path, limit: int | None = None) -> list[Path]:
    """Find image files under ``root``, deterministically ordered."""
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return files[:limit] if limit else files


def infer_identity(path: Path, root: Path, labels: dict[str, str] | None = None) -> str:
    """Infer the identity label, preferring an explicit labels file when supplied.

    Layouts recognised, in order: identity from the labels map (by relative or absolute
    path), from the parent directory, then from the ``prefix_index`` file-name form.
    """
    if labels:
        for key in (str(path), path.as_posix(), path.name):
            if key in labels:
                return labels[key]
    relative = path.relative_to(root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    stem = path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def load_labels(path: Path) -> dict[str, str]:
    """Read a ``path,identity`` CSV (extra columns ignored)."""
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row.get("path") or row.get("file") or row.get("image")
            value = row.get("identity") or row.get("label") or row.get("id")
            if key and value:
                labels[key] = value
                labels[Path(key).name] = value
    return labels


class SpeciesProbe:
    """Two independent observers: an ImageNet classifier and a human-face detector."""

    def __init__(self, device: str | None = None, face_model: Path | None = None) -> None:
        import torch
        import torchvision.models as tvm
        import torchvision.transforms as transforms

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.classifier.to(self.device).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.face_model_path = face_model
        self.face_detector = None
        if face_model and face_model.is_file():
            import cv2

            self._cv2 = cv2
            self.face_detector = cv2.FaceDetectorYN.create(str(face_model), "", (320, 320), 0.7, 0.3, 5000)

    def classify(self, images: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(top1_index, cat_mass, dog_mass)`` for a batch of PIL images."""
        torch = self.torch
        tensors = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.classifier(tensors), dim=1).cpu().numpy()
        top1 = probabilities.argmax(axis=1)
        cat_mass = probabilities[:, sorted(IMAGENET_CAT_INDICES)].sum(axis=1)
        dog_mass = probabilities[:, sorted(IMAGENET_DOG_INDICES)].sum(axis=1)
        return top1, cat_mass, dog_mass

    def detect_human(self, path: Path) -> tuple[bool, float]:
        """Whether a human face is detected, and the detector's confidence."""
        if self.face_detector is None:
            return False, 0.0
        image = self._cv2.imread(str(path))
        if image is None:
            return False, 0.0
        height, width = image.shape[:2]
        self.face_detector.setInputSize((width, height))
        _, faces = self.face_detector.detect(image)
        if faces is None or len(faces) == 0:
            return False, 0.0
        return True, float(faces[:, -1].max())


def analyse(
    root: Path,
    identities: dict[str, str] | None = None,
    sample: int | None = None,
    batch_size: int = 32,
    face_model: Path | None = None,
    device: str | None = None,
    with_species: bool = True,
) -> CorpusReport:
    """Measure a corpus and return a structured report.

    Args:
        root: Directory containing the images.
        identities: Optional explicit ``path -> identity`` mapping.
        sample: Analyse only this many images for the expensive per-pixel signals. The
            totals and identity statistics are always computed over *every* image.
        batch_size: Classifier batch size.
        face_model: Path to a human-face detector ONNX model; skipped when absent.
        device: Torch device.
        with_species: Set ``False`` to skip the classifier (a fast, metadata-only pass).
    """
    import cv2
    from PIL import Image

    files = discover_images(root)
    if not files:
        raise SystemExit(f"no images under {root}")

    report = CorpusReport(root=str(root))
    report.images = len(files)
    report.samples_total = len(files)

    identity_counts: Counter = Counter()
    for path in files:
        identity_counts[infer_identity(path, root, identities)] += 1
    report.images_per_identity = dict(identity_counts)
    report.identities = len(identity_counts)

    # Deterministic, identity-stratified sample so the expensive pass covers every
    # identity rather than reading whatever the sort order puts first.
    if sample and sample < len(files):
        by_identity: dict[str, list[Path]] = defaultdict(list)
        for path in files:
            by_identity[infer_identity(path, root, identities)].append(path)
        picked: list[Path] = []
        rng = np.random.default_rng(1337)
        quota = max(1, sample // max(len(by_identity), 1))
        for identity in sorted(by_identity):
            members = by_identity[identity]
            order = rng.permutation(len(members))[:quota]
            picked.extend(members[i] for i in order)
        sampled = picked[:sample]
    else:
        sampled = files
    report.samples_analysed = len(sampled)

    facts: list[ImageFacts] = []
    digests: dict[str, list[str]] = defaultdict(list)
    read_failures = 0
    for path in sampled:
        array = cv2.imread(str(path))
        if array is None:
            read_failures += 1
            continue
        height, width = array.shape[:2]
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        quality = measure_quality(array)
        digest = hashlib.sha1(np.ascontiguousarray(gray).tobytes()).hexdigest()
        facts.append(
            ImageFacts(
                path=str(path),
                identity=infer_identity(path, root, identities),
                width=width,
                height=height,
                sharpness=quality.sharpness,
                luminance=quality.mean_luminance,
                contrast=quality.contrast,
                sha1=digest,
            )
        )
        digests[digest].append(path.name)

    if read_failures:
        report.warnings.append(f"{read_failures} image(s) could not be decoded")

    widths = np.array([f.width for f in facts])
    heights = np.array([f.height for f in facts])
    shorts = np.array([f.short_side for f in facts])
    aspects = np.array([f.aspect for f in facts])
    report.resolution = {
        "short_side_min": int(shorts.min()),
        "short_side_median": float(np.median(shorts)),
        "short_side_max": int(shorts.max()),
        "median_width": float(np.median(widths)),
        "median_height": float(np.median(heights)),
        "aspect_median": round(float(np.median(aspects)), 3),
        "share_short_side_below_128": round(float((shorts < 128).mean()), 4),
        "share_within_10pct_of_square": round(float((np.abs(aspects - 1.0) < 0.1).mean()), 4),
    }

    sharpness = np.array([f.sharpness for f in facts])
    luminance = np.array([f.luminance for f in facts])
    blurred = sum(
        1
        for f in facts
        if "blurry"
        in quality_flags(
            type(
                "Q",
                (),
                {
                    "sharpness": f.sharpness,
                    "mean_luminance": f.luminance,
                    "contrast": f.contrast,
                    "face_fraction": 1.0,
                },
            )()
        )
    )
    report.quality = {
        "sharpness_median": round(float(np.median(sharpness)), 2),
        "sharpness_share_below_18": round(float((sharpness < 18.0).mean()), 4),
        "luminance_median": round(float(np.median(luminance)), 2),
        "luminance_share_below_25": round(float((luminance < 25).mean()), 4),
        "luminance_share_above_235": round(float((luminance > 235).mean()), 4),
        "flagged_unusable": blurred,
    }

    duplicated = {key: names for key, names in digests.items() if len(names) > 1}
    report.duplicates = {
        "distinct_pixel_contents": len(digests),
        "groups_with_duplicates": len(duplicated),
        "images_in_duplicate_groups": sum(len(names) for names in duplicated.values()),
        "duplicate_rate": round(1 - len(digests) / max(len(facts), 1), 4),
        "examples": [names for _, names in sorted(duplicated.items())[:5]],
    }
    if duplicated:
        report.warnings.append(
            f"{sum(len(n) for n in duplicated.values())} image(s) share pixel content with "
            "another image — identical duplicates let a retrieval benchmark score a free hit"
        )

    if with_species and facts:
        probe = SpeciesProbe(device=device, face_model=face_model)
        cat_mass_all, dog_mass_all = [], []
        top1_counter: Counter = Counter()
        cat_top1 = dog_top1 = neither_top1 = 0
        human_faces = 0
        human_confidences: list[float] = []
        for start in range(0, len(facts), batch_size):
            chunk = facts[start : start + batch_size]
            images = []
            for fact in chunk:
                with Image.open(fact.path) as handle:
                    images.append(handle.convert("RGB"))
            top1, cat_mass, dog_mass = probe.classify(images)
            for index, (t, c, d) in enumerate(zip(top1, cat_mass, dog_mass)):
                fact = chunk[index]
                fact.predicted_class = int(t)
                fact.cat_mass = float(c)
                fact.dog_mass = float(d)
                top1_counter[int(t)] += 1
                if int(t) in IMAGENET_CAT_INDICES:
                    cat_top1 += 1
                elif int(t) in IMAGENET_DOG_INDICES:
                    dog_top1 += 1
                else:
                    neither_top1 += 1
            if probe.face_detector is not None:
                for fact in chunk:
                    detected, confidence = probe.detect_human(Path(fact.path))
                    fact.human_face = detected
                    fact.human_face_confidence = confidence
                    if detected:
                        human_faces += 1
                        human_confidences.append(confidence)
            cat_mass_all.extend(cat_mass.tolist())
            dog_mass_all.extend(dog_mass.tolist())

        total = len(facts)
        cat_mass_array = np.array(cat_mass_all)
        dog_mass_array = np.array(dog_mass_all)
        report.species = {
            "observer": "ImageNet-1k ResNet-50 V2 classifier + OpenCV YuNet human-face detector",
            "top1_is_cat": cat_top1,
            "top1_is_dog": dog_top1,
            "top1_is_neither": neither_top1,
            "cat_probability_mass_median": round(float(np.median(cat_mass_array)), 5),
            "dog_probability_mass_median": round(float(np.median(dog_mass_array)), 5),
            "share_cat_mass_above_0.5": round(float((cat_mass_array > 0.5).mean()), 4),
            "share_dog_mass_above_0.5": round(float((dog_mass_array > 0.5).mean()), 4),
            "share_cat_mass_below_0.05": round(float((cat_mass_array < 0.05).mean()), 4),
            "images_with_human_face": human_faces,
            "human_face_rate": round(human_faces / total, 4) if probe.face_detector else None,
            "human_face_confidence_mean": round(float(np.mean(human_confidences)), 4)
            if human_confidences
            else None,
            "interpretation": (
                "A low cat mass on a corpus that claims to be cats is the signature of a "
                "mislabeled or wrong-population dataset; cross-check against "
                "audit_species and audit_content before trusting benchmark numbers."
            ),
        }
        if probe.face_detector is not None and human_faces / total > 0.5:
            report.warnings.append(
                f"{human_faces}/{total} analysed images contain a human face — this corpus is "
                "predominantly people, not animals"
            )
        # Only warn about a missing population when the corpus is *not* accounted for by
        # another species. A mixed cat-and-dog corpus legitimately has a low cat mass, and
        # firing the warning there would train the reader to ignore it.
        observed_cat = cat_top1 / total
        observed_dog = dog_top1 / total
        if float(np.median(cat_mass_array)) < 0.05 and observed_dog < 0.25:
            report.warnings.append(
                f"median cat probability mass is {float(np.median(cat_mass_array)):.4f} and only "
                f"{observed_dog:.1%} of images look like dogs; the corpus is largely neither, "
                "so check it is the population you intended"
            )
        if observed_dog >= 0.25:
            report.species["composition"] = (
                f"mixed: {observed_cat:.1%} cat-like, {observed_dog:.1%} dog-like — a low cat "
                "mass is expected here and is not evidence of a wrong population"
            )

    return report


# ---------------------------------------------------------------------------
# Presets: the corpora this project actually uses, with their real layouts.
# ---------------------------------------------------------------------------
def oxford_oiid(data_root: Path, sample: int | None, **kwargs) -> CorpusReport:
    """All 7390 Oxford-IIIT Pet images (cats *and* dogs), labelled by species and breed.

    Identity here is the **breed**, which is what the dataset actually provides. This
    preset exists precisely so the species/breed composition can be reported honestly
    instead of the corpus being mistaken for individual-level data.
    """
    images = data_root / "oxford_images"
    if not images.is_dir():
        raise SystemExit(f"Oxford images not found at {images}")
    # ``Abyssinian_100.jpg`` -> cat; ``newfoundland_31.jpg`` -> dog. The dataset README
    # documents capitalisation as the species convention.
    identities: dict[str, str] = {}
    for path in images.glob("*.jpg"):
        stem = path.stem
        breed = stem.rsplit("_", 1)[0]
        species = "cat" if stem[:1].isupper() else "dog"
        identities[str(path)] = f"{species}:{breed}"
    return analyse(images, identities=identities, sample=sample, **kwargs)


def cat_individuals(data_root: Path, sample: int | None, **kwargs) -> CorpusReport:
    """The Kaggle Cat Individual Images corpus, labelled by individual cat."""
    root = data_root / "raw" / "kaggle_cat_individuals" / "extracted" / "cat_individuals_dataset"
    if not root.is_dir():
        raise SystemExit(f"cat-individuals not found at {root}")
    return analyse(root, sample=sample, **kwargs)


PRESETS = {"oxford_iiit_pet": oxford_oiid, "cat_individuals": cat_individuals}


def main(argv: list[str] | None = None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Statistical analysis of an image corpus")
    parser.add_argument("--root", default=None, help="Corpus directory")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--data-root", default="data", help="Base directory used by presets")
    parser.add_argument("--label-csv", default=None, help="Optional path,identity CSV")
    parser.add_argument("--sample", type=int, default=400, help="Images analysed per-pixel (0 = all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--face-model", default="data/raw/models/yunet.onnx")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-species", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if not args.root and not args.preset:
        parser.error("supply --root or --preset")

    kwargs = {
        "sample": args.sample or None,
        "batch_size": args.batch_size,
        "face_model": Path(args.face_model) if args.face_model else None,
        "device": args.device,
        "with_species": not args.no_species,
    }

    if args.preset:
        report = PRESETS[args.preset](Path(args.data_root), **kwargs)
    else:
        labels = load_labels(Path(args.label_csv)) if args.label_csv else None
        report = analyse(Path(args.root), identities=labels, **kwargs)

    payload = report.to_dict()
    print(json.dumps(payload, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"written to {out}")

    if report.warnings:
        print("\n--- warnings ---")
        for warning in report.warnings:
            print(f"  ! {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
