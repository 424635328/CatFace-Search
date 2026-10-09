"""Is the Oxford-IIIT per-image number an individual identifier?

The question decides whether 1 157 cropped cat faces currently sitting unused in ``data/faces`` are
trainable identity data or merely 1 157 samples of 12 breed classes. ``parse_identity`` collapses
``Abyssinian_100`` to ``Abyssinian``, which is right if the number is a file index and wrong if it
identifies a cat.

The Oxford-IIIT Pet dataset documents its naming only as "category_N", so the answer is measured
rather than assumed: embed the crops with this project's own trained model and compare descriptor
similarity for same-breed pairs whose numbers are close against pairs whose numbers are far apart. If
consecutive numbers are the same cat, close pairs will look like the same individual; if the number is
just a file index, both groups look like unrelated cats.

The control is a pair of *known* different cats from the Kaggle corpus, so "high" and "low" similarity
have a scale.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catface.models.embedder import Embedder, embed_records

REPO = Path(__file__).resolve().parents[1]
FACES = REPO / "data" / "faces"
CHECKPOINT = REPO / "artifacts" / "train" / "dinov2s-arcface" / "best.pt"

PATTERN = re.compile(r"^oiid_cat__(?P<breed>[A-Za-z_]+)_(?P<number>\d+)\.jpg$")


def collect() -> dict[str, list[tuple[int, Path]]]:
    by_breed: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(FACES.iterdir()):
        match = PATTERN.match(path.name)
        if match:
            by_breed[match.group("breed")].append((int(match.group("number")), path))
    return by_breed


def main() -> int:
    by_breed = collect()
    total = sum(len(v) for v in by_breed.values())
    print(f"Oxford cat crops found : {total} across {len(by_breed)} breeds")
    if total == 0:
        print("nothing to measure")
        return 1

    # Sample pairs from a few breeds, keeping the embedding job small and the question sharp.
    close_pairs: list[tuple[Path, Path]] = []
    far_pairs: list[tuple[Path, Path]] = []
    for entries in by_breed.values():
        entries.sort()
        for index in range(0, min(len(entries) - 1, 8)):
            close_pairs.append((entries[index][1], entries[index + 1][1]))
        for index in range(min(len(entries) - 1, 8)):
            partner = index + len(entries) // 2
            if partner < len(entries):
                far_pairs.append((entries[index][1], entries[partner][1]))

    print(f"same-breed adjacent-number pairs : {len(close_pairs)}")
    print(f"same-breed distant-number pairs  : {len(far_pairs)}")

    embedder = Embedder.load(CHECKPOINT, device="cuda")
    print(f"embedder: {embedder.config.backbone}")

    def similarities(pairs: list[tuple[Path, Path]], label: str) -> np.ndarray:
        left = embed_records(embedder, [str(a) for a, _ in pairs], batch_size=32)
        right = embed_records(embedder, [str(b) for _, b in pairs], batch_size=32)
        a = np.asarray(left.vectors, dtype=np.float32)
        b = np.asarray(right.vectors, dtype=np.float32)
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        scores = (a * b).sum(axis=1)
        print(
            f"  {label:<32} n={len(scores):<4} mean={scores.mean():.4f} "
            f"median={np.median(scores):.4f} p90={np.percentile(scores, 90):.4f}"
        )
        return scores

    print()
    print("descriptor similarity:")
    adjacent = similarities(close_pairs, "adjacent numbers (same breed)")
    distant = similarities(far_pairs, "distant numbers (same breed)")

    # Scale: two photographs of one known cat from the Kaggle corpus.
    kaggle = sorted(FACES.glob("cat_individuals__0490_0490_00*.jpg.jpg"))[:2]
    if len(kaggle) == 2:
        same_cat = similarities([(kaggle[0], kaggle[1])], "KNOWN same cat (control)")
        print(f"  -> control similarity {same_cat[0]:.4f}")

    print()
    gap = float(adjacent.mean() - distant.mean())
    print(f"adjacent minus distant : {gap:+.4f}")
    if gap > 0.05:
        print(
            "=> adjacent numbers look like the SAME individual: the number is an identity label, "
            "and collapsing it to the breed discards real identity data."
        )
    else:
        print(
            "=> adjacent numbers look like DIFFERENT individuals: the number is a file index, so "
            "collapsing it to the breed matches the data."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
