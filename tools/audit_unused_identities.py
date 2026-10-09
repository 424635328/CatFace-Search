"""Are the seven unused raw identities recoverable as training data?

The manifest covers 503 of the 509 folders in the Kaggle corpus. If any of the seven omitted
identities were assigned to the training split, that identity contributed nothing to training while
occupying a slot — pure waste in the one dimension the evidence says is the bottleneck.

This matters because the previous rounds established, with measurements, that the retrieval stage and
any linear head are exhausted: the only remaining lever is more identity-labelled data. Before
building anything, the question is whether data is being left on the floor.
"""

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "manifests" / "cat_individuals_manifest.jsonl"
SPLITS = REPO / "data" / "manifests" / "cat_individuals_splits"
RAW = REPO / "data" / "raw" / "kaggle_cat_individuals" / "extracted" / "cat_individuals_dataset"
FACES = REPO / "data" / "faces"


def manifest_records() -> list[dict]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "_manifest_version" in record:
            continue
        rows.append(record)
    return rows


records = manifest_records()
used = {r["identity"] for r in records}
print(f"manifest            : {len(records)} images, {len(used)} identities")

image_split: dict[str, str] = {}
for name in ("train", "val", "test"):
    path = SPLITS / f"{name}.txt"
    if not path.is_file():
        print(f"  missing split file: {path}")
        continue
    for image_id in path.read_text(encoding="utf-8").splitlines():
        if image_id.strip():
            image_split[image_id.strip()] = name

identity_split: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
for record in records:
    split = image_split.get(record["image_id"])
    if split:
        identity_split[split].add(record["identity"])
print(
    f"split identities    : train={len(identity_split['train'])} "
    f"val={len(identity_split['val'])} test={len(identity_split['test'])}"
)

raw_folders = sorted(d.name for d in RAW.iterdir() if d.is_dir())
print(f"raw folders         : {len(raw_folders)}")
print(f"folders never in the manifest: {len([d for d in raw_folders if d not in used])}")

print()
print("the omitted identities, and whether they were counted as training identities:")
omitted = [name for name in raw_folders if name not in used]
for name in omitted:
    files = [f for f in (RAW / name).rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    crops = list(FACES.glob(f"cat_individuals__{name}_*"))
    where = [split for split, ids in identity_split.items() if name in ids]
    print(f"  {name}: raw={len(files):>4}  crops={len(crops):>4}  in split={where or 'nowhere'}")

print()
# The corpus is 509 folders; if the split assignment counted the omitted ones, training reserved
# identities that contributed no images.
total_accounted = len(identity_split["train"]) + len(identity_split["val"]) + len(identity_split["test"])
print(f"identities accounted for by the splits: {total_accounted} of {len(raw_folders)} raw folders")
print()
counts = Counter(r["identity"] for r in records)
print(
    f"images per identity in the corpus: min={min(counts.values())} "
    f"median={sorted(counts.values())[len(counts) // 2]} max={max(counts.values())}"
)
