"""One-off diagnostic: are Oxford's list.txt and xmls/ consistent?

Kept as a tool (not a test) because it documents a genuine data-source quirk that
future work needs to know about.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catface.data.annotation import parse_identity, parse_oiid_list, parse_oiid_xml

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/oxford_annotations")
rows = parse_oiid_list(root / "list.txt")
xmls = {p.stem: p for p in (root / "xmls").glob("*.xml")}

by_species: dict[str, list[str]] = {}
absent: list[str] = []
for image_id, _class_id, _declared, _breed in rows:
    path = xmls.get(image_id)
    if path is None:
        absent.append(image_id)
        continue
    name = str(parse_oiid_xml(path)["name"])
    by_species.setdefault(name, []).append(image_id)

print(f"list.txt rows          : {len(rows)}")
print(f"xml files              : {len(xmls)}")
print(f"rows with no XML       : {len(absent)}")
print(f"per object name        : { {k: len(v) for k, v in by_species.items()} }")

cats = by_species.get("cat", [])
dogs = by_species.get("dog", [])
print(f"cat rows               : {len(cats)}")
print(f"cat identities         : {len({parse_identity(i) for i in cats})}")
print(f"dog rows               : {len(dogs)}")
print(f"dog identities         : {len({parse_identity(i) for i in dogs})}")
print(f"cat files on disk      : {len(list((root.parent / 'oxford_images').glob('*.jpg')))}")
print(f"capitalised cat ids    : {sum(1 for i in cats if i[:1].isupper())}")
print(f"capitalised dog ids    : {sum(1 for i in dogs if i[:1].isupper())}")
print(f"sample cats            : {sorted(cats)[:5]}")
print(f"sample dogs            : {sorted(dogs)[:5]}")
ids_in_list = {r[0] for r in rows}
print(f"xml stems absent from list: {sum(1 for stem in xmls if stem not in ids_in_list)}")
print(f"sample absent-from-list: {[s for s in list(xmls)[:5] if s not in ids_in_list]}")
