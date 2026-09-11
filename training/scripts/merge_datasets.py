"""
Purpose : Turn 2-3 downloaded Roboflow datasets (YOLO format) into ONE clean dataset.
          - translate each dataset's class names into our TARGET_CLASSES using mappings/<dataset>.json
          - drop labels for classes we do not want
          - drop exact duplicate images (same bytes)
          - split 80/10/10 with a fixed seed, write data.yaml
Depends : pyyaml. No ultralytics needed.
Question you should be able to answer: why must all datasets share ONE class numbering before training?
Usage   : python training/scripts/merge_datasets.py --raw training/data/raw --mappings training/mappings --out training/data/merged
"""
from __future__ import annotations
import argparse, hashlib, json, random, shutil
from pathlib import Path
import yaml

TARGET_CLASSES = [
    "person", "dog",
    "car", "bicycle", "motorbike", "auto", "bus",
    "stairs_up", "stairs_down", "curb",
    "pole", "chair", "table",
    "door", "crosswalk",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def load_source_names(source_dir: Path) -> list[str]:
    """Read the class list from a Roboflow data.yaml (list or {id: name} dict)."""
    y = yaml.safe_load((source_dir / "data.yaml").read_text())
    names = y["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def remap_label_line(line: str, source_names: list[str], mapping: dict[str, str]) -> str | None:
    """'<src_id> cx cy w h' -> '<our_id> cx cy w h', or None if the class is unmapped/invalid."""
    parts = line.split()
    if len(parts) < 5:
        return None
    src_id = int(parts[0])
    if src_id < 0 or src_id >= len(source_names):
        return None
    target = mapping.get(source_names[src_id])
    if target is None or target not in TARGET_CLASSES:
        return None
    return " ".join([str(TARGET_CLASSES.index(target)), *parts[1:5]])


def remap_label_text(text: str, source_names: list[str], mapping: dict[str, str]) -> list[str]:
    out = []
    for line in text.splitlines():
        r = remap_label_line(line, source_names, mapping)
        if r:
            out.append(r)
    return out


def split_name(index: int, n: int, seed: int) -> str:
    """Deterministic split: shuffle indices once with the seed, then slice 80/10/10."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    pos = order.index(index)
    if pos < int(n * SPLIT_FRACTIONS["train"]):
        return "train"
    if pos < int(n * (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"])):
        return "val"
    return "test"


def iter_pairs(source_dir: Path):
    """Yield (image_path, label_path) for every image in train/valid/test of a Roboflow export."""
    for split in ("train", "valid", "val", "test"):
        img_dir = source_dir / split / "images"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                yield img, source_dir / split / "labels" / (img.stem + ".txt")


def merge(raw_dir: Path, mappings_dir: Path, out_dir: Path, seed: int = 42) -> dict:
    """Return {split: {class_name: instance_count}}."""
    items = []  # (new_name, image_path, remapped_label_lines)
    seen_hashes: set[str] = set()
    for source_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        mapping_file = mappings_dir / f"{source_dir.name}.json"
        if not mapping_file.exists():
            print(f"skip {source_dir.name}: no mapping file {mapping_file}")
            continue
        mapping = json.loads(mapping_file.read_text())
        names = load_source_names(source_dir)
        for img, lbl in iter_pairs(source_dir):
            digest = hashlib.md5(img.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue
            lines = remap_label_text(lbl.read_text(), names, mapping) if lbl.exists() else []
            if not lines:
                continue
            seen_hashes.add(digest)
            items.append((f"{source_dir.name}_{img.name}", img, lines))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts = {s: {} for s in SPLIT_FRACTIONS}
    for s in SPLIT_FRACTIONS:
        (out_dir / "images" / s).mkdir(parents=True)
        (out_dir / "labels" / s).mkdir(parents=True)
    for i, (new_name, img, lines) in enumerate(items):
        s = split_name(i, len(items), seed)
        shutil.copy2(img, out_dir / "images" / s / new_name)
        (out_dir / "labels" / s / (Path(new_name).stem + ".txt")).write_text("\n".join(lines) + "\n")
        for line in lines:
            cls = TARGET_CLASSES[int(line.split()[0])]
            counts[s][cls] = counts[s].get(cls, 0) + 1

    (out_dir / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": len(TARGET_CLASSES), "names": TARGET_CLASSES,
    }, sort_keys=False))
    (out_dir / "counts.json").write_text(json.dumps(counts, indent=2))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="training/data/raw")
    ap.add_argument("--mappings", default="training/mappings")
    ap.add_argument("--out", default="training/data/merged")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    counts = merge(Path(a.raw), Path(a.mappings), Path(a.out), a.seed)
    for s, c in counts.items():
        print(s, sum(c.values()), "instances:", dict(sorted(c.items())))


if __name__ == "__main__":
    main()
