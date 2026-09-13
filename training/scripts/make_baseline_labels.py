"""
Purpose : Build a copy of the TEST split whose labels use COCO class ids, so the original
          (not fine-tuned) yolo11n.pt can be scored on it with `model.val()`.
          Classes COCO does not have (stairs, curb, door, pole, crosswalk, auto) are dropped:
          the baseline scores 0 on them by definition — that is the point of the comparison.
Depends : pyyaml, ultralytics (only to read COCO names from yolo11n.pt in main()).
Question you should be able to answer: why can't we just run yolo11n.pt val on our data.yaml directly?
Usage   : python training/scripts/make_baseline_labels.py && yolo val model=yolo11n.pt data=training/data/baseline/data.yaml split=test
"""
from __future__ import annotations
import os, shutil, sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_datasets import TARGET_CLASSES

OURS_TO_COCO = {
    "person": "person", "dog": "dog", "car": "car", "bicycle": "bicycle",
    "motorbike": "motorcycle", "bus": "bus", "chair": "chair", "table": "dining table",
}


def remap_to_coco_line(line: str, our_names: list[str], coco_names: list[str]) -> str | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    ours = our_names[int(parts[0])]
    coco = OURS_TO_COCO.get(ours)
    if coco is None or coco not in coco_names:
        return None
    return " ".join([str(coco_names.index(coco)), *parts[1:5]])


def build(merged: Path, out: Path, coco_names: list[str]) -> int:
    if out.exists():
        shutil.rmtree(out)
    (out / "labels" / "test").mkdir(parents=True)
    # One symlink per image, NOT a symlinked directory: ultralytics resolve()s the split dir and
    # then swaps images->labels in the path, which would send it to merged/labels (wrong ids).
    (out / "images" / "test").mkdir(parents=True)
    for img in (merged / "images" / "test").iterdir():
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            os.symlink(img.resolve(), out / "images" / "test" / img.name)
    kept = 0
    for lbl in (merged / "labels" / "test").glob("*.txt"):
        lines = [r for r in (remap_to_coco_line(l, TARGET_CLASSES, coco_names) for l in lbl.read_text().splitlines()) if r]
        (out / "labels" / "test" / lbl.name).write_text("\n".join(lines) + ("\n" if lines else ""))
        kept += len(lines)
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out.resolve()), "train": "images/test", "val": "images/test", "test": "images/test",
        "nc": len(coco_names), "names": coco_names}, sort_keys=False))
    return kept


if __name__ == "__main__":
    from ultralytics import YOLO
    coco_names = list(YOLO("yolo11n.pt").names.values())
    n = build(Path("training/data/merged"), Path("training/data/baseline"), coco_names)
    print("baseline test labels written:", n)
