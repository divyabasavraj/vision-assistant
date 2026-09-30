"""
Purpose : Repeat-factor sampling (Gupta et al., LVIS 2019) for the train split: write a list of
          image paths in which images containing rare classes appear more than once, so v2 sees
          auto / chair / crosswalk / stairs more often without touching the val or test split.
Depends : standard library only. Ultralytics accepts a .txt of image paths as `train:` and keeps
          duplicates (get_img_files sorts, it does not de-duplicate).
Question you should be able to answer: why oversample images instead of boxes?  A detector
          trains on whole images, so the only way to show a rare box more often is to show its
          image more often — which also repeats the common boxes that share the image. That is
          why person/car counts rise a little too, and why the repeat factor is capped.
Usage   : python training/scripts/oversample.py training/data/merged --threshold 0.15
"""
import argparse
import math
import random
from collections import Counter
from pathlib import Path


def image_classes(label_dir: Path) -> dict[str, set[int]]:
    """label stem -> set of class ids present in that image."""
    out = {}
    for f in sorted(label_dir.glob("*.txt")):
        out[f.stem] = {int(line.split()[0]) for line in f.read_text().splitlines() if line.strip()}
    return out


def repeat_factors(classes: dict[str, set[int]], threshold: float, cap: float = 4.0) -> dict[str, float]:
    """LVIS rule: class factor r_c = max(1, sqrt(t / f_c)), f_c = fraction of images containing c;
    an image takes the largest factor of the classes in it, capped so no image dominates."""
    n = len(classes)
    freq = Counter(c for s in classes.values() for c in s)
    r_c = {c: max(1.0, math.sqrt(threshold / (k / n))) for c, k in freq.items()}
    return {stem: min(cap, max((r_c[c] for c in s), default=1.0)) for stem, s in classes.items()}


def copies(factors: dict[str, float], seed: int = 0) -> dict[str, int]:
    """Stochastic rounding: 1.4 becomes 2 copies 40 % of the time, else 1 — keeps the expected
    count right without every image of a 1.4-factor class being doubled."""
    rng = random.Random(seed)
    return {stem: int(r) + (rng.random() < r - int(r)) for stem, r in sorted(factors.items())}


def write_list(root: Path, threshold: float, dst: Path, seed: int = 0) -> Counter:
    """Write the oversampled train list for dataset `root`; returns instances per class after."""
    label_dir, image_dir = root / "labels" / "train", root / "images" / "train"
    images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    classes = image_classes(label_dir)
    n_copies = copies(repeat_factors(classes, threshold), seed)
    lines, inst = [], Counter()
    for stem, k in n_copies.items():
        lines += [str(images[stem].resolve())] * k
        for line in (label_dir / f"{stem}.txt").read_text().splitlines():
            if line.strip():
                inst[int(line.split()[0])] += k
    dst.write_text("\n".join(lines) + "\n")
    return inst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, help="dataset root containing images/ and labels/")
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    dst = args.out or args.root / f"train_rfs{args.threshold}.txt"
    inst = write_list(args.root, args.threshold, dst)
    print(f"wrote {dst}  ({sum(1 for _ in dst.open())} entries)")
    print({c: inst[c] for c in sorted(inst)})


if __name__ == "__main__":
    main()
