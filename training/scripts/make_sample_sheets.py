"""
Purpose : Make one contact sheet per split (train/val/test) - 12 random images with their
          ground-truth boxes and class names drawn on - so you can *show* the dataset.
Depends : pillow, pyyaml (both in training/.venv).
Usage   : python training/scripts/make_sample_sheets.py                # seed 7, 4x3 tiles
          python training/scripts/make_sample_sheets.py --seed 42      # a different random draw
Output  : training/results/samples/{train,val,test}_samples.jpg  (open them in VS Code)
Question you should be able to answer: why draw the *label file*, not the model's output, here?
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load_font(size: int):
    for cand in ("/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_tile(img_path: Path, labels_dir: Path, names: list[str], tile: int, font, small) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, W // 250)
    lbl = labels_dir / (img_path.stem + ".txt")
    for line in lbl.read_text().splitlines() if lbl.exists() else []:
        c, cx, cy, w, h = line.split()[:5]
        c = int(c); cx, cy, w, h = map(float, (cx, cy, w, h))
        x0, y0, x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        col = PAL[c % len(PAL)]
        d.rectangle([x0, y0, x1, y1], outline=col, width=lw)
        t = names[c]; tw = d.textlength(t, font=font) + 8
        ty = max(0, y0 - 20)
        d.rectangle([x0, ty, x0 + tw, ty + 20], fill=col)
        d.text((x0 + 4, ty + 2), t, fill="white", font=font)
    im.thumbnail((tile, tile))
    canvas = Image.new("RGB", (tile, tile), "#1a1a19")
    canvas.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
    dd = ImageDraw.Draw(canvas)
    dd.rectangle([0, tile - 18, tile, tile], fill="#1a1a19")
    dd.text((6, tile - 16), f"{img_path.name.split('_')[0]} · {W}×{H}", fill="#c3c2b7", font=small)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data" / "merged", type=Path)
    ap.add_argument("--out", default=ROOT / "results" / "samples", type=Path)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--tile", type=int, default=320)
    a = ap.parse_args()

    names = yaml.safe_load((a.data / "data.yaml").read_text())["names"]
    font, small = load_font(15), load_font(12)
    a.out.mkdir(parents=True, exist_ok=True)
    gap = 6
    for split in ("train", "val", "test"):
        img_dir = a.data / "images" / split
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        random.Random(a.seed).shuffle(imgs)
        pick = imgs[: a.cols * a.rows]
        sheet = Image.new("RGB", (a.cols * a.tile + (a.cols + 1) * gap, a.rows * a.tile + (a.rows + 1) * gap), "#fcfcfb")
        for i, p in enumerate(pick):
            r, c = divmod(i, a.cols)
            sheet.paste(draw_tile(p, a.data / "labels" / split, names, a.tile, font, small),
                        (gap + c * (a.tile + gap), gap + r * (a.tile + gap)))
        dest = a.out / f"{split}_samples.jpg"
        sheet.save(dest, quality=86)
        print(f"{split}: {len(imgs)} images, {len(pick)} shown -> {dest}")


if __name__ == "__main__":
    main()
