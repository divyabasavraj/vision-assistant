"""
Purpose : Per-class recall and precision of the deployed detector as the score threshold moves.
          Answers one question: does SCORE_THRESHOLD = 0.5 silence the weak classes?
Why     : curb, pole and crosswalk cannot be demonstrated live in a classroom, so the evidence
          for them has to be a measured table rather than a camera pointed at one example.
Run     : training/.venv/bin/python training/scripts/threshold_sweep.py
Note    : predictions are collected ONCE at the lowest threshold and filtered afterwards. A box
          kept by NMS at a low threshold is still kept at a higher one, so this is identical to
          re-running the model per threshold, and ~5x cheaper.
"""
import json
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "training/data/merged"
WEIGHTS = ROOT / "training/results/n640_v1/best.pt"
THRESHOLDS = [0.25, 0.35, 0.45, 0.50, 0.60]
MATCH_IOU = 0.5          # a prediction counts only if it overlaps the truth this much
NAMES = [n.strip("- ") for n in (DATA / "data.yaml").read_text().split("names:")[1].split("\n") if n.strip().startswith("-")]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(label_path, w, h):
    """YOLO txt -> [(cls, x1,y1,x2,y2)] in pixels."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def main():
    imgs = sorted((DATA / "images/test").iterdir())
    model = YOLO(str(WEIGHTS))
    print(f"{len(imgs)} test images, {len(NAMES)} classes, weights {WEIGHTS.name}")

    # cls -> threshold -> counts
    tp = defaultdict(lambda: defaultdict(int))
    fp = defaultdict(lambda: defaultdict(int))
    gt_total = defaultdict(int)

    for i, img in enumerate(imgs):
        if i % 200 == 0:
            print(f"  {i}/{len(imgs)}", flush=True)
        res = model.predict(str(img), conf=min(THRESHOLDS), iou=0.45, imgsz=640, verbose=False, device="cpu")[0]
        h, w = res.orig_shape
        gt = load_gt(DATA / "labels/test" / (img.stem + ".txt"), w, h)
        for c, *_ in gt:
            gt_total[c] += 1

        preds = [(int(b.cls), float(b.conf), *map(float, b.xyxy[0])) for b in res.boxes]
        preds.sort(key=lambda p: -p[1])

        for t in THRESHOLDS:
            claimed = set()
            for c, score, *box in preds:
                if score < t:
                    continue
                best, best_iou = -1, MATCH_IOU
                for j, (gc, *gbox) in enumerate(gt):
                    if gc != c or j in claimed:
                        continue
                    v = iou(box, gbox)
                    if v >= best_iou:
                        best, best_iou = j, v
                if best >= 0:
                    claimed.add(best)
                    tp[c][t] += 1
                else:
                    fp[c][t] += 1

    rows = []
    for c, name in enumerate(NAMES):
        n = gt_total[c]
        row = {"class": name, "instances": n}
        for t in THRESHOLDS:
            recall = tp[c][t] / n if n else 0.0
            denom = tp[c][t] + fp[c][t]
            row[f"r@{t}"] = round(recall, 3)
            row[f"p@{t}"] = round(tp[c][t] / denom, 3) if denom else 0.0
        rows.append(row)

    out = ROOT / "training/results/threshold_sweep.json"
    out.write_text(json.dumps({"thresholds": THRESHOLDS, "match_iou": MATCH_IOU, "rows": rows}, indent=2))

    hdr = f"{'class':<12}{'inst':>6}" + "".join(f"{'R@'+str(t):>9}" for t in THRESHOLDS)
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r[f"r@0.5"]):
        print(f"{r['class']:<12}{r['instances']:>6}" + "".join(f"{r[f'r@{t}']:>9.3f}" for t in THRESHOLDS))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
