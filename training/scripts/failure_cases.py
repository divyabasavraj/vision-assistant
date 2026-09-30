"""
Purpose : Find the shipped model's worst mistakes on the test split, one per failure type, and draw
          them (ground truth green, predictions red) for the report (spec §7 item 5).
            missed_close_curb     a curb at your feet, not detected
            missed_close_stairs   stairs close up, not detected
            missed_close_person   a person close up, not detected
            chair_as_person       a "person" box on a labelled chair
            wrong_class           a confident box of the wrong class on a labelled object
            phantom               a confident box over nothing labelled
          Picks are deterministic (largest missed box / most confident false box), not hand-chosen.
Depends : ultralytics, PIL; reuses the matching and band rules in evaluate.py (the app's thresholds).
Question you should be able to answer: why pick by a rule instead of browsing for bad examples?
          Browsing finds the funniest mistakes; a rule finds the ones that matter most for the user.
Usage   : training/.venv/bin/python training/scripts/failure_cases.py --weights training/results/n640_v1/best.pt
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

from evaluate import CLASS_THRESHOLD, DATA, NAMES, ROOT, SCORE_THRESHOLD, band_of, collect, iou

OUT = ROOT / "training/results/failures"
THRESH = [CLASS_THRESHOLD.get(n, SCORE_THRESHOLD) for n in NAMES]
C = {n: i for i, n in enumerate(NAMES)}


def analyse(preds, gts):
    """Same greedy match as evaluate.match, but keeps the boxes: (kept preds, claimed gts, false preds)."""
    kept = [p for p in preds if p[1] >= THRESH[p[0]]]
    claimed, false = set(), []
    for c, s, box in kept:
        best, best_iou = -1, 0.5
        for j, (gc, gbox, _) in enumerate(gts):
            if gc == c and j not in claimed and iou(box, gbox) >= best_iou:
                best, best_iou = j, iou(box, gbox)
        if best >= 0:
            claimed.add(best)
        else:
            false.append((c, s, box))
    return kept, claimed, false


def candidates(images):
    """Every image gets a severity score per failure type; the highest wins."""
    best = {}

    def offer(kind, sev, i, note):
        if sev > best.get(kind, (-1,))[0]:
            best[kind] = (sev, i, note)

    for i, (preds, gts) in enumerate(images):
        _, claimed, false = analyse(preds, gts)
        for j, (gc, _, hf) in enumerate(gts):
            if j in claimed or band_of(gc, hf) != "close":
                continue
            name = NAMES[gc]
            kind = {"curb": "missed_close_curb", "stairs_up": "missed_close_stairs",
                    "stairs_down": "missed_close_stairs", "person": "missed_close_person"}.get(name)
            if kind:
                offer(kind, hf, i, f"missed {name}, box height {hf:.2f} of frame")
        for c, s, box in false:
            over = [(iou(box, gb), gc) for gc, gb, _ in gts]
            top_iou, top_c = max(over) if over else (0.0, -1)
            if c == C["person"] and top_c == C["chair"] and top_iou >= 0.5:
                offer("chair_as_person", s, i, f"chair called person at {s:.2f}")
            elif top_iou >= 0.5 and top_c != c and not (c == C["person"] and top_c == C["chair"]):
                offer("wrong_class", s, i, f"{NAMES[top_c]} called {NAMES[c]} at {s:.2f}")
            elif top_iou < 0.1:
                offer("phantom", s, i, f"{NAMES[c]} at {s:.2f} over nothing labelled")
    return best


def draw(img_path, preds, gts, out):
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 250)
    for gc, box, _ in gts:
        d.rectangle(box, outline=(40, 200, 60), width=lw)
        d.text((box[0] + 3, box[3] - 14), NAMES[gc], fill=(40, 200, 60))
    for c, s, box in preds:
        d.rectangle(box, outline=(230, 40, 40), width=lw)
        d.text((box[0] + 3, box[1] + 2), f"{NAMES[c]} {s:.2f}", fill=(230, 40, 40))
    im.thumbnail((800, 800))
    im.save(out, quality=85)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    images = collect(YOLO(str(a.weights)), a.device, 640)
    paths = sorted((DATA / "images/test").iterdir())
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for kind, (sev, i, note) in sorted(candidates(images).items()):
        preds, gts = images[i]
        kept, _, _ = analyse(preds, gts)
        draw(paths[i], kept, gts, OUT / f"{kind}.jpg")
        summary[kind] = {"image": paths[i].name, "note": note}
        print(f"{kind:22s} {paths[i].name}  {note}")
    (OUT / "failures.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
