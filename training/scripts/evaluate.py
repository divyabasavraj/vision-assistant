"""
Purpose : Every evaluation number for one set of weights, on the untouched 904-image test split,
          through one code path — so v1 and v2 are compared by identical arithmetic.
            1. standard detector metrics (Ultralytics val): mAP50, mAP50-95, P, R, per class
            2. the app's operating point: the thresholds shipped in lib/detector.ts
               (0.35, curb 0.5, person 0.4), per-class P / R / F1, and recall in the narrator's
               far / near / close bands (bucketOf() in lib/narrator.ts)
            3. a per-class threshold sweep, to re-check the shipped thresholds for a new model
            4. deployment cost: ONNX size and CPU latency (onnxruntime, 2 threads)
Depends : ultralytics, onnxruntime. Writes training/results/<name>_eval.json.
Question you should be able to answer: why report both flat recall and band recall?  Flat recall
          is the fair comparison with other detectors; band recall describes the product, which
          never speaks about far objects that are not straight ahead.
Usage   : training/.venv/bin/python training/scripts/evaluate.py --weights training/results/n640_v1/best.pt --name n640_v1
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "training/data/merged"
NAMES = ["person", "dog", "car", "bicycle", "motorbike", "auto", "bus", "stairs_up", "stairs_down",
         "curb", "pole", "chair", "table", "door", "crosswalk"]
# lib/detector.ts
SCORE_THRESHOLD, CLASS_THRESHOLD, NMS_IOU = 0.35, {"curb": 0.5, "person": 0.4}, 0.45
# lib/narrator.ts: far < 0.25*s <= near < 0.50*s <= close, s = the class's real-world height
NEAR, CLOSE = 0.25, 0.50
SIZE = {"person": 1.0, "dog": 0.35, "car": 0.9, "bicycle": 0.65, "motorbike": 0.75, "auto": 1.0,
        "bus": 1.8, "stairs_up": 0.6, "stairs_down": 0.6, "curb": 0.1, "pole": 1.5, "chair": 0.55,
        "table": 0.45, "door": 1.2, "crosswalk": 0.15}
SWEEP = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MATCH_IOU = 0.5
BANDS = ("far", "near", "close")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def band_of(cls: int, height_frac: float) -> str:
    s = SIZE[NAMES[cls]]
    return "far" if height_frac < NEAR * s else "near" if height_frac < CLOSE * s else "close"


def match(preds, gts, thresholds):
    """Greedy by score: each prediction above its class's bar claims the best unclaimed
    same-class ground truth at IoU >= 0.5. Returns (claimed gt indices, false-positive classes)."""
    claimed, fps = set(), []
    for c, score, box in preds:
        if score < thresholds[c]:
            continue
        best, best_iou = -1, MATCH_IOU
        for j, (gc, gbox, _) in enumerate(gts):
            if gc == c and j not in claimed:
                v = iou(box, gbox)
                if v >= best_iou:
                    best, best_iou = j, v
        if best >= 0:
            claimed.add(best)
        else:
            fps.append(c)
    return claimed, fps


def collect(model, device, imgsz):
    """One model pass at a low bar; every threshold is then evaluated offline. A box NMS keeps at
    a low threshold is still kept at a higher one, so this equals re-running per threshold."""
    out = []
    for img in sorted((DATA / "images/test").iterdir()):
        r = model.predict(str(img), conf=min(SWEEP), iou=NMS_IOU, imgsz=imgsz, verbose=False, device=device)[0]
        h, w = r.orig_shape
        gts = []
        for line in (DATA / "labels/test" / f"{img.stem}.txt").read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            gts.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h), bh))
        preds = sorted(((int(b.cls), float(b.conf), tuple(map(float, b.xyxy[0]))) for b in r.boxes),
                       key=lambda p: -p[1])
        out.append((preds, gts))
    return out


def score(images, thresholds):
    tp, fp, n = defaultdict(int), defaultdict(int), defaultdict(int)
    band_hit, band_n = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    for preds, gts in images:
        claimed, fps = match(preds, gts, thresholds)
        for c in fps:
            fp[c] += 1
        for j, (c, _, hf) in enumerate(gts):
            b = band_of(c, hf)
            n[c] += 1
            band_n[c][b] += 1
            if j in claimed:
                tp[c] += 1
                band_hit[c][b] += 1
    rows = {}
    for c, name in enumerate(NAMES):
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / n[c] if n[c] else 0.0
        rows[name] = dict(tp=tp[c], fp=fp[c], fn=n[c] - tp[c], n=n[c], precision=round(p, 3),
                          recall=round(r, 3), f1=round(2 * p * r / (p + r), 3) if p + r else 0.0,
                          **{f"recall_{b}": (round(band_hit[c][b] / band_n[c][b], 3) if band_n[c][b] else None)
                             for b in BANDS},
                          **{f"n_{b}": band_n[c][b] for b in BANDS})
    tot = {b: (sum(band_hit[c][b] for c in range(len(NAMES))), sum(band_n[c][b] for c in range(len(NAMES))))
           for b in BANDS}
    rows["ALL"] = dict(
        precision=round(sum(tp.values()) / max(1, sum(tp.values()) + sum(fp.values())), 3),
        recall=round(sum(tp.values()) / max(1, sum(n.values())), 3),
        **{f"recall_{b}": round(h / t, 3) for b, (h, t) in tot.items() if t},
        **{f"n_{b}": t for b, (_, t) in tot.items()})
    return rows


def onnx_cost(weights: Path, imgsz: int):
    import numpy as np
    import onnxruntime as ort
    from ultralytics import YOLO
    path = Path(YOLO(str(weights)).export(format="onnx", opset=17, imgsz=imgsz, simplify=True, dynamic=False))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(5):
        sess.run(None, {name: x})
    times = []
    for _ in range(50):
        t = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t) * 1000)
    return dict(onnx=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                onnx_mb=round(path.stat().st_size / 1e6, 1), cpu_ms_median=round(statistics.median(times), 1),
                output_shape=list(sess.get_outputs()[0].shape))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--no-onnx", action="store_true")
    args = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    assert [model.names[i] for i in range(len(model.names))] == NAMES, "class order differs from labels.json"

    m = model.val(data=str(DATA / "data.yaml"), split="test", imgsz=args.imgsz, device=args.device,
                  plots=False, verbose=False)
    idx = [int(c) for c in m.box.ap_class_index]
    standard = dict(map50=round(float(m.box.map50), 3), map50_95=round(float(m.box.map), 3),
                    precision=round(float(m.box.mp), 3), recall=round(float(m.box.mr), 3),
                    per_class={NAMES[c]: dict(precision=round(float(m.box.p[i]), 3), recall=round(float(m.box.r[i]), 3),
                                              ap50=round(float(m.box.ap50[i]), 3), ap50_95=round(float(m.box.ap[i]), 3))
                               for i, c in enumerate(idx)})

    images = collect(model, args.device, args.imgsz)
    shipped = [CLASS_THRESHOLD.get(n, SCORE_THRESHOLD) for n in NAMES]
    app = score(images, shipped)
    sweep = {}
    for t in SWEEP:
        rows = score(images, [t] * len(NAMES))
        sweep[str(t)] = {k: {f: v[f] for f in ("precision", "recall", "recall_near", "recall_close") if f in v}
                         for k, v in rows.items()}

    out = dict(name=args.name, weights=str(args.weights), imgsz=args.imgsz, test_images=len(images),
               params=sum(q.numel() for q in model.model.parameters()),
               standard=standard, app_operating_point=dict(thresholds=dict(zip(NAMES, shipped)), **{"per_class": app}),
               sweep=sweep)
    if not args.no_onnx:
        out["deployment"] = onnx_cost(args.weights, args.imgsz)
    dst = ROOT / "training/results" / f"{args.name}_eval.json"
    dst.write_text(json.dumps(out, indent=1))

    a = app["ALL"]
    print(f"{args.name}: mAP50 {standard['map50']}  mAP50-95 {standard['map50_95']}  "
          f"P {standard['precision']}  R {standard['recall']}")
    print(f"  app point: P {a['precision']}  R {a['recall']}  far {a.get('recall_far')}  "
          f"near {a.get('recall_near')}  close {a.get('recall_close')}")
    if "deployment" in out:
        print(f"  onnx {out['deployment']['onnx_mb']} MB, {out['deployment']['cpu_ms_median']} ms CPU")
    print("wrote", dst)


if __name__ == "__main__":
    main()
