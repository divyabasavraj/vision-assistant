"""
Purpose : Train + test-evaluate one model configuration and save its numbers as JSON, so the
          Task 9 ablations (image size, model size, augmentation) are one command each and
          every arm is measured exactly the same way.
Depends : ultralytics, torch. Runs on Apple-Silicon GPU (device="mps") or CUDA/CPU.
Question you should be able to answer: why must every ablation arm share the same epoch budget,
          seed and evaluation split?  Because only one thing may change per arm; otherwise the
          difference in mAP cannot be attributed to the thing you changed.
Usage   : training/.venv/bin/python training/scripts/run_ablations.py --name n640_v2 --epochs 30 \
              --base training/results/n640_v1/best.pt
          training/.venv/bin/python training/scripts/run_ablations.py --name n320_v2 --imgsz 320
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "training" / "data" / "merged" / "data.yaml"
RUNS = ROOT / "training" / "runs"

NO_AUG = dict(mosaic=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, scale=0.0, translate=0.0, erasing=0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name, e.g. n640_v2")
    p.add_argument("--base", default="yolo11n.pt", help="starting weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="mps")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--close-mosaic", type=int, default=None,
                   help="epochs at the end with mosaic off; default keeps v1's ratio of 20%% of epochs")
    p.add_argument("--no-aug", action="store_true", help="ablation C: turn augmentation off")
    args = p.parse_args()

    from ultralytics import YOLO  # imported late so --help is instant

    extra = dict(NO_AUG) if args.no_aug else {}
    # v1 used close_mosaic=10 of 50 epochs. Keep that 20%% ratio at every budget, otherwise a
    # short run would spend almost all of its epochs with mosaic already switched off and the
    # augmentation ablation would compare two nearly-unaugmented arms.
    close_mosaic = args.close_mosaic if args.close_mosaic is not None else max(1, round(args.epochs * 0.2))
    RUNS.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = YOLO(args.base)
    model.train(
        data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=10, seed=0, device=args.device, workers=args.workers, close_mosaic=close_mosaic,
        project=str(RUNS), name=args.name, exist_ok=True, plots=True, **extra,
    )
    train_min = (time.time() - t0) / 60

    best = YOLO(RUNS / args.name / "weights" / "best.pt")
    m = best.val(data=str(DATA), split="test", imgsz=args.imgsz, device=args.device,
                 project=str(RUNS), name=f"{args.name}_test", exist_ok=True)
    classes = [best.names[i] for i in range(len(best.names))]
    # ap50 / ap are indexed by the classes that actually appear in the split; maps is length-nc.
    present = list(getattr(m, "ap_class_index", getattr(m.box, "ap_class_index", range(len(classes)))))
    out = {
        "run": args.name, "base": args.base, "imgsz": args.imgsz, "epochs": args.epochs,
        "batch": args.batch, "augmentation": not args.no_aug, "device": args.device,
        "close_mosaic": close_mosaic,
        "train_minutes": round(train_min, 1),
        "params": sum(q.numel() for q in best.model.parameters()),
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "per_class_map50": {classes[int(c)]: float(a) for c, a in zip(present, m.box.ap50)},
        "per_class_map50_95": {classes[int(c)]: float(m.box.maps[int(c)]) for c in present},
        "speed_ms": m.speed,
    }
    dst = RUNS / f"{args.name}_test.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"{args.name}  mAP50 {out['map50']:.3f}  mAP50-95 {out['map50_95']:.3f}  ({train_min:.0f} min)")
    print("wrote", dst)


if __name__ == "__main__":
    main()
