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
          On Kaggle: --device 0 --data /kaggle/working/merged/data.yaml --runs /kaggle/working/runs
          --oversample 0.15 trains on the repeat-factor list from oversample.py (val/test untouched).
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
    p.add_argument("--data", type=Path, default=DATA, help="dataset yaml")
    p.add_argument("--runs", type=Path, default=RUNS, help="output directory")
    p.add_argument("--oversample", type=float, default=None,
                   help="repeat-factor threshold t for the train split (oversample.py); off by default")
    # 'auto' ignores lr0 and picks by run length (>10k iterations SGD 0.01, else AdamW), which is
    # how v2 silently switched optimizer from v1. Continuing from trained weights should pin both.
    p.add_argument("--optimizer", default="auto", help="SGD / AdamW / auto (Ultralytics default)")
    p.add_argument("--lr0", type=float, default=None, help="initial learning rate; ignored under auto")
    p.add_argument("--warmup-epochs", type=float, default=None, help="default 3; 0 when fine-tuning on")
    args = p.parse_args()

    from ultralytics import YOLO  # imported late so --help is instant

    extra = dict(NO_AUG) if args.no_aug else {}
    extra["optimizer"] = args.optimizer
    if args.lr0 is not None:
        extra["lr0"] = args.lr0
    if args.warmup_epochs is not None:
        extra["warmup_epochs"] = args.warmup_epochs
    # v1 used close_mosaic=10 of 50 epochs. Keep that 20%% ratio at every budget, otherwise a
    # short run would spend almost all of its epochs with mosaic already switched off and the
    # augmentation ablation would compare two nearly-unaugmented arms.
    close_mosaic = args.close_mosaic if args.close_mosaic is not None else max(1, round(args.epochs * 0.2))
    runs, data = args.runs, args.data
    runs.mkdir(parents=True, exist_ok=True)
    train_data = data
    if args.oversample is not None:
        import yaml
        from oversample import write_list
        cfg = yaml.safe_load(data.read_text())
        root = Path(cfg.get("path") or data.parent)
        if not root.is_absolute():
            root = (data.parent / root).resolve()
        lst = runs / f"{args.name}_train_rfs.txt"
        write_list(root, args.oversample, lst)
        cfg.update(path=str(root), train=str(lst.resolve()))
        train_data = runs / f"{args.name}_data.yaml"
        train_data.write_text(yaml.safe_dump(cfg, sort_keys=False))

    t0 = time.time()
    model = YOLO(args.base)
    model.train(
        data=str(train_data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=10, seed=0, device=args.device, workers=args.workers, close_mosaic=close_mosaic,
        project=str(runs), name=args.name, exist_ok=True, plots=True, **extra,
    )
    train_min = (time.time() - t0) / 60

    best = YOLO(runs / args.name / "weights" / "best.pt")
    m = best.val(data=str(data), split="test", imgsz=args.imgsz, device=args.device,
                 project=str(runs), name=f"{args.name}_test", exist_ok=True)
    classes = [best.names[i] for i in range(len(best.names))]
    # ap50 / ap are indexed by the classes that actually appear in the split; maps is length-nc.
    present = list(getattr(m, "ap_class_index", getattr(m.box, "ap_class_index", range(len(classes)))))
    out = {
        "run": args.name, "base": args.base, "imgsz": args.imgsz, "epochs": args.epochs,
        "batch": args.batch, "augmentation": not args.no_aug, "device": args.device,
        "close_mosaic": close_mosaic, "oversample": args.oversample,
        "optimizer": args.optimizer, "lr0": args.lr0, "warmup_epochs": args.warmup_epochs,
        "train_minutes": round(train_min, 1),
        "params": sum(q.numel() for q in best.model.parameters()),
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "per_class_map50": {classes[int(c)]: float(a) for c, a in zip(present, m.box.ap50)},
        "per_class_map50_95": {classes[int(c)]: float(m.box.maps[int(c)]) for c in present},
        "per_class_p": {classes[int(c)]: float(v) for c, v in zip(present, m.box.p)},
        "per_class_r": {classes[int(c)]: float(v) for c, v in zip(present, m.box.r)},
        "speed_ms": m.speed,
    }
    dst = runs / f"{args.name}_test.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"{args.name}  mAP50 {out['map50']:.3f}  mAP50-95 {out['map50_95']:.3f}  ({train_min:.0f} min)")
    print("wrote", dst)


if __name__ == "__main__":
    main()
