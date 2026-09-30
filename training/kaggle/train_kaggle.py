"""
Purpose : Kaggle driver for Task 9 — unpack the merged dataset from the private Kaggle dataset
          `divyabasavraj25/vision-assistant-merged`, then run one job's arms with
          run_ablations.py on the T4. JOB = "v2" / "v3" (candidates) or "ablations" (4 arms).
Depends : Kaggle GPU (T4) + internet (pip ultralytics, COCO weights). Everything lands in
          /kaggle/working/runs, which is what `kaggle kernels output` downloads.
Question you should be able to answer: why is v2 a separate Kaggle run from the ablations?
          v2 is the model that may ship; running it alone brings its numbers back in ~2 h
          instead of after all five arms, and a failed ablation cannot take it down with it.
Usage   : push with training/kaggle/push.sh v2   (or: push.sh ablations)
"""
import glob
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

JOB = "v2"  # push.sh rewrites this line for the job being pushed
# push.sh embeds the current training/scripts files here, so a script change does not need a new
# version of the (large) Kaggle dataset. Empty = fall back to the copies uploaded with the dataset.
SCRIPTS: dict[str, str] = {}

W = Path("/kaggle/working")
RUNS = W / "runs"
ULTRALYTICS = "ultralytics==8.4.147"  # same version as the Mac venv that trained v1


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_input(pattern: str) -> Path | None:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return Path(hits[0]) if hits else None


def prepare_data() -> Path:
    """Return a data.yaml whose `path` points at the merged dataset on this machine."""
    import yaml
    zip_path = find_input("vision_dataset_merged.zip")
    if zip_path:  # unpack to local disk: faster reads, and ultralytics can write its label cache
        say(f"unzipping {zip_path}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall("/tmp/data")
        root = Path("/tmp/data/merged")
    else:  # Kaggle extracted the archive on upload; /kaggle/input is read-only, so copy it out
        src = find_input("merged/images").parent
        say(f"copying {src}")
        root = Path(shutil.copytree(src, "/tmp/data/merged"))
    names = ["person", "dog", "car", "bicycle", "motorbike", "auto", "bus", "stairs_up",
             "stairs_down", "curb", "pole", "chair", "table", "door", "crosswalk"]  # = data.yaml on the Mac
    cfg = dict(path=str(root), train="images/train", val="images/val", test="images/test",
               nc=len(names), names=names)
    dst = W / "merged_data.yaml"
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    for split in ("train", "val", "test"):
        say(f"{split}: {len(list((root / 'images' / split).iterdir()))} images")
    return dst


def run(name: str, *args: str) -> bool:
    if (RUNS / f"{name}_test.json").exists():
        say(f"{name} already done"); return True
    say(f"START {name} {' '.join(args)}")
    (RUNS / "logs").mkdir(parents=True, exist_ok=True)
    with open(RUNS / "logs" / f"{name}.log", "w") as log:
        rc = subprocess.run([sys.executable, str(W / "scripts" / "run_ablations.py"), "--name", name,
                             "--data", str(DATA), "--runs", str(RUNS), "--device", "0",
                             "--workers", "4", *args], stdout=log, stderr=subprocess.STDOUT).returncode
    tail = (RUNS / "logs" / f"{name}.log").read_text().splitlines()[-3:]
    say(f"END {name} (exit {rc})\n  " + "\n  ".join(tail))
    return rc == 0


subprocess.run([sys.executable, "-m", "pip", "install", "-q", ULTRALYTICS], check=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
(W / "scripts").mkdir(exist_ok=True)
for f in ("run_ablations.py", "oversample.py"):
    dst = W / "scripts" / f
    dst.write_text(SCRIPTS[f]) if f in SCRIPTS else dst.write_bytes(find_input(f).read_bytes())
DATA = prepare_data()

if JOB == "v2":
    # Continue from v1 (still improving at epoch 50) on the repeat-factor-sampled train split.
    run("n640_v2", "--base", str(find_input("n640_v1_best.pt")), "--epochs", "30", "--batch", "32",
        "--oversample", "0.15")
elif JOB == "v3":
    # v2's lesson: continue v1 with v1's optimizer (SGD) at a tenth of its rate, no warm-up, and no
    # oversampling, so the only change from v1 is 30 more epochs.
    run("n640_v3", "--base", str(find_input("n640_v1_best.pt")), "--epochs", "30", "--batch", "32",
        "--optimizer", "SGD", "--lr0", "0.001", "--warmup-epochs", "0")
elif JOB == "ablations":
    # Same protocol as v2_decisions.md Decision 5: COCO start, 12 epochs, batch 16, no oversampling.
    run("n640_e12", "--epochs", "12", "--batch", "16")
    run("n320_e12", "--epochs", "12", "--batch", "16", "--imgsz", "320")
    run("n640_noaug_e12", "--epochs", "12", "--batch", "16", "--no-aug")
    if not run("s640_e12", "--base", "yolo11s.pt", "--epochs", "12", "--batch", "16"):
        run("s640_e12_b8", "--base", "yolo11s.pt", "--epochs", "12", "--batch", "8")
say("DONE")
for j in sorted(RUNS.glob("*_test.json")):
    print(j.name, j.read_text()[:400])
