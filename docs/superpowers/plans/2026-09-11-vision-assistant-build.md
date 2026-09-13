# Vision-to-Speech Assistant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a public Vercel URL where a phone worn in a chest pouch narrates obstacles in real time (fine-tuned YOLO11n running in the browser) and answers spoken questions (cloud vision-LLM), plus the trained model, ablations, journal, and viva prep that make it a gradeable Deep Learning project.

**Architecture:** Two tiers. Tier 1 is a YOLO11n detector fine-tuned by us on merged Roboflow navigation datasets, exported to ONNX and run in the phone browser by ONNX Runtime Web; a pure-logic narration engine turns boxes into short spoken phrases. Tier 2 is one Next.js server route `/api/ask` that sends a question + one JPEG frame to Gemini (free tier) with Groq as fallback. Camera, mic and speaker are all browser APIs; the camera sits behind a `FrameSource` interface so a wearable camera can replace it later.

**Tech Stack:** Python 3.11+ / `ultralytics` / Google Colab (training) · Next.js App Router + TypeScript + Tailwind · `onnxruntime-web` · Web Speech API · Vercel AI SDK (`ai`, `@ai-sdk/google`, `@ai-sdk/groq`) · `zod` · Vitest + pytest · Vercel Hobby.

**Spec:** `docs/superpowers/specs/2026-09-11-vision-assistant-design.md`

## Global Constraints

- Budget: **$0**. Only free tiers (Vercel Hobby, Google AI Studio free key, Groq free key, Colab/Kaggle free GPU, Roboflow free account).
- Deadline: **2026-09-22**. Timeline in spec §10 is the schedule; each task below names its day.
- Training hardware: MacBook M5 Air (PyTorch MPS) for quick runs; Colab/Kaggle GPU for real runs.
- Base model: **YOLO11n**, COCO pre-trained, fine-tuned (not trained from scratch). Export ONNX opset 17, fixed input `1×3×640×640`.
- Dataset target: 3,000–6,000 images, 80/10/10 split, **test set untouched until final evaluation**. Merge time cap: 4 hours, else use the single best dataset and note it in the report.
- `/api/ask`: input `{ question: string, image: base64 JPEG }`, reject > 1 MB, rate-limit 10/min per IP, output `{ answer }` or `{ error: "retry" | "offline" }`. API keys only in Vercel env vars.
- Narration constants (spec §5): IoU match ≥ 0.4, confirm after 3 frames, drop after 10 unseen frames, one phrase per 1.5 s, same object not re-announced within 5 s unless closer, buckets far < 0.25 ≤ near < 0.5 ≤ close (box height / frame height), zones = thirds of frame width.
- Trigger word: **"vision"**. Voice commands "pause"/"resume" never reach the server.
- Learning rule (spec §8): every source file starts with a plain-English header (purpose, dependencies, one question the student should be able to answer). One `docs/journal/YYYY-MM-DD.md` per working day. `docs/viva-prep.md` grows each task. No file over ~200 lines without a reason. **The student runs training, phone tests, and eval collection.**
- Every task starts with a **Plain-English** block. Read it, say "ok", and only then write code.

---

## File Structure (locked)

```
PROJECT FINAL/
  app/
    layout.tsx                 dark full-screen shell, metadata, viewport
    page.tsx                   main screen: mounts Assistant, renders components
    globals.css                Tailwind import
    api/ask/route.ts           the one server function (Tier 2)
  lib/
    frameSource.ts             FrameSource interface + CameraFrameSource (phone camera)
    detector.ts                load ONNX, preprocess (letterbox), run, decode, NMS
    narrator.ts                tracking + events + priority + phrase text (pure logic)
    voiceCommands.ts           parse "vision, ..." transcripts into commands (pure logic)
    speech.ts                  TTS + SpeechRecognition wrappers (browser only)
    listener.ts                always-on trigger-word listener with auto-restart
    ask.ts                     client: frame → JPEG base64, POST /api/ask
    assistant.ts               orchestrator: wires camera, detector, narrator, speech, ask
    server/askLogic.ts         validation, fallback chain, rate limiter (pure, testable)
    server/providers.ts        Gemini + Groq answerers via AI SDK
  components/
    CameraView.tsx             <video> + <canvas> box overlay + double-tap pause
    TalkButton.tsx             hold-to-talk backup button
    Caption.tsx                large high-contrast text of what was said/heard
  tests/                       Vitest unit tests (pure logic only)
    detector.test.ts
    narrator.test.ts
    voiceCommands.test.ts
    askLogic.test.ts
  public/models/best.onnx      deployed detector (copied from training output)
  training/
    requirements.txt
    scripts/merge_datasets.py  merge + remap + dedupe + split Roboflow exports
    scripts/make_baseline_labels.py  relabel test set into COCO ids for baseline eval
    scripts/quantize.py        fp32 → int8 ONNX for the quantization ablation
    mappings/<dataset>.json    source class name → our class name, one per raw dataset
    tests/test_merge.py, tests/test_baseline.py
    train.ipynb                Colab notebook (all runs, curves, tables)
    data.yaml                  produced by merge script
    data/                      gitignored (raw/, merged/, baseline/)
    results/                   metrics json/csv copied back from Colab (committed)
  docs/
    journal/YYYY-MM-DD.md
    viva-prep.md
    eval/                      narration logs, Q&A sheet, latency numbers (Task 11)
    report/                    report outline + figures (Task 12)
```

---

## Task 1: Project skeleton, tooling, journal (Day 1 — Sep 11)

**Plain-English:** We create the empty house before furnishing it. `git` so every change is saved with a message; Next.js so one project holds both the web page and the server function; Tailwind so styling is quick; Vitest so we can test pure logic (narrator, decoder) on the laptop without a phone. The journal and viva files exist from day one so writing them becomes a habit, not a last-day panic.

**Files:**
- Create: Next.js scaffold (via CLI), `vitest.config.ts`, `.gitignore` additions, `docs/journal/TEMPLATE.md`, `docs/journal/2026-09-11.md`, `docs/viva-prep.md`, `training/requirements.txt`, `README.md`

- [x] **Step 1: Init git and scaffold Next.js in the current folder**

```bash
cd "/Users/divyabasavraj/projects all/PROJECT FINAL"
git init
npx create-next-app@latest . --yes --typescript --tailwind --eslint --app --src-dir=false --import-alias "@/*" --turbopack
```
If create-next-app refuses because the folder is not empty (it has `docs/`), run it in a temp folder and move the files in:
```bash
npx create-next-app@latest /tmp/va-scaffold --yes --typescript --tailwind --eslint --app --src-dir=false --import-alias "@/*" --turbopack
rsync -a --exclude node_modules /tmp/va-scaffold/ "/Users/divyabasavraj/projects all/PROJECT FINAL/"
npm install
```

- [x] **Step 2: Add test + runtime dependencies**

```bash
npm install onnxruntime-web ai @ai-sdk/google @ai-sdk/groq zod
npm install -D vitest
```

- [x] **Step 3: Vitest config and test script**

Create `vitest.config.ts`:
```ts
// Vitest config. We only test pure logic files (no browser), so no DOM environment is needed.
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
});
```
Add to `package.json` scripts: `"test": "vitest run"`, `"test:watch": "vitest"`.

- [x] **Step 4: Smoke test that Vitest runs**

Create `tests/smoke.test.ts`:
```ts
import { describe, it, expect } from "vitest";
describe("vitest", () => { it("runs", () => { expect(1 + 1).toBe(2); }); });
```
Run: `npm test` → Expected: `1 passed`. Then delete `tests/smoke.test.ts`.

- [x] **Step 5: gitignore training junk, keep the deployed model**

Append to `.gitignore`:
```
# training
training/data/
training/.venv/
training/runs/
*.pt
!public/models/best.onnx
.env*.local
```

- [x] **Step 6: Python env for training scripts**

Create `training/requirements.txt`:
```
ultralytics>=8.3
pyyaml
pytest
onnx
onnxruntime
```
```bash
python3 -m venv training/.venv
source training/.venv/bin/activate
pip install -r training/requirements.txt
python -c "import ultralytics, torch; print(ultralytics.__version__, torch.backends.mps.is_available())"
```
Expected: version printed and `True` (MPS = Apple GPU available).

- [x] **Step 7: Journal template, first entry, viva file**

Create `docs/journal/TEMPLATE.md`:
```markdown
# YYYY-MM-DD — Day N

## What I built today
-

## Why I chose it this way
-

## What broke and how I fixed it
-

## What I learned (in my own words)
-

## Numbers from today (if any)
-

## Tomorrow
-
```
Create `docs/journal/2026-09-11.md` from the template; fill "What I built today" with: project skeleton, git, Next.js, Vitest, Python env. Fill "What I learned" in your own words (what is Next.js, what is a unit test, what is a virtual environment).

Create `docs/viva-prep.md`:
```markdown
# Viva preparation — questions a professor may ask

Short answers in plain words. Grows every day.

## Q: What is the project in one sentence?
A web app for blind users: a small object detector that I fine-tuned runs inside the phone browser and speaks obstacles in real time; a large cloud vision model answers spoken questions on demand.

## Q: Why is this a deep learning project if you used a pre-trained model?
The core deliverable is a neural network (YOLO11n) that I trained further on my own merged dataset so it learns new classes (stairs, curb, door, pole…). I measure before vs after, run ablations, and deploy it. Fine-tuning is how almost all real-world detectors are built; training from scratch would need millions of images.

## Q: Why two tiers instead of one big model?
Cloud vision models take 2–4 s per call and are rate-limited, useless for walking. A nano model on the phone gives results in under 100 ms and costs nothing. Each model does what it is good at.
```

- [x] **Step 8: README and commit**

Create `README.md` with the one-sentence goal, how to run (`npm run dev`), how to test (`npm test`), and links to the spec, plan, journal and viva files.
```bash
git add -A
git commit -m "chore: scaffold Next.js app, vitest, python env, journal + viva files"
```

---

## Task 2: Dataset merge script (Day 1 — Sep 11–12)

**Plain-English:** Public datasets each use their own class names and numbering ("Stairs", "stair", "steps"…). Our script reads each downloaded dataset, translates its class names into our single list using a small mapping file you write by hand, drops classes we don't want, removes exact duplicate images, and splits into train/val/test with a fixed random seed (so the split is reproducible). The output is one folder + one `data.yaml` that `ultralytics` understands. We test the logic with tiny fake datasets before touching real data.

**Files:**
- Create: `training/scripts/merge_datasets.py`, `training/tests/test_merge.py`, `training/mappings/README.md`

**Interfaces:**
- Produces: `TARGET_CLASSES: list[str]`, `remap_label_line(line, source_names, mapping) -> str | None`, `split_name(index, n, seed) -> "train"|"val"|"test"`, `merge(raw_dir, mappings_dir, out_dir, seed) -> dict` (per-class, per-split counts), `training/data/merged/data.yaml`.

- [x] **Step 1: Write the failing tests**

Create `training/tests/test_merge.py`:
```python
# Tests for merge_datasets.py using tiny fake datasets built in a temp folder.
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_datasets as m


def test_remap_label_line_translates_class_id():
    source_names = ["Stairs", "Person", "Trash"]
    mapping = {"Stairs": "stairs_up", "Person": "person"}
    line = "0 0.5 0.5 0.2 0.3"
    out = m.remap_label_line(line, source_names, mapping)
    assert out == f"{m.TARGET_CLASSES.index('stairs_up')} 0.5 0.5 0.2 0.3"


def test_remap_label_line_drops_unmapped_class():
    assert m.remap_label_line("2 0.5 0.5 0.2 0.3", ["a", "b", "Trash"], {"a": "person"}) is None


def test_split_is_deterministic_and_roughly_80_10_10():
    names = [m.split_name(i, 1000, seed=42) for i in range(1000)]
    again = [m.split_name(i, 1000, seed=42) for i in range(1000)]
    assert names == again
    assert 750 <= names.count("train") <= 850
    assert 50 <= names.count("val") <= 150
    assert 50 <= names.count("test") <= 150


def _fake_dataset(root: Path, name: str, class_names, files):
    """files: list of (filename, image_bytes, label_text)"""
    d = root / name
    (d / "train" / "images").mkdir(parents=True)
    (d / "train" / "labels").mkdir(parents=True)
    (d / "data.yaml").write_text("names: " + json.dumps(class_names) + "\nnc: %d\n" % len(class_names))
    for fname, img_bytes, label in files:
        (d / "train" / "images" / fname).write_bytes(img_bytes)
        (d / "train" / "labels" / (Path(fname).stem + ".txt")).write_text(label)


def test_merge_remaps_dedupes_and_writes_yaml(tmp_path: Path):
    raw = tmp_path / "raw"; raw.mkdir()
    maps = tmp_path / "mappings"; maps.mkdir()
    out = tmp_path / "merged"
    _fake_dataset(raw, "dsA", ["Stairs", "Person"], [
        ("a1.jpg", b"IMG1", "0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.1 0.1\n"),
        ("a2.jpg", b"IMG2", "1 0.5 0.5 0.2 0.3\n"),
    ])
    _fake_dataset(raw, "dsB", ["car", "Stairs"], [
        ("b1.jpg", b"IMG1", "0 0.5 0.5 0.2 0.3\n"),   # same bytes as a1 -> duplicate, dropped
        ("b2.jpg", b"IMG3", "1 0.5 0.5 0.2 0.3\n"),
        ("b3.jpg", b"IMG4", "5 0.5 0.5 0.2 0.3\n"),   # bad class id -> line dropped, image dropped (no labels left)
    ])
    (maps / "dsA.json").write_text(json.dumps({"Stairs": "stairs_up", "Person": "person"}))
    (maps / "dsB.json").write_text(json.dumps({"car": "car", "Stairs": "stairs_up"}))

    counts = m.merge(raw, maps, out, seed=1)

    kept = sorted(p.name for s in ("train", "val", "test") for p in (out / "images" / s).glob("*.jpg"))
    assert kept == ["dsA_a1.jpg", "dsA_a2.jpg", "dsB_b2.jpg"]
    yaml_text = (out / "data.yaml").read_text()
    assert "stairs_up" in yaml_text and "train: images/train" in yaml_text
    total_stairs = sum(counts[s].get("stairs_up", 0) for s in counts)
    assert total_stairs == 2
```

- [x] **Step 2: Run tests to verify they fail**

Run: `source training/.venv/bin/activate && pytest training/tests/test_merge.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'merge_datasets'`.

- [x] **Step 3: Write the merge script**

Create `training/scripts/merge_datasets.py`:
```python
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
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_merge.py -q` → Expected: `4 passed`.

- [x] **Step 5: Mapping instructions**

Create `training/mappings/README.md`:
```markdown
One JSON file per folder in training/data/raw/, same name as the folder.
Keys = class names exactly as written in that dataset's data.yaml.
Values = one of our TARGET_CLASSES (see scripts/merge_datasets.py).
Leave out any class you don't want; its labels are dropped.
Example — training/mappings/stairs-v3.json:
{ "Stair Up": "stairs_up", "Stair Down": "stairs_down", "Person": "person" }
```

- [x] **Step 6: Commit**

```bash
git add training/scripts/merge_datasets.py training/tests/test_merge.py training/mappings/README.md
git commit -m "feat(training): dataset merge script with remap, dedupe, deterministic split"
```

- [ ] **Step 7: STUDENT — download datasets (time-boxed: 4 h total for Steps 7–9)**

1. Make a free account at https://universe.roboflow.com.
2. Search: "obstacle detection visually impaired", "stairs detection", "blind navigation", "sidewalk obstacle", "curb detection". Pick 2–3 datasets with: ≥ 1,000 images, classes overlapping our list, and a license that allows use (CC BY 4.0 is fine).
3. On each dataset page: Download → Format **YOLOv11** (or YOLOv8, same layout) → download zip.
4. Unzip each into `training/data/raw/<short-name>/` so that `training/data/raw/<short-name>/data.yaml` exists.
5. Open each `data.yaml`, read the `names:` list, and write `training/mappings/<short-name>.json`.
Note the dataset URLs and licenses in `docs/journal/2026-09-11.md` (the report needs them).

- [ ] **Step 8: STUDENT — run the merge and inspect**

```bash
source training/.venv/bin/activate
python training/scripts/merge_datasets.py
cat training/data/merged/counts.json
```
Check: total images 3,000–6,000; every class with fewer than **100 train instances** → remove it from `TARGET_CLASSES` and from the mapping files, re-run. Write the final class list and counts in the journal.
Open 10 random images with their labels to eyeball quality (quick way: `yolo` can plot — see Task 4 Step 1 notebook cell for `plot_labels`, or use Roboflow's preview).

- [ ] **Step 9: Journal + viva + commit**

Append to `docs/viva-prep.md`:
```markdown
## Q: How did you build the dataset?
Merged 2–3 public Roboflow datasets. Each had its own class names, so I wrote a mapping file per dataset to translate names into one shared list, dropped classes I didn't need, removed duplicate images by file hash, and split 80/10/10 with a fixed random seed so the split is reproducible. The test split was never used for training or tuning.

## Q: What is the YOLO label format?
One .txt per image; each line is `class_id cx cy w h` with coordinates as fractions of image width/height (0–1), so labels don't depend on image size.
```
```bash
git add training/mappings docs
git commit -m "data: dataset mappings, final class list, day-1 journal"
```

---

## Task 3: Baseline evaluation of the un-tuned COCO model (Day 1–2)

**Plain-English:** Before we train anything we measure how good the *original* YOLO11n (trained on COCO's 80 everyday classes) is on our test set. That is the "before" number. COCO knows person, car, dog, chair… but has no idea what stairs or a curb are — so on those classes its score is 0 by definition. To compare fairly we re-label our test set into COCO's class numbers for the classes both know, then run the standard `val` command.

**Files:**
- Create: `training/scripts/make_baseline_labels.py`, `training/tests/test_baseline.py`

**Interfaces:**
- Consumes: `TARGET_CLASSES` from Task 2, `training/data/merged/`.
- Produces: `OURS_TO_COCO: dict[str,str]`, `remap_to_coco_line(line, our_names, coco_names) -> str | None`, `training/data/baseline/data.yaml`, `training/results/baseline_test.json`.

- [x] **Step 1: Write the failing test**

Create `training/tests/test_baseline.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_baseline_labels as b

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog"]  # first 17 is enough for the test
OURS = ["person", "dog", "car", "bicycle", "motorbike", "stairs_up"]


def test_overlapping_class_is_translated():
    assert b.remap_to_coco_line("4 0.5 0.5 0.1 0.1", OURS, COCO) == "3 0.5 0.5 0.1 0.1"   # motorbike -> motorcycle


def test_new_class_is_dropped():
    assert b.remap_to_coco_line("5 0.5 0.5 0.1 0.1", OURS, COCO) is None   # stairs_up: COCO can't know it
```

- [x] **Step 2: Run to verify it fails**

Run: `pytest training/tests/test_baseline.py -q` → Expected: `ModuleNotFoundError`.

- [x] **Step 3: Write the script**

Create `training/scripts/make_baseline_labels.py`:
```python
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
    (out / "images").mkdir(parents=True)
    os.symlink((merged / "images" / "test").resolve(), out / "images" / "test")
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
```

- [x] **Step 4: Run tests** → `pytest training/tests -q` → Expected: `6 passed`.

- [x] **Step 5: STUDENT — run the baseline on your Mac**

```bash
source training/.venv/bin/activate
python training/scripts/make_baseline_labels.py
yolo val model=yolo11n.pt data=training/data/baseline/data.yaml split=test imgsz=640 device=mps project=training/runs name=baseline
```
Expected: a table with per-class P, R, mAP50, mAP50-95 for person/car/dog/etc. Copy the summary line and per-class rows into `training/results/baseline_test.md`. Record: overall mAP50 (on overlapping classes only), and note "0.0 on all new classes: stairs_up, stairs_down, curb, pole, door, crosswalk, auto".

- [x] **Step 6: Journal, viva, commit**

Append to `docs/viva-prep.md`:
```markdown
## Q: What is mAP50?
For each class, draw the precision–recall curve counting a prediction as correct when its box overlaps the true box by IoU ≥ 0.5; the area under that curve is AP. mAP50 is the mean of AP over classes. mAP50-95 averages over IoU thresholds 0.5…0.95 and is stricter.

## Q: What is IoU?
Intersection over Union: area where two boxes overlap divided by the area they cover together. 1 = identical, 0 = no overlap.

## Q: What was the baseline?
The original COCO yolo11n with no fine-tuning, scored on my untouched test split after translating my labels into COCO's ids. It gets reasonable AP on person/car/dog and exactly 0 on stairs, curb, door, pole, crosswalk, auto because those classes do not exist in COCO.
```
```bash
git add training/scripts/make_baseline_labels.py training/tests/test_baseline.py training/results docs
git commit -m "feat(training): baseline COCO evaluation on our test split"
```

---

## Task 4: First fine-tune on Colab, export ONNX (Day 2 — Sep 13)

**Plain-English:** Fine-tuning = keep the pre-trained network, replace the last layer so it outputs our class count, and continue training on our images with a small learning rate. The network keeps its general skill (edges, shapes, "person") and learns the new concepts. We run it on Colab's free GPU because 50 epochs on 4,000 images takes ~1 h there vs many hours on the Mac. Then we convert the PyTorch weights to ONNX — a universal format the browser can run.

**Files:**
- Create: `training/train.ipynb` (cells below), `training/results/n640_v1.md`, `public/models/best.onnx`, `public/models/labels.json`

**Interfaces:**
- Produces: `public/models/best.onnx` (input `images` 1×3×640×640 float32, output `output0` 1×(4+nc)×8400), `public/models/labels.json` = JSON array of class names in id order.

- [x] **Step 1: STUDENT — zip the merged dataset and upload to Google Drive** (zipped 518 MB at `training/data/merged.zip`; upload still to do)

```bash
cd training/data && zip -qr merged.zip merged && cd ../..
```
Upload `merged.zip` to Google Drive folder `MyDrive/vision/`.

- [x] **Step 2: Create the notebook cells (save as `training/train.ipynb`, run on Colab with GPU runtime: Runtime → Change runtime type → T4 GPU)**

Cell 1 — setup:
```python
!pip -q install ultralytics
from google.colab import drive
drive.mount('/content/drive')
!rm -rf /content/data && mkdir -p /content/data && unzip -q /content/drive/MyDrive/vision/merged.zip -d /content/data
import yaml, pathlib
y = yaml.safe_load(open('/content/data/merged/data.yaml'))
y['path'] = '/content/data/merged'
yaml.safe_dump(y, open('/content/data/merged/data.yaml', 'w'), sort_keys=False)
print(y['names'])
```
Cell 2 — dataset figures for the report (per-class bar chart + sample images with boxes):
```python
import json, matplotlib.pyplot as plt
counts = json.load(open('/content/data/merged/counts.json'))
classes = y['names']
plt.figure(figsize=(10,4))
for i, s in enumerate(['train','val','test']):
    plt.bar([x + i*0.27 for x in range(len(classes))], [counts[s].get(c,0) for c in classes], width=0.27, label=s)
plt.xticks([x+0.27 for x in range(len(classes))], classes, rotation=60); plt.legend(); plt.title('instances per class per split'); plt.tight_layout()
plt.savefig('/content/drive/MyDrive/vision/fig_class_counts.png', dpi=150)
```
Cell 3 — train run 1:
```python
from ultralytics import YOLO
RUNS = '/content/drive/MyDrive/vision/runs'
model = YOLO('yolo11n.pt')
model.train(data='/content/data/merged/data.yaml', epochs=50, imgsz=640, batch=32, patience=10,
            project=RUNS, name='n640_v1', exist_ok=True, seed=0)
```
Cell 4 — evaluate on the test split + save numbers:
```python
best = YOLO(f'{RUNS}/n640_v1/weights/best.pt')
m = best.val(data='/content/data/merged/data.yaml', split='test', imgsz=640, project=RUNS, name='n640_v1_test', exist_ok=True)
print('mAP50', m.box.map50, 'mAP50-95', m.box.map)
for i, ap in enumerate(m.box.maps): print(classes[i], round(float(ap), 3))
json.dump({'map50': float(m.box.map50), 'map': float(m.box.map), 'per_class_map50_95': {classes[i]: float(a) for i, a in enumerate(m.box.maps)}},
          open(f'{RUNS}/n640_v1_test.json', 'w'), indent=2)
```
Cell 5 — export ONNX:
```python
best.export(format='onnx', opset=17, imgsz=640, simplify=True, dynamic=False)
import shutil; shutil.copy(f'{RUNS}/n640_v1/weights/best.onnx', '/content/drive/MyDrive/vision/best_v1.onnx')
json.dump(classes, open('/content/drive/MyDrive/vision/labels.json', 'w'))
```

- [ ] **Step 3: STUDENT — run all cells; while training, watch `results.csv` and the printed val mAP per epoch**

Expected: training ends by epoch ~30–50 (early stop `patience=10`); test mAP50 clearly above baseline on shared classes and > 0 on new classes. If mAP50 on any class < 0.2, note it — Task 9 addresses it.

- [ ] **Step 4: STUDENT — bring results home**

Download from Drive: `best_v1.onnx` → `public/models/best.onnx`; `labels.json` → `public/models/labels.json`; `runs/n640_v1/results.csv`, `results.png`, `confusion_matrix.png` and `n640_v1_test.json` → `training/results/n640_v1/`. Download the notebook (File → Download .ipynb) → `training/train.ipynb`.

- [ ] **Step 5: Sanity-check the ONNX file on the Mac**

```bash
python - <<'PY'
import onnxruntime as ort, numpy as np, json
s = ort.InferenceSession('public/models/best.onnx')
i = s.get_inputs()[0]; o = s.get_outputs()[0]
print(i.name, i.shape, o.name, o.shape)
out = s.run(None, {i.name: np.zeros((1,3,640,640), np.float32)})[0]
print(out.shape, 'classes in labels.json:', len(json.load(open('public/models/labels.json'))))
PY
```
Expected: `images [1, 3, 640, 640] output0 [1, <4+nc>, 8400]` and `(1, 4+nc, 8400)`. Write these numbers down — `detector.ts` relies on them.

- [ ] **Step 6: Journal, viva, commit**

Append to `docs/viva-prep.md`:
```markdown
## Q: What is fine-tuning and why not train from scratch?
Fine-tuning starts from weights already trained on COCO (millions of boxes) and continues training on my ~4k images. Early layers already detect edges/textures/shapes; only the higher layers and the new output head need to adapt. From scratch, 4k images is far too few — the model would overfit and score badly.

## Q: What does epoch, batch, patience mean?
Epoch = one pass over all training images. Batch = images processed together per gradient step (32). Patience = stop if validation mAP has not improved for 10 epochs (early stopping, avoids overfitting and wasted time).

## Q: What is ONNX and why export?
Open Neural Network Exchange — a framework-neutral file describing the network graph and weights. The browser cannot run PyTorch, but ONNX Runtime Web can run ONNX using WebGPU or WebAssembly.

## Q: What does the model output look like?
One tensor 1×(4+nc)×8400: for each of 8400 candidate positions, a box (cx, cy, w, h in input pixels) plus one score per class. We keep positions whose best class score passes a threshold and remove overlapping duplicates with NMS.
```
```bash
git add public/models training/train.ipynb training/results docs
git commit -m "feat(model): first fine-tuned YOLO11n (n640_v1) exported to ONNX"
```

---

## Task 5: Talking camera — FrameSource, speech, trigger word, first deploy (Day 3 — Sep 14)

**Plain-English:** Today the page opens the back camera, speaks, and listens for "vision, …". No detector yet. Three browser features do the work: `getUserMedia` (camera), `speechSynthesis` (text-to-speech), `SpeechRecognition` (speech-to-text). Browsers only allow camera/mic/speaker after the user taps something, so the page starts with one big **Start** button. The camera is wrapped in a `FrameSource` interface: everything else asks "give me the current frame" and never cares whether it came from a phone or a future ESP32-CAM. Parsing the recognised text into a command is pure string logic, so it gets unit tests. At the end we deploy — camera and mic require HTTPS, and Vercel gives us that for free.

**Files:**
- Create: `lib/frameSource.ts`, `lib/speech.ts`, `lib/voiceCommands.ts`, `lib/listener.ts`, `lib/assistant.ts`, `components/CameraView.tsx`, `components/Caption.tsx`, `components/TalkButton.tsx`, `tests/voiceCommands.test.ts`
- Modify: `app/page.tsx`, `app/layout.tsx`

**Interfaces:**
- Produces:
  - `type Frame = { image: CanvasImageSource; width: number; height: number }`
  - `interface FrameSource { start(): Promise<void>; stop(): void; attach(video: HTMLVideoElement): void; getFrame(): Frame | null }`, `class CameraFrameSource implements FrameSource`
  - `speak(text: string, opts?: { interrupt?: boolean }): Promise<void>`, `isSpeaking(): boolean`, `stopSpeaking(): void`
  - `createRecognizer(opts: { continuous: boolean; onResult: (text: string, isFinal: boolean) => void; onEnd: () => void; onError: (e: string) => void }): Recognizer | null` where `type Recognizer = { start(): void; stop(): void }`
  - `type Command = { kind: "ask"; question: string } | { kind: "pause" } | { kind: "resume" } | { kind: "describe" } | { kind: "read" } | { kind: "empty" }`, `parseCommand(transcript: string, trigger?: string): Command | null`
  - `startWakeWordListener(opts): () => void` (returns stop)
  - `class Assistant` with `start()`, `stop()`, `pauseNarration()`, `resumeNarration()`, `handleCommand(cmd: Command)`, `state: AssistantState`, `onState` callback
  - `type AssistantState = { status: "idle"|"starting"|"running"|"error"; caption: string; heard: string; listening: boolean; narrationPaused: boolean; error?: string; backend?: string; fps: number; boxes: Box[] }` (`Box` is defined in Task 6; until then declare `boxes: unknown[]` and fix in Task 6)

- [x] **Step 1: Failing tests for command parsing**

Create `tests/voiceCommands.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { parseCommand } from "../lib/voiceCommands";

describe("parseCommand", () => {
  it("ignores text without the trigger word", () => {
    expect(parseCommand("what is this")).toBeNull();
  });
  it("extracts the question after the trigger", () => {
    expect(parseCommand("Vision, what's this?")).toEqual({ kind: "ask", question: "what's this" });
  });
  it("handles the trigger in the middle of noise", () => {
    expect(parseCommand("um okay vision read the sign please")).toEqual({ kind: "ask", question: "read the sign please" });
  });
  it("maps pause/resume/describe/read to local commands", () => {
    expect(parseCommand("vision pause")).toEqual({ kind: "pause" });
    expect(parseCommand("vision, stop")).toEqual({ kind: "pause" });
    expect(parseCommand("vision resume")).toEqual({ kind: "resume" });
    expect(parseCommand("vision describe")).toEqual({ kind: "describe" });
    expect(parseCommand("vision read")).toEqual({ kind: "read" });
  });
  it("returns empty when nothing follows the trigger", () => {
    expect(parseCommand("vision")).toEqual({ kind: "empty" });
    expect(parseCommand("vision.")).toEqual({ kind: "empty" });
  });
  it("supports a custom trigger word", () => {
    expect(parseCommand("hey buddy what is that", "buddy")).toEqual({ kind: "ask", question: "what is that" });
  });
});
```

- [x] **Step 2: Run to verify it fails** — `npm test` → Expected: FAIL "Cannot find module '../lib/voiceCommands'".

- [x] **Step 3: Implement `lib/voiceCommands.ts`**

```ts
/**
 * Purpose : Turn what the microphone heard ("um vision, read the sign") into a Command.
 *           Pure string logic — no browser APIs — so it is unit-tested in tests/voiceCommands.test.ts.
 * Depends : nothing.
 * Question you should be able to answer: why are "pause" and "resume" handled here and never sent to the server?
 */
export type Command =
  | { kind: "ask"; question: string }
  | { kind: "pause" }
  | { kind: "resume" }
  | { kind: "describe" }
  | { kind: "read" }
  | { kind: "empty" };

export const TRIGGER_WORD = "vision";

const PAUSE_WORDS = ["pause", "stop", "quiet", "mute"];
const RESUME_WORDS = ["resume", "continue", "start", "unmute"];

export function parseCommand(transcript: string, trigger: string = TRIGGER_WORD): Command | null {
  const text = transcript.toLowerCase();
  const idx = text.indexOf(trigger);
  if (idx === -1) return null;
  const after = text
    .slice(idx + trigger.length)
    .replace(/^[\s,.:;!?-]+/, "")   // strip "vision," / "vision:" punctuation
    .replace(/[.?!,]+$/, "")        // strip trailing punctuation
    .trim();
  if (after === "") return { kind: "empty" };
  const first = after.split(/\s+/)[0];
  if (PAUSE_WORDS.includes(first)) return { kind: "pause" };
  if (RESUME_WORDS.includes(first)) return { kind: "resume" };
  if (after === "describe") return { kind: "describe" };
  if (after === "read") return { kind: "read" };
  return { kind: "ask", question: after };
}
```

- [x] **Step 4: Run tests** — `npm test` → Expected: `6 passed`.

- [x] **Step 5: `lib/frameSource.ts`**

```ts
/**
 * Purpose : "Give me the current camera frame." Everything else (detector, Q&A) talks to this
 *           interface, never to the camera directly, so a wearable camera can replace the phone later.
 * Depends : browser getUserMedia. Client-only.
 * Question you should be able to answer: what would you have to write to use an ESP32-CAM instead?
 */
export type Frame = { image: CanvasImageSource; width: number; height: number };

export interface FrameSource {
  start(): Promise<void>;
  stop(): void;
  attach(video: HTMLVideoElement): void;   // show the live picture on screen
  getFrame(): Frame | null;                // null until the first frame is ready
}

export class CameraFrameSource implements FrameSource {
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    if (this.video) await this.playOn(this.video);
  }

  attach(video: HTMLVideoElement): void {
    this.video = video;
    if (this.stream) void this.playOn(video);
  }

  private async playOn(video: HTMLVideoElement) {
    video.srcObject = this.stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
  }

  getFrame(): Frame | null {
    const v = this.video;
    if (!v || v.readyState < 2 || v.videoWidth === 0) return null;
    return { image: v, width: v.videoWidth, height: v.videoHeight };
  }

  stop(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }
}
```

- [x] **Step 6: `lib/speech.ts`**

```ts
/**
 * Purpose : Thin wrappers over the browser's built-in text-to-speech (speechSynthesis)
 *           and speech-to-text (SpeechRecognition). Client-only.
 * Depends : Web Speech API (Chrome Android + Safari iOS support it; Firefox mobile does not).
 * Question you should be able to answer: why do we need isSpeaking() for the microphone logic?
 */
export type Recognizer = { start(): void; stop(): void };

let speaking = false;

export function isSpeaking(): boolean { return speaking; }

export function stopSpeaking(): void {
  window.speechSynthesis.cancel();
  speaking = false;
}

export function speak(text: string, opts: { interrupt?: boolean } = {}): Promise<void> {
  return new Promise((resolve) => {
    if (opts.interrupt) window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.1;
    u.onstart = () => { speaking = true; };
    u.onend = () => { speaking = false; resolve(); };
    u.onerror = () => { speaking = false; resolve(); };
    window.speechSynthesis.speak(u);
  });
}

// Minimal typing for the prefixed SpeechRecognition constructor (not in TS's DOM lib on all versions).
type SRResultEvent = { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> };
type SRInstance = {
  continuous: boolean; interimResults: boolean; lang: string;
  onresult: ((e: SRResultEvent) => void) | null; onend: (() => void) | null; onerror: ((e: { error: string }) => void) | null;
  start(): void; stop(): void; abort(): void;
};
type SRCtor = new () => SRInstance;

function getCtor(): SRCtor | null {
  const w = window as unknown as { SpeechRecognition?: SRCtor; webkitSpeechRecognition?: SRCtor };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function createRecognizer(opts: {
  continuous: boolean;
  onResult: (text: string, isFinal: boolean) => void;
  onEnd: () => void;
  onError: (error: string) => void;
}): Recognizer | null {
  const Ctor = getCtor();
  if (!Ctor) return null;
  const r = new Ctor();
  r.continuous = opts.continuous;
  r.interimResults = true;
  r.lang = "en-IN";
  r.onresult = (e) => {
    let text = "";
    let isFinal = false;
    for (let i = e.resultIndex; i < e.results.length; i++) {
      text += e.results[i][0].transcript;
      if (e.results[i].isFinal) isFinal = true;
    }
    opts.onResult(text.trim(), isFinal);
  };
  r.onend = () => opts.onEnd();
  r.onerror = (e) => opts.onError(e.error);
  return { start: () => { try { r.start(); } catch { /* already started */ } }, stop: () => r.stop() };
}
```

- [x] **Step 7: `lib/listener.ts` — always-on trigger-word listener**

```ts
/**
 * Purpose : Keep speech-to-text running all the time, look for the trigger word ("vision"),
 *           and hand the parsed Command to the assistant. Browsers stop recognition after a few
 *           seconds of silence, so we restart it automatically and count the restarts (for the report).
 * Depends : lib/speech.ts, lib/voiceCommands.ts.
 * Question you should be able to answer: how do we avoid the app answering its own spoken phrases?
 */
import { createRecognizer, type Recognizer } from "./speech";
import { parseCommand, type Command, TRIGGER_WORD } from "./voiceCommands";

export type ListenerOptions = {
  trigger?: string;
  onCommand: (cmd: Command) => void;
  onTranscript: (text: string) => void;   // for the on-screen caption
  shouldIgnore: () => boolean;            // true while the app itself is speaking
  onRestart?: (count: number) => void;
  onUnsupported?: () => void;
};

export function startWakeWordListener(opts: ListenerOptions): () => void {
  const trigger = opts.trigger ?? TRIGGER_WORD;
  let stopped = false;
  let restarts = 0;
  let rec: Recognizer | null = null;

  const boot = () => {
    rec = createRecognizer({
      continuous: true,
      onResult: (text, isFinal) => {
        if (opts.shouldIgnore()) return;
        opts.onTranscript(text);
        if (!isFinal) return;
        const cmd = parseCommand(text, trigger);
        if (cmd) opts.onCommand(cmd);
      },
      onEnd: () => {
        if (stopped) return;
        restarts += 1;
        opts.onRestart?.(restarts);
        setTimeout(() => { if (!stopped) rec?.start(); }, 300);
      },
      onError: (err) => {
        if (err === "not-allowed" || err === "service-not-allowed") { stopped = true; opts.onUnsupported?.(); }
        // "no-speech" and "aborted" are normal; onEnd will restart us.
      },
    });
    if (!rec) { opts.onUnsupported?.(); return; }
    rec.start();
  };

  boot();
  return () => { stopped = true; rec?.stop(); };
}
```

- [x] **Step 8: `lib/assistant.ts` v1 (camera + voice only; detection is added in Task 6)**

```ts
/**
 * Purpose : The conductor. Owns the camera, the mouth (TTS) and the ears (trigger-word listener),
 *           and later the detector, narrator and Q&A. React only renders what this class reports via onState.
 * Depends : lib/frameSource.ts, lib/speech.ts, lib/listener.ts, lib/voiceCommands.ts.
 * Question you should be able to answer: why keep this logic out of the React component?
 */
import { CameraFrameSource, type FrameSource } from "./frameSource";
import { speak, isSpeaking, stopSpeaking } from "./speech";
import { startWakeWordListener } from "./listener";
import type { Command } from "./voiceCommands";

export type AssistantState = {
  status: "idle" | "starting" | "running" | "error";
  caption: string;        // last thing the app said
  heard: string;          // live transcript
  listening: boolean;     // trigger-word listener active
  narrationPaused: boolean;
  error?: string;
  backend?: string;       // "webgpu" | "wasm" (Task 6)
  fps: number;            // detector frames per second (Task 6)
  boxes: unknown[];       // detections (typed as Box[] in Task 6)
  sttRestarts: number;
};

export class Assistant {
  state: AssistantState = { status: "idle", caption: "", heard: "", listening: false, narrationPaused: false, fps: 0, boxes: [], sttRestarts: 0 };
  readonly frames: FrameSource;
  private stopListening: (() => void) | null = null;

  constructor(private onState: (s: AssistantState) => void, frames?: FrameSource) {
    this.frames = frames ?? new CameraFrameSource();
  }

  private set(patch: Partial<AssistantState>) {
    this.state = { ...this.state, ...patch };
    this.onState(this.state);
  }

  async start(): Promise<void> {
    this.set({ status: "starting" });
    try {
      await this.frames.start();
    } catch (e) {
      this.set({ status: "error", error: "Camera permission denied or no camera: " + String(e) });
      return;
    }
    this.stopListening = startWakeWordListener({
      onCommand: (c) => void this.handleCommand(c),
      onTranscript: (t) => this.set({ heard: t }),
      shouldIgnore: () => isSpeaking(),
      onRestart: (n) => this.set({ sttRestarts: n }),
      onUnsupported: () => this.set({ listening: false, caption: "Voice not supported here — use the button" }),
    });
    this.set({ status: "running", listening: this.stopListening !== null });
    await this.say("camera ready");
  }

  stop(): void {
    this.stopListening?.();
    this.frames.stop();
    stopSpeaking();
    this.set({ status: "idle", listening: false });
  }

  async say(text: string, interrupt = false): Promise<void> {
    this.set({ caption: text });
    await speak(text, { interrupt });
  }

  pauseNarration(): void { this.set({ narrationPaused: true }); void this.say("paused", true); }
  resumeNarration(): void { this.set({ narrationPaused: false }); void this.say("listening", true); }

  async handleCommand(cmd: Command): Promise<void> {
    switch (cmd.kind) {
      case "pause": return this.pauseNarration();
      case "resume": return this.resumeNarration();
      case "empty": return this.say("I didn't catch that", true);
      case "describe": return this.ask("Describe what is in front of me in two short sentences.");
      case "read": return this.ask("Read any text you can see, exactly as written.");
      case "ask": return this.ask(cmd.question);
    }
  }

  // Replaced by the real /api/ask call in Task 8. For now: echo, so we can test the voice path end to end.
  async ask(question: string): Promise<void> {
    await this.say("you asked: " + question, true);
  }
}
```

- [x] **Step 9: Components**

`components/Caption.tsx`:
```tsx
/**
 * Purpose : Big high-contrast text of what the app said and what it heard — for low-vision users and the demo.
 * Depends : nothing.
 * Question you should be able to answer: why is aria-live="polite" on the caption?
 */
export function Caption({ said, heard }: { said: string; heard: string }) {
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 p-4 text-white">
      <p aria-live="polite" className="text-3xl font-bold leading-tight drop-shadow-[0_2px_4px_rgba(0,0,0,.9)]">{said}</p>
      <p className="mt-2 text-lg text-yellow-300 drop-shadow-[0_2px_4px_rgba(0,0,0,.9)]">{heard}</p>
    </div>
  );
}
```
`components/CameraView.tsx` (overlay drawing is added in Task 6):
```tsx
"use client";
/**
 * Purpose : Shows the live camera picture full-screen and reports double-taps (pause/resume narration).
 * Depends : a FrameSource to attach the <video> to.
 * Question you should be able to answer: why must the <video> be muted and playsInline on phones?
 */
import { useEffect, useRef } from "react";
import type { FrameSource } from "@/lib/frameSource";

export function CameraView({ frames, onDoubleTap }: { frames: FrameSource; onDoubleTap: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const lastTap = useRef(0);

  useEffect(() => { if (videoRef.current) frames.attach(videoRef.current); }, [frames]);

  const handleTap = () => {
    const now = Date.now();
    if (now - lastTap.current < 350) onDoubleTap();
    lastTap.current = now;
  };

  return (
    <div className="absolute inset-0 bg-black" onPointerDown={handleTap} role="button" aria-label="Camera view. Double tap to pause or resume narration.">
      <video ref={videoRef} className="h-full w-full object-cover" muted playsInline />
    </div>
  );
}
```
`components/TalkButton.tsx` (hold-to-talk backup; one-shot recognition):
```tsx
"use client";
/**
 * Purpose : Backup input for the demo: hold the bottom third of the screen, ask, release.
 *           Uses a one-shot (non-continuous) recognizer so it does not fight the always-on listener.
 * Depends : lib/speech.ts createRecognizer.
 * Question you should be able to answer: why is this button not needed when the phone is worn?
 */
import { useRef, useState } from "react";
import { createRecognizer, type Recognizer } from "@/lib/speech";

export function TalkButton({ onQuestion, disabled }: { onQuestion: (q: string) => void; disabled?: boolean }) {
  const [holding, setHolding] = useState(false);
  const rec = useRef<Recognizer | null>(null);
  const text = useRef("");

  const down = () => {
    if (disabled) return;
    text.current = "";
    rec.current = createRecognizer({
      continuous: false,
      onResult: (t) => { text.current = t; },
      onEnd: () => { setHolding(false); if (text.current) onQuestion(text.current); },
      onError: () => setHolding(false),
    });
    if (!rec.current) return;
    navigator.vibrate?.(60);
    setHolding(true);
    rec.current.start();
  };
  const up = () => rec.current?.stop();

  return (
    <button
      type="button"
      aria-label="Hold to ask a question"
      disabled={disabled}
      onPointerDown={down} onPointerUp={up} onPointerCancel={up} onPointerLeave={up}
      className={`absolute inset-x-0 bottom-0 h-1/3 select-none text-2xl font-bold text-white ${holding ? "bg-red-700" : "bg-blue-800/80"} disabled:opacity-40`}
    >
      {holding ? "Listening… release to ask" : "Hold to ask"}
    </button>
  );
}
```

- [x] **Step 10: Page and layout**

Replace `app/page.tsx`:
```tsx
"use client";
/**
 * Purpose : The one screen. Start button → Assistant.start(); then camera, captions, presets, hold-to-talk.
 * Depends : lib/assistant.ts, components/*.
 * Question you should be able to answer: why does the app need a Start tap before the camera opens?
 */
import { useEffect, useState } from "react";
import { Assistant, type AssistantState } from "@/lib/assistant";
import { CameraView } from "@/components/CameraView";
import { Caption } from "@/components/Caption";
import { TalkButton } from "@/components/TalkButton";

export default function Page() {
  const [state, setState] = useState<AssistantState | null>(null);
  const [assistant, setAssistant] = useState<Assistant | null>(null);
  useEffect(() => {
    const a = new Assistant(setState);   // created only in the browser, after first render
    setAssistant(a);
    return () => a.stop();
  }, []);
  if (!assistant) return <main className="h-dvh bg-black" />;

  if (!state || state.status === "idle" || state.status === "starting") {
    return (
      <main className="flex h-dvh items-center justify-center bg-black p-6">
        <button type="button" onClick={() => void assistant.start()} disabled={state?.status === "starting"}
          className="h-48 w-full rounded-2xl bg-blue-700 text-4xl font-bold text-white disabled:opacity-50" aria-label="Start the assistant">
          {state?.status === "starting" ? "Starting…" : "Start"}
        </button>
      </main>
    );
  }

  return (
    <main className="relative h-dvh overflow-hidden bg-black">
      <CameraView frames={assistant.frames} onDoubleTap={() => (state.narrationPaused ? assistant.resumeNarration() : assistant.pauseNarration())} />
      <Caption said={state.caption} heard={state.heard} />
      {state.status === "error" && <p role="alert" className="absolute inset-x-0 top-1/3 bg-red-800 p-4 text-xl text-white">{state.error}</p>}
      <div className="absolute inset-x-0 bottom-1/3 flex gap-2 p-2">
        <button type="button" className="flex-1 rounded-xl bg-gray-800/90 py-4 text-xl text-white" onClick={() => void assistant.handleCommand({ kind: "describe" })} aria-label="Describe the scene">Describe</button>
        <button type="button" className="flex-1 rounded-xl bg-gray-800/90 py-4 text-xl text-white" onClick={() => void assistant.handleCommand({ kind: "read" })} aria-label="Read text in view">Read</button>
      </div>
      <TalkButton onQuestion={(q) => void assistant.handleCommand({ kind: "ask", question: q })} />
      <p className="absolute bottom-1 right-2 text-xs text-gray-400">{state.listening ? "🎤 say “vision, …”" : "🎤 off"} · restarts {state.sttRestarts}</p>
    </main>
  );
}
```
In `app/layout.tsx` set metadata `title: "Vision Assistant"`, add `export const viewport = { width: "device-width", initialScale: 1, maximumScale: 1, themeColor: "#000000" }`, and give `<body>` the class `bg-black text-white`.

- [ ] **Step 11: Run locally, then on the phone over the LAN**

```bash
npm run dev -- --hostname 0.0.0.0 --experimental-https
```
Open the printed `https://<mac-ip>:3000` on the phone (accept the self-signed certificate). Expected: Start → camera appears → hears "camera ready" → say "vision, what is this" → caption shows "you asked: what is this" and the phone speaks it. Say "vision pause" → "paused". Note which browser you used (Chrome on Android is the best supported; iOS Safari's continuous recognition stops more often — check `restarts` counter).

- [x] **Step 12: Lint, test, commit**

```bash
npm run lint && npm test && git add -A && git commit -m "feat(app): camera FrameSource, TTS, always-on trigger-word listener, start screen"
```

- [ ] **Step 13: Deploy to Vercel (production, public URL)**

```bash
npm i -g vercel@latest
vercel login
vercel --yes            # links/creates the project, preview deploy
vercel --prod --yes     # public production URL
```
Open the production URL on the phone (no certificate warnings now). Repeat Step 11's checks. Put the URL in `README.md`.

- [ ] **Step 14: Journal, viva, commit**

Append to `docs/viva-prep.md`:
```markdown
## Q: Why does the app run in a browser and not as a native app?
Zero cost, no app store, one public link works on Android and iPhone, and the browser already gives camera, microphone, text-to-speech and speech-to-text for free. Vercel hosts it on HTTPS, which the camera/mic APIs require.

## Q: How does the trigger word work?
The browser's speech-to-text runs continuously. Every final sentence is checked for the word "vision"; the words after it become the command. Pause/resume are handled on the phone. Recognition is ignored while the phone itself is speaking so it does not answer its own voice. Browsers stop recognition after silence, so we restart it automatically and count restarts.

## Q: What is the FrameSource interface for?
It hides where frames come from. Today it is the phone camera; a future ESP32-CAM streaming JPEGs would be a second class with the same four methods, and nothing else in the app changes.
```
```bash
git add docs README.md && git commit -m "docs: day-3 journal, viva entries, public URL"
```

---

## Task 6: Detector in the browser — decode, NMS, ONNX Runtime Web, overlay (Day 4 — Sep 15)

**Plain-English:** The model wants a 640×640 picture with pixel values 0–1 laid out as three colour planes (all reds, then all greens, then all blues). Phone frames are 1280×720, so we shrink them to fit inside 640×640 and pad the rest with grey — that is *letterboxing*. The model returns 8400 candidate boxes; most are junk with low scores, and the good ones come in overlapping duplicates. *Decoding* keeps candidates whose best class score passes a threshold; *NMS* (non-maximum suppression) removes duplicates by dropping any box that overlaps a higher-scoring box of the same class by IoU > 0.45. Decode and NMS are pure maths, so they get unit tests. ONNX Runtime Web tries the phone GPU (WebGPU) first and falls back to WebAssembly on the CPU.

**Files:**
- Create: `lib/detector.ts`, `tests/detector.test.ts`
- Modify: `lib/assistant.ts` (detection loop), `components/CameraView.tsx` (overlay), `app/page.tsx` (fps/backend badge), `next.config.ts`

**Interfaces:**
- Consumes: `Frame` from Task 5; `public/models/best.onnx`, `public/models/labels.json` from Task 4.
- Produces:
  - `type Box = { cls: number; label: string; score: number; cx: number; cy: number; w: number; h: number }` — all coordinates are fractions (0–1) of the **camera frame**.
  - `type RawBox = { cls: number; score: number; x1: number; y1: number; x2: number; y2: number }` (input-pixel space)
  - `decodeOutput(data: Float32Array, numClasses: number, numAnchors: number, scoreThreshold: number): RawBox[]`
  - `iou(a: RawBox, b: RawBox): number`, `nms(boxes: RawBox[], iouThreshold: number): RawBox[]`
  - `type Letterbox = { scale: number; padX: number; padY: number; frameW: number; frameH: number }`, `computeLetterbox(frameW, frameH, inputSize): Letterbox`, `toFrameBox(r: RawBox, lb: Letterbox, labels: string[]): Box`
  - `class Detector { static load(modelUrl, labelsUrl, inputSize?): Promise<Detector>; detect(frame: Frame): Promise<Box[]>; backend: string; labels: string[] }`
  - `AssistantState.boxes: Box[]`, `AssistantState.backend`, `AssistantState.fps`

- [x] **Step 1: Failing tests for the pure parts**

Create `tests/detector.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { decodeOutput, iou, nms, computeLetterbox, toFrameBox, type RawBox } from "../lib/detector";

// Build a fake model output [4+nc, anchors] laid out row-major like ONNX gives it.
function fakeOutput(nc: number, anchors: number, fill: (a: number) => number[] /* cx,cy,w,h,...scores */): Float32Array {
  const rows = 4 + nc;
  const data = new Float32Array(rows * anchors);
  for (let a = 0; a < anchors; a++) { const v = fill(a); for (let r = 0; r < rows; r++) data[r * anchors + a] = v[r]; }
  return data;
}

describe("decodeOutput", () => {
  it("keeps only anchors whose best class score passes the threshold", () => {
    // scores are powers of two so they survive Float32Array exactly and toEqual works
    const data = fakeOutput(2, 3, (a) => (a === 1 ? [100, 100, 50, 40, 0.125, 0.75] : [0, 0, 10, 10, 0.0625, 0.03125]));
    const out = decodeOutput(data, 2, 3, 0.5);
    expect(out).toHaveLength(1);
    expect(out[0]).toEqual({ cls: 1, score: 0.75, x1: 75, y1: 80, x2: 125, y2: 120 });
  });
});

describe("iou / nms", () => {
  const a: RawBox = { cls: 0, score: 0.9, x1: 0, y1: 0, x2: 10, y2: 10 };
  const b: RawBox = { cls: 0, score: 0.8, x1: 5, y1: 0, x2: 15, y2: 10 };  // half overlap
  const c: RawBox = { cls: 1, score: 0.7, x1: 5, y1: 0, x2: 15, y2: 10 };  // same place, other class
  it("computes iou", () => { expect(iou(a, b)).toBeCloseTo(50 / 150, 5); expect(iou(a, { ...a, x1: 20, x2: 30 })).toBe(0); });
  it("drops overlapping lower-score boxes of the same class only", () => {
    expect(nms([b, a, c], 0.3)).toEqual([a, c]);
  });
  it("keeps both when overlap is under the threshold", () => { expect(nms([a, b], 0.5)).toEqual([a, b]); });
});

describe("letterbox mapping", () => {
  it("maps a box from 640-input space back to frame fractions", () => {
    const lb = computeLetterbox(1280, 720, 640);       // scale 0.5, padX 0, padY (640-360)/2 = 140
    expect(lb).toEqual({ scale: 0.5, padX: 0, padY: 140, frameW: 1280, frameH: 720 });
    const box = toFrameBox({ cls: 0, score: 0.9, x1: 160, y1: 140 + 90, x2: 480, y2: 140 + 270 }, lb, ["person"]);
    expect(box.label).toBe("person");
    expect(box.cx).toBeCloseTo(0.5); expect(box.cy).toBeCloseTo(0.5);
    expect(box.w).toBeCloseTo(0.5); expect(box.h).toBeCloseTo(0.5);
  });
});
```

- [x] **Step 2: Run to verify it fails** — `npm test` → FAIL "Cannot find module '../lib/detector'".

- [x] **Step 3: Implement `lib/detector.ts`**

```ts
/**
 * Purpose : Run our fine-tuned YOLO11n ONNX model in the browser and return boxes in frame fractions.
 *           Pure parts (decodeOutput, iou, nms, letterbox maths) are tested in tests/detector.test.ts.
 * Depends : onnxruntime-web (loaded lazily on the client), public/models/best.onnx + labels.json.
 * Question you should be able to answer: what is letterboxing and why do we need to undo it?
 */
import type { Frame } from "./frameSource";

export type Box = { cls: number; label: string; score: number; cx: number; cy: number; w: number; h: number };
export type RawBox = { cls: number; score: number; x1: number; y1: number; x2: number; y2: number };
export type Letterbox = { scale: number; padX: number; padY: number; frameW: number; frameH: number };

export const SCORE_THRESHOLD = 0.35;
export const NMS_IOU = 0.45;
const ORT_VERSION = "1.22.0"; // keep equal to the version in package.json (npm ls onnxruntime-web)

export function decodeOutput(data: Float32Array, numClasses: number, numAnchors: number, scoreThreshold: number): RawBox[] {
  const out: RawBox[] = [];
  for (let a = 0; a < numAnchors; a++) {
    let cls = -1, score = 0;
    for (let c = 0; c < numClasses; c++) {
      const s = data[(4 + c) * numAnchors + a];
      if (s > score) { score = s; cls = c; }
    }
    if (score < scoreThreshold) continue;
    const cx = data[a], cy = data[numAnchors + a], w = data[2 * numAnchors + a], h = data[3 * numAnchors + a];
    out.push({ cls, score, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
  }
  return out;
}

export function iou(a: RawBox, b: RawBox): number {
  const iw = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const ih = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = iw * ih;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union <= 0 ? 0 : inter / union;
}

export function nms(boxes: RawBox[], iouThreshold: number): RawBox[] {
  const sorted = [...boxes].sort((p, q) => q.score - p.score);
  const kept: RawBox[] = [];
  for (const b of sorted) {
    if (kept.every((k) => k.cls !== b.cls || iou(k, b) <= iouThreshold)) kept.push(b);
  }
  return kept;
}

export function computeLetterbox(frameW: number, frameH: number, inputSize: number): Letterbox {
  const scale = Math.min(inputSize / frameW, inputSize / frameH);
  return { scale, padX: (inputSize - frameW * scale) / 2, padY: (inputSize - frameH * scale) / 2, frameW, frameH };
}

export function toFrameBox(r: RawBox, lb: Letterbox, labels: string[]): Box {
  const x1 = (r.x1 - lb.padX) / lb.scale / lb.frameW, x2 = (r.x2 - lb.padX) / lb.scale / lb.frameW;
  const y1 = (r.y1 - lb.padY) / lb.scale / lb.frameH, y2 = (r.y2 - lb.padY) / lb.scale / lb.frameH;
  const clamp = (v: number) => Math.min(1, Math.max(0, v));
  const cx1 = clamp(x1), cx2 = clamp(x2), cy1 = clamp(y1), cy2 = clamp(y2);
  return { cls: r.cls, label: labels[r.cls] ?? String(r.cls), score: r.score, cx: (cx1 + cx2) / 2, cy: (cy1 + cy2) / 2, w: cx2 - cx1, h: cy2 - cy1 };
}

// ---- browser-only part below ----
type Ort = typeof import("onnxruntime-web");
type Session = import("onnxruntime-web").InferenceSession;

export class Detector {
  backend = "unknown";
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private input: Float32Array;

  private constructor(private ort: Ort, private session: Session, public labels: string[], public inputSize: number) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = inputSize;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
    this.input = new Float32Array(3 * inputSize * inputSize);
  }

  static async load(modelUrl: string, labelsUrl: string, inputSize = 640): Promise<Detector> {
    const ort = await import("onnxruntime-web");
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    const labels: string[] = await (await fetch(labelsUrl)).json();
    let session: Session, backend: string;
    try {
      session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["webgpu"] });
      backend = "webgpu";
    } catch {
      session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
      backend = "wasm";
    }
    const d = new Detector(ort, session, labels, inputSize);
    d.backend = backend;
    return d;
  }

  private preprocess(frame: Frame): Letterbox {
    const lb = computeLetterbox(frame.width, frame.height, this.inputSize);
    const S = this.inputSize;
    this.ctx.fillStyle = "#727272";
    this.ctx.fillRect(0, 0, S, S);
    this.ctx.drawImage(frame.image, lb.padX, lb.padY, frame.width * lb.scale, frame.height * lb.scale);
    const px = this.ctx.getImageData(0, 0, S, S).data;
    const plane = S * S;
    for (let i = 0; i < plane; i++) {
      this.input[i] = px[i * 4] / 255;
      this.input[plane + i] = px[i * 4 + 1] / 255;
      this.input[2 * plane + i] = px[i * 4 + 2] / 255;
    }
    return lb;
  }

  async detect(frame: Frame): Promise<Box[]> {
    const lb = this.preprocess(frame);
    const S = this.inputSize;
    const tensor = new this.ort.Tensor("float32", this.input, [1, 3, S, S]);
    const out = await this.session.run({ [this.session.inputNames[0]]: tensor });
    const t = out[this.session.outputNames[0]];
    const [, rows, anchors] = t.dims;           // [1, 4+nc, 8400]
    const raw = decodeOutput(t.data as Float32Array, rows - 4, anchors, SCORE_THRESHOLD);
    return nms(raw, NMS_IOU).map((r) => toFrameBox(r, lb, this.labels));
  }
}
```
Check the version constant: `npm ls onnxruntime-web` and set `ORT_VERSION` to exactly that.

- [x] **Step 4: Run tests** — `npm test` → Expected: all pass (voiceCommands 6 + detector 5).

- [x] **Step 5: Detection loop in `lib/assistant.ts`**

Change `boxes: unknown[]` to `boxes: Box[]` (import `type Box` and `Detector` from `./detector`). Add fields and methods:
```ts
  private detector: Detector | null = null;
  private loopTimer: number | null = null;
  private frameTimes: number[] = [];
```
In `start()`, after the camera starts and before the listener:
```ts
    try {
      this.detector = await Detector.load("/models/best.onnx", "/models/labels.json");
      this.set({ backend: this.detector.backend });
    } catch (e) {
      this.set({ status: "error", error: "Model failed to load: " + String(e) });
      return;
    }
```
After `this.set({ status: "running", ... })`: `this.scheduleLoop();`. In `stop()`: `if (this.loopTimer) clearTimeout(this.loopTimer);`. Add:
```ts
  private scheduleLoop() { this.loopTimer = window.setTimeout(() => void this.tick(), 0); }

  private async tick(): Promise<void> {
    if (this.state.status !== "running" || !this.detector) return;
    const frame = this.frames.getFrame();
    if (frame) {
      const t0 = performance.now();
      const boxes = await this.detector.detect(frame);
      this.frameTimes.push(performance.now() - t0);
      if (this.frameTimes.length > 20) this.frameTimes.shift();
      const avg = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
      this.set({ boxes, fps: Math.round(1000 / Math.max(avg, 1)) });
      this.onBoxes(boxes, performance.now());   // Task 7 puts the narrator here
    }
    this.scheduleLoop();
  }

  // Hook for the narration engine (filled in Task 7).
  protected onBoxes(_boxes: Box[], _nowMs: number): void {}
```

- [x] **Step 6: Box overlay in `components/CameraView.tsx`**

Add prop `boxes: Box[]` and a `<canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />` after the video. Add:
```tsx
  useEffect(() => {
    const c = canvasRef.current, v = videoRef.current;
    if (!c || !v) return;
    // The video is object-cover, so map frame fractions onto the displayed (cropped) area.
    c.width = c.clientWidth; c.height = c.clientHeight;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    const vw = v.videoWidth || 1, vh = v.videoHeight || 1;
    const scale = Math.max(c.width / vw, c.height / vh);
    const dx = (c.width - vw * scale) / 2, dy = (c.height - vh * scale) / 2;
    ctx.lineWidth = 3; ctx.font = "bold 18px sans-serif";
    for (const b of boxes) {
      const x = dx + (b.cx - b.w / 2) * vw * scale, y = dy + (b.cy - b.h / 2) * vh * scale;
      const w = b.w * vw * scale, h = b.h * vh * scale;
      ctx.strokeStyle = "#00ff88"; ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "#00ff88"; ctx.fillText(`${b.label} ${(b.score * 100) | 0}%`, x + 4, y + 20);
    }
  }, [boxes]);
```
Import `type Box` from `@/lib/detector`. In `app/page.tsx` pass `boxes={state.boxes}` and extend the bottom-right badge with `· {state.backend} {state.fps} fps`.

- [ ] **Step 7: Build check and STUDENT phone test**

```bash
npm run build
```
If the build fails on `onnxruntime-web` (typical message mentions `.wasm`/`.mjs` or `node:` modules), add to `next.config.ts`:
```ts
const nextConfig = { serverExternalPackages: ["onnxruntime-web"] };
export default nextConfig;
```
and if still failing, load ORT from the CDN instead of the bundle: in `app/layout.tsx` add `<Script src={`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.min.js`} strategy="beforeInteractive" />` and in `Detector.load` replace the dynamic import with `const ort = (window as unknown as { ort: Ort }).ort;`.

Deploy: `vercel --prod --yes`. On the phone: Start → wait for boxes. Record in the journal: backend (webgpu/wasm), fps, phone model, browser. Point it at a person, a chair, stairs. Expected: ≥ 5 fps on wasm, ≥ 12 fps on webgpu, boxes roughly on the objects.

**Decision point (end of Day 4):** if the model does not load or fps < 3 on wasm, try `inputSize=320` (re-export ONNX with `imgsz=320` on Colab — Task 9 does this anyway for the ablation). If still unusable, fall back to server-side inference as spec §10 says: a Python function on Vercel — record the decision in the journal and ask for the fallback plan before continuing.

- [ ] **Step 8: Journal, viva, commit**

Append to `docs/viva-prep.md`:
```markdown
## Q: What is NMS?
The model proposes many overlapping boxes for one object. Non-maximum suppression sorts by score and drops any box that overlaps an already-kept box of the same class by IoU > 0.45, keeping only the best.

## Q: What is WebGPU vs WASM here?
Two ways ONNX Runtime Web can run the network in a browser: WebGPU uses the phone's graphics chip (fast, newer phones); WebAssembly runs on the CPU (works everywhere, slower). We try WebGPU first and fall back.

## Q: Why 640 input and why does frame rate matter?
640 is what we trained at; halving to 320 is ~4× fewer pixels (faster, less accurate) — that is one of my ablations. The narration engine needs several frames per second to confirm an object over 3 frames and react before the user reaches it.
```
```bash
git add -A && git commit -m "feat(app): in-browser YOLO11n via ONNX Runtime Web with decode, NMS, overlay"
```

---

## Task 7: Narration engine (Day 5 — Sep 16)

**Plain-English:** Boxes arrive 10+ times a second, but a blind user must not hear 10 phrases a second. The narrator remembers objects between frames (tracking by overlap), waits until an object has been seen 3 frames in a row (kills flicker), and only speaks on four events: something new, something got closer, something moved into the path, or the path became clear. It picks the most important event, speaks at most once per 1.5 s, and builds fixed-shape phrases: `<what>, <where>, <how far>`. It is pure logic — a function of (boxes, time) — so we test it with hand-written box lists before ever pointing a camera.

**Files:**
- Create: `lib/narrator.ts`, `tests/narrator.test.ts`
- Modify: `lib/assistant.ts` (`onBoxes`)

**Interfaces:**
- Consumes: `Box` from Task 6.
- Produces: `NARRATOR_CONFIG`, `type Phrase = { text: string; priority: 1|2|3|4; interrupt: boolean }`, `createNarrator(cfg?): { update(boxes: Box[], nowMs: number): Phrase | null; reset(): void }`, helpers `zoneOf(cx)`, `bucketOf(h, cfg)`, `phraseFor(label, zone, bucket)`.

- [x] **Step 1: Failing tests**

Create `tests/narrator.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { createNarrator, zoneOf, bucketOf, phraseFor, NARRATOR_CONFIG, type Phrase } from "../lib/narrator";
import type { Box } from "../lib/detector";

const box = (label: string, cx: number, h: number, cls = 0): Box => ({ cls, label, score: 0.9, cx, cy: 0.5, w: 0.2, h });

describe("helpers", () => {
  it("zones by thirds", () => { expect(zoneOf(0.1)).toBe("left"); expect(zoneOf(0.5)).toBe("ahead"); expect(zoneOf(0.9)).toBe("right"); });
  it("buckets by height", () => { expect(bucketOf(0.1, NARRATOR_CONFIG)).toBe("far"); expect(bucketOf(0.3, NARRATOR_CONFIG)).toBe("near"); expect(bucketOf(0.7, NARRATOR_CONFIG)).toBe("close"); });
  it("formats phrases", () => {
    expect(phraseFor("person", "left", "near")).toBe("person, left, near");
    expect(phraseFor("stairs_down", "ahead", "far")).toBe("stairs down, ahead");   // level changes omit "far"
    expect(phraseFor("stairs_down", "ahead", "close")).toBe("stairs down, ahead, close");
  });
});

describe("narrator events", () => {
  it("says nothing until an object is seen 3 frames in a row, then announces it once", () => {
    const n = createNarrator();
    expect(n.update([box("person", 0.2, 0.3)], 0)).toBeNull();
    expect(n.update([box("person", 0.2, 0.3)], 100)).toBeNull();
    expect(n.update([box("person", 0.2, 0.3)], 200)).toEqual({ text: "person, left, near", priority: 4, interrupt: false });
    expect(n.update([box("person", 0.2, 0.3)], 300)).toBeNull();
  });

  it("filters a one-frame flicker", () => {
    const n = createNarrator();
    n.update([box("dog", 0.5, 0.3)], 0);
    n.update([], 100);
    n.update([box("dog", 0.5, 0.3)], 200);
    expect(n.update([box("dog", 0.5, 0.3)], 300)).toBeNull();   // only 2 consecutive frames so far
  });

  it("announces again when the object gets closer, with priority 2 for moving classes", () => {
    const n = createNarrator();
    for (let t = 0; t < 300; t += 100) n.update([box("person", 0.2, 0.3)], t);
    expect(n.update([box("person", 0.2, 0.6)], 2000)).toEqual({ text: "person, left, close", priority: 2, interrupt: false });
  });

  it("rate-limits to one phrase per 1.5 s but lets priority 1 interrupt", () => {
    const n = createNarrator();
    const both = [box("chair", 0.2, 0.3, 1), box("table", 0.8, 0.3, 2)];
    for (let t = 0; t < 200; t += 100) n.update(both, t);
    expect(n.update(both, 200)?.text).toBe("chair, left, near");
    expect(n.update(both, 300)).toBeNull();                              // table waits
    expect(n.update(both, 1800)?.text).toBe("table, right, near");        // 1.5 s later it speaks
    const pole = [...both, box("pole", 0.5, 0.7, 3)];
    for (let t = 1900; t < 2100; t += 100) n.update(pole, t);
    expect(n.update(pole, 2100)).toEqual({ text: "pole, ahead, close", priority: 1, interrupt: true });
  });

  it("does not repeat the same object within 5 s unless closer", () => {
    const n = createNarrator();
    for (let t = 0; t < 300; t += 100) n.update([box("person", 0.2, 0.3)], t);
    expect(n.update([box("person", 0.5, 0.3)], 2000)).toBeNull();                 // moved to "ahead" but < 5 s
    expect(n.update([box("person", 0.5, 0.3)], 6000)?.text).toBe("person, ahead, near");
  });

  it("says path clear when an ahead-close object disappears", () => {
    const n = createNarrator();
    for (let t = 0; t < 300; t += 100) n.update([box("pole", 0.5, 0.7)], t);
    let out: Phrase | null = null;
    for (let f = 0; f <= NARRATOR_CONFIG.dropAfterFrames; f++) out = n.update([], 3000 + f * 100);  // 11 empty frames -> missed > 10
    expect(out).toEqual({ text: "path clear", priority: 4, interrupt: false });
  });
});
```

- [x] **Step 2: Run to verify it fails** — `npm test` → FAIL "Cannot find module '../lib/narrator'".

- [x] **Step 3: Implement `lib/narrator.ts`**

```ts
/**
 * Purpose : Decide WHEN to speak and WHAT to say from a stream of boxes. Pure logic, no browser.
 *           Tracks objects across frames (IoU matching), confirms after 3 frames, speaks only on events
 *           (new / closer / in path / clear), one phrase per 1.5 s, priority-1 may interrupt.
 * Depends : Box type from lib/detector.ts. Tested in tests/narrator.test.ts.
 * Question you should be able to answer: why do we track objects instead of speaking every frame?
 */
import type { Box } from "./detector";
import { iou, type RawBox } from "./detector";

export const NARRATOR_CONFIG = {
  matchIou: 0.4,
  confirmFrames: 3,
  dropAfterFrames: 10,
  minGapMs: 1500,
  repeatSameObjectMs: 5000,
  nearHeight: 0.25,
  closeHeight: 0.5,
};
export type NarratorConfig = typeof NARRATOR_CONFIG;
export type Zone = "left" | "ahead" | "right";
export type Bucket = "far" | "near" | "close";
export type Phrase = { text: string; priority: 1 | 2 | 3 | 4; interrupt: boolean };

const MOVING = new Set(["person", "dog", "car", "bicycle", "motorbike", "auto", "bus"]);
const LEVEL_CHANGE = new Set(["stairs_up", "stairs_down", "curb"]);
const BUCKET_RANK: Record<Bucket, number> = { far: 0, near: 1, close: 2 };

export function zoneOf(cx: number): Zone { return cx < 1 / 3 ? "left" : cx < 2 / 3 ? "ahead" : "right"; }
export function bucketOf(h: number, cfg: NarratorConfig): Bucket { return h < cfg.nearHeight ? "far" : h < cfg.closeHeight ? "near" : "close"; }
export function phraseFor(label: string, zone: Zone, bucket: Bucket): string {
  const name = label.replace(/_/g, " ");
  if (LEVEL_CHANGE.has(label) && bucket === "far") return `${name}, ${zone}`;
  return `${name}, ${zone}, ${bucket}`;
}

type Track = {
  id: number; label: string; box: Box;
  seen: number; missed: number; confirmed: boolean;
  spokenAt: number | null; spokenBucket: Bucket | null; spokenZone: Zone | null;
};
type Candidate = { track: Track; phrase: Phrase; bucket: Bucket; zone: Zone };

const toRaw = (b: Box): RawBox => ({ cls: b.cls, score: b.score, x1: b.cx - b.w / 2, y1: b.cy - b.h / 2, x2: b.cx + b.w / 2, y2: b.cy + b.h / 2 });

export function createNarrator(cfg: NarratorConfig = NARRATOR_CONFIG) {
  let tracks: Track[] = [];
  let nextId = 1;
  let lastPhraseAt = -Infinity;
  let pendingClear = false;

  function match(boxes: Box[]) {
    const unmatched = new Set(boxes.map((_, i) => i));
    for (const t of tracks) {
      let best = -1, bestIou = cfg.matchIou;
      for (const i of unmatched) {
        if (boxes[i].label !== t.label) continue;
        const v = iou(toRaw(t.box), toRaw(boxes[i]));
        if (v >= bestIou) { bestIou = v; best = i; }
      }
      if (best >= 0) { t.box = boxes[best]; t.seen += 1; t.missed = 0; unmatched.delete(best); }
      else { t.missed += 1; }
    }
    // An unconfirmed track that misses even one frame is thrown away: "3 consecutive frames" means consecutive.
    tracks = tracks.filter((t) => t.confirmed || t.missed === 0);
    for (const i of unmatched) {
      tracks.push({ id: nextId++, label: boxes[i].label, box: boxes[i], seen: 1, missed: 0, confirmed: false, spokenAt: null, spokenBucket: null, spokenZone: null });
    }
    const dropped = tracks.filter((t) => t.missed > cfg.dropAfterFrames);
    if (dropped.some((t) => t.spokenZone === "ahead" && t.spokenBucket === "close")) pendingClear = true;
    tracks = tracks.filter((t) => t.missed <= cfg.dropAfterFrames);
  }

  function priorityOf(t: Track, zone: Zone, bucket: Bucket, closer: boolean): 1 | 2 | 3 | 4 {
    if (zone === "ahead" && bucket === "close") return 1;
    if (MOVING.has(t.label) && closer) return 2;
    if (LEVEL_CHANGE.has(t.label)) return 3;
    return 4;
  }

  function candidates(now: number): Candidate[] {
    const out: Candidate[] = [];
    for (const t of tracks) {
      if (t.missed > 0) continue;
      if (!t.confirmed && t.seen >= cfg.confirmFrames) t.confirmed = true;
      if (!t.confirmed) continue;
      const zone = zoneOf(t.box.cx), bucket = bucketOf(t.box.h, cfg);
      const isNew = t.spokenAt === null;
      const closer = t.spokenBucket !== null && BUCKET_RANK[bucket] > BUCKET_RANK[t.spokenBucket];
      const inPath = zone === "ahead" && t.spokenZone !== null && t.spokenZone !== "ahead";
      if (!isNew && !closer && !inPath) continue;
      if (!isNew && !closer && t.spokenAt !== null && now - t.spokenAt < cfg.repeatSameObjectMs) continue;
      const priority = priorityOf(t, zone, bucket, closer);
      out.push({ track: t, zone, bucket, phrase: { text: phraseFor(t.label, zone, bucket), priority, interrupt: priority === 1 } });
    }
    return out;
  }

  return {
    update(boxes: Box[], nowMs: number): Phrase | null {
      match(boxes);
      const cands = candidates(nowMs).sort((a, b) => a.phrase.priority - b.phrase.priority);
      const gapOk = nowMs - lastPhraseAt >= cfg.minGapMs;
      const pick = cands[0];
      if (pick && (gapOk || pick.phrase.priority === 1)) {
        pick.track.spokenAt = nowMs; pick.track.spokenBucket = pick.bucket; pick.track.spokenZone = pick.zone;
        lastPhraseAt = nowMs;
        pendingClear = false;
        return pick.phrase;
      }
      if (pendingClear && gapOk) {
        const stillBlocked = tracks.some((t) => t.confirmed && t.missed === 0 && zoneOf(t.box.cx) === "ahead" && bucketOf(t.box.h, cfg) === "close");
        pendingClear = false;
        if (!stillBlocked) { lastPhraseAt = nowMs; return { text: "path clear", priority: 4, interrupt: false }; }
      }
      return null;
    },
    reset() { tracks = []; lastPhraseAt = -Infinity; pendingClear = false; },
  };
}
```

- [x] **Step 4: Run tests** — `npm test` → Expected: all narrator tests pass. If the "path clear" test fails on timing, check that `dropAfterFrames` frames of misses have elapsed (`missed > 10` needs 11 empty frames — the test loops `f = 0..10`, i.e. 11 frames).

- [x] **Step 5: Wire into `lib/assistant.ts`**

```ts
import { createNarrator } from "./narrator";
// field:
  private narrator = createNarrator();
// replace the empty hook:
  protected onBoxes(boxes: Box[], nowMs: number): void {
    if (this.state.narrationPaused || this.answering) return;
    const phrase = this.narrator.update(boxes, nowMs);
    if (!phrase) return;
    if (isSpeaking() && !phrase.interrupt) return;   // don't pile up phrases while one is still being spoken
    void this.say(phrase.text, phrase.interrupt);
  }
// field used by Task 8 (true while a question is being answered):
  private answering = false;
```
In `pauseNarration()` also call `this.narrator.reset()` so old tracks don't fire on resume.

- [ ] **Step 6: STUDENT — live tuning walk**

Deploy (`vercel --prod --yes`), wear the phone in the pouch, walk a corridor with a friend. Listen for: phrases too often? (raise `minGapMs`) too late? (lower `confirmFrames` to 2 on low fps) "close" too early/late? (adjust `closeHeight`). Change constants only in `NARRATOR_CONFIG`; note each change and why in the journal. Re-run `npm test` after any change (tests encode the spec thresholds — if you deliberately change a threshold, update the test and write the reason in the journal).

- [ ] **Step 7: Journal, viva, commit**

Append to `docs/viva-prep.md`:
```markdown
## Q: How does the app decide when to speak?
It tracks each object across frames by matching boxes of the same class with IoU ≥ 0.4. An object must be seen 3 frames in a row before it counts (removes flicker). It speaks only on events: new object, got closer (box height crossed a bucket), moved into the middle third, or the path became clear. At most one phrase per 1.5 s; an object ahead and close can interrupt. Same object is not repeated within 5 s unless it got closer.

## Q: How do you estimate distance without a depth sensor?
From box height as a fraction of the frame: < 25% far, 25–50% near, > 50% close. It is a rough proxy (a child and an adult differ) — listed as a limitation.

## Q: Why is the narrator a pure function with unit tests?
Because it is the safety-relevant logic and it must behave the same on every phone. Feeding it fake box lists lets me prove each rule (confirm, rate limit, priority, clear) without a camera.
```
```bash
git add -A && git commit -m "feat(app): narration engine with tracking, events, priority and rate limit"
```

---

## Task 8: Voice Q&A — `/api/ask` with Gemini + Groq fallback (Day 6 — Sep 17)

**Plain-English:** The small detector cannot read a medicine strip or count money. For that, one photo and the question go to a large vision-language model in the cloud. We use one server function so the API keys never reach the phone. The server checks the request shape (zod), limits each phone to 10 questions a minute (protects the free quota), asks Gemini, and if Gemini is rate-limited or errors, asks Groq instead. The fixed system prompt makes answers short, concrete and honest about blur. The pure parts — validation, fallback order, rate limiter — get unit tests with fake "answerers", so we never spend quota in tests.

**Files:**
- Create: `lib/server/askLogic.ts`, `lib/server/providers.ts`, `app/api/ask/route.ts`, `lib/ask.ts`, `tests/askLogic.test.ts`, `.env.local` (gitignored)
- Modify: `lib/assistant.ts` (`ask()`), `lib/speech.ts` (`beep()`)

**Interfaces:**
- Consumes: `Frame` (Task 5), `Assistant.say`, `answering` flag (Task 7).
- Produces:
  - `AskBody` zod schema, `type AskRequest = { question: string; image: string }`, `validateAskBody(body: unknown): { ok: true; data: AskRequest } | { ok: false; message: string }`
  - `type Answerer = (question: string, imageBase64: string) => Promise<string>`, `answerWithFallback(answerers: Answerer[], q, img): Promise<{ answer: string } | { error: "retry" }>`
  - `createRateLimiter(limit: number, windowMs: number): (key: string, now?: number) => boolean` (true = allowed)
  - `SYSTEM_PROMPT`, `MAX_IMAGE_BASE64 = 1_400_000` (1 MB of JPEG ≈ 1.37 MB base64)
  - `geminiAnswerer`, `groqAnswerer: Answerer`
  - client: `frameToJpeg(frame: Frame, maxSide?: number, quality?: number): string`, `askServer(question, imageBase64): Promise<{ answer: string; ms: number } | { error: "retry" | "offline" | "server"; ms: number }>`
  - `beep(): void`

- [ ] **Step 1: Failing tests**

Create `tests/askLogic.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { validateAskBody, answerWithFallback, createRateLimiter, MAX_IMAGE_BASE64 } from "../lib/server/askLogic";

describe("validateAskBody", () => {
  it("accepts a question and a base64 image", () => {
    const r = validateAskBody({ question: "what is this", image: "abc" });
    expect(r).toEqual({ ok: true, data: { question: "what is this", image: "abc" } });
  });
  it("rejects missing fields and oversized images", () => {
    expect(validateAskBody({ question: "" }).ok).toBe(false);
    expect(validateAskBody({ question: "hi", image: "x".repeat(MAX_IMAGE_BASE64 + 1) }).ok).toBe(false);
  });
});

describe("answerWithFallback", () => {
  it("uses the first provider when it works", async () => {
    const r = await answerWithFallback([async () => "A", async () => "B"], "q", "img");
    expect(r).toEqual({ answer: "A" });
  });
  it("falls back when the first provider throws", async () => {
    const r = await answerWithFallback([async () => { throw new Error("429"); }, async () => "B"], "q", "img");
    expect(r).toEqual({ answer: "B" });
  });
  it("returns retry when all fail or answer empty", async () => {
    expect(await answerWithFallback([async () => { throw new Error("x"); }], "q", "img")).toEqual({ error: "retry" });
    expect(await answerWithFallback([async () => "   "], "q", "img")).toEqual({ error: "retry" });
  });
});

describe("createRateLimiter", () => {
  it("allows N per window per key, then blocks, then frees after the window", () => {
    const allow = createRateLimiter(3, 60_000);
    expect([allow("a", 0), allow("a", 1), allow("a", 2), allow("a", 3)]).toEqual([true, true, true, false]);
    expect(allow("b", 3)).toBe(true);
    expect(allow("a", 60_001)).toBe(true);
  });
});
```

- [ ] **Step 2: Run** — `npm test` → FAIL "Cannot find module '../lib/server/askLogic'".

- [ ] **Step 3: `lib/server/askLogic.ts`**

```ts
/**
 * Purpose : The testable brain of /api/ask — request validation, provider fallback order, per-key rate limit.
 *           No network calls here; providers are passed in as functions (see providers.ts).
 * Depends : zod.
 * Question you should be able to answer: why does the fallback treat an empty answer the same as an error?
 */
import { z } from "zod";

export const MAX_IMAGE_BASE64 = 1_400_000;   // ≈ 1 MB of JPEG once base64-encoded
export const SYSTEM_PROMPT =
  "You are assisting a blind person through their phone camera. Answer in one or two short sentences. " +
  "Be direct and concrete. If there is text, read it exactly. If the image is too dark or blurry to tell, say so instead of guessing.";

export const AskBody = z.object({
  question: z.string().trim().min(1).max(300),
  image: z.string().min(1).max(MAX_IMAGE_BASE64),
});
export type AskRequest = z.infer<typeof AskBody>;

export function validateAskBody(body: unknown): { ok: true; data: AskRequest } | { ok: false; message: string } {
  const r = AskBody.safeParse(body);
  return r.success ? { ok: true, data: r.data } : { ok: false, message: r.error.issues.map((i) => i.message).join("; ") };
}

export type Answerer = (question: string, imageBase64: string) => Promise<string>;

export async function answerWithFallback(answerers: Answerer[], question: string, imageBase64: string): Promise<{ answer: string } | { error: "retry" }> {
  for (const answer of answerers) {
    try {
      const text = (await answer(question, imageBase64)).trim();
      if (text) return { answer: text };
    } catch (e) {
      console.warn("provider failed:", (e as Error).message);
    }
  }
  return { error: "retry" };
}

export function createRateLimiter(limit: number, windowMs: number): (key: string, now?: number) => boolean {
  const hits = new Map<string, number[]>();
  return (key, now = Date.now()) => {
    const recent = (hits.get(key) ?? []).filter((t) => now - t < windowMs);
    if (recent.length >= limit) { hits.set(key, recent); return false; }
    recent.push(now);
    hits.set(key, recent);
    return true;
  };
}
```

- [ ] **Step 4: Run tests** — `npm test` → all pass.

- [ ] **Step 5: `lib/server/providers.ts`**

```ts
/**
 * Purpose : The two real vision-LLM callers, both through the Vercel AI SDK so they share one shape.
 *           Gemini first (free tier), Groq's Llama vision as fallback. Keys come from env vars only.
 * Depends : ai, @ai-sdk/google (GOOGLE_GENERATIVE_AI_API_KEY), @ai-sdk/groq (GROQ_API_KEY).
 * Question you should be able to answer: why do we not call these models from the phone directly?
 */
import { generateText } from "ai";
import { google } from "@ai-sdk/google";
import { groq } from "@ai-sdk/groq";
import { SYSTEM_PROMPT, type Answerer } from "./askLogic";

// If a model id is retired, check https://ai.google.dev/gemini-api/docs/models and https://console.groq.com/docs/models
export const GEMINI_MODEL = "gemini-2.5-flash";
export const GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct";

function makeAnswerer(model: Parameters<typeof generateText>[0]["model"]): Answerer {
  return async (question, imageBase64) => {
    const { text } = await generateText({
      model,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: [{ type: "text", text: question }, { type: "image", image: imageBase64, mediaType: "image/jpeg" }] }],
      maxOutputTokens: 120,
      abortSignal: AbortSignal.timeout(12_000),
    });
    return text;
  };
}

export const geminiAnswerer: Answerer = makeAnswerer(google(GEMINI_MODEL));
export const groqAnswerer: Answerer = makeAnswerer(groq(GROQ_MODEL));
```

- [ ] **Step 6: `app/api/ask/route.ts`**

```ts
/**
 * Purpose : The one server function. POST { question, image } -> { answer } | { error }.
 * Depends : lib/server/askLogic.ts, lib/server/providers.ts. Runs on Vercel Node.js (Fluid Compute).
 * Question you should be able to answer: what happens when both Gemini and Groq fail?
 */
import { NextResponse } from "next/server";
import { validateAskBody, answerWithFallback, createRateLimiter } from "@/lib/server/askLogic";
import { geminiAnswerer, groqAnswerer } from "@/lib/server/providers";

export const maxDuration = 30;
const allow = createRateLimiter(10, 60_000);   // per instance; good enough for a demo, noted in the report

export async function POST(req: Request) {
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";
  if (!allow(ip)) return NextResponse.json({ error: "retry" }, { status: 429 });

  let body: unknown;
  try { body = await req.json(); } catch { return NextResponse.json({ error: "bad json" }, { status: 400 }); }
  const v = validateAskBody(body);
  if (!v.ok) return NextResponse.json({ error: v.message }, { status: 400 });

  const result = await answerWithFallback([geminiAnswerer, groqAnswerer], v.data.question, v.data.image);
  return NextResponse.json(result, { status: "answer" in result ? 200 : 503 });
}
```

- [ ] **Step 7: Keys — local and Vercel**

Get free keys: Gemini at https://aistudio.google.com/apikey, Groq at https://console.groq.com/keys. Create `.env.local` (gitignored):
```
GOOGLE_GENERATIVE_AI_API_KEY=...
GROQ_API_KEY=...
```
Add to Vercel: `vercel env add GOOGLE_GENERATIVE_AI_API_KEY production` and `vercel env add GROQ_API_KEY production` (paste when prompted). Test locally:
```bash
npm run dev &
IMG=$(python3 -c "import base64,sys;print(base64.b64encode(open('training/data/merged/images/test/'+__import__('os').listdir('training/data/merged/images/test')[0],'rb').read()).decode())")
curl -s -X POST localhost:3000/api/ask -H 'content-type: application/json' -d "{\"question\":\"describe this\",\"image\":\"$IMG\"}"
```
Expected: `{"answer":"..."}` in 2–4 s. Temporarily put a wrong Gemini key to see the Groq fallback work (log line `provider failed:`), then restore.

- [ ] **Step 8: Client side — `lib/ask.ts` and `beep()`**

`lib/ask.ts`:
```ts
/**
 * Purpose : Phone side of Q&A: shrink the current frame to a small JPEG and POST it with the question.
 * Depends : Frame from lib/frameSource.ts; the /api/ask route.
 * Question you should be able to answer: why shrink to 640 px and quality 0.7 before sending?
 */
import type { Frame } from "./frameSource";

export function frameToJpeg(frame: Frame, maxSide = 640, quality = 0.7): string {
  const scale = Math.min(1, maxSide / Math.max(frame.width, frame.height));
  const c = document.createElement("canvas");
  c.width = Math.round(frame.width * scale);
  c.height = Math.round(frame.height * scale);
  c.getContext("2d")!.drawImage(frame.image, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", quality).split(",")[1];
}

export type AskResult = { answer: string; ms: number } | { error: "retry" | "offline" | "server"; ms: number };

export async function askServer(question: string, imageBase64: string): Promise<AskResult> {
  const t0 = performance.now();
  const ms = () => Math.round(performance.now() - t0);
  if (typeof navigator !== "undefined" && navigator.onLine === false) return { error: "offline", ms: ms() };
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, image: imageBase64 }), signal: AbortSignal.timeout(20_000),
    });
    const data = (await res.json()) as { answer?: string; error?: string };
    if (res.ok && data.answer) return { answer: data.answer, ms: ms() };
    if (res.status === 429 || res.status === 503) return { error: "retry", ms: ms() };
    return { error: "server", ms: ms() };
  } catch {
    return { error: navigator.onLine === false ? "offline" : "server", ms: ms() };
  }
}
```
Add to `lib/speech.ts`:
```ts
let audioCtx: AudioContext | null = null;
/** Short 880 Hz beep: "I heard the trigger word, ask now." */
export function beep(durationMs = 120): void {
  audioCtx ??= new AudioContext();
  const o = audioCtx.createOscillator(); const g = audioCtx.createGain();
  o.frequency.value = 880; g.gain.value = 0.2;
  o.connect(g).connect(audioCtx.destination);
  o.start(); o.stop(audioCtx.currentTime + durationMs / 1000);
}
```

- [ ] **Step 9: Real `ask()` in `lib/assistant.ts`**

Replace the echo version:
```ts
import { frameToJpeg, askServer } from "./ask";
import { beep } from "./speech";
// add to AssistantState: lastAskMs: number   (initialise 0)

  async ask(question: string): Promise<void> {
    if (this.answering) return;
    this.answering = true;
    try {
      beep();
      stopSpeaking();
      const frame = this.frames.getFrame();
      if (!frame) { await this.say("camera not ready", true); return; }
      this.set({ caption: "asking…", heard: question });
      const result = await askServer(question, frameToJpeg(frame));
      this.set({ lastAskMs: result.ms });
      if ("answer" in result) await this.say(result.answer, true);
      else if (result.error === "offline") await this.say("no connection — questions need internet", true);
      else if (result.error === "retry") await this.say("please try again in a moment", true);
      else await this.say("something went wrong, try again", true);
    } finally {
      this.answering = false;
      this.narrator.reset();    // scene may have changed while we were away
    }
  }
```
Show `state.lastAskMs` in the bottom-right badge (`· Q&A {lastAskMs} ms`).

- [ ] **Step 10: STUDENT — end-to-end phone test, then deploy**

`npm run lint && npm test && npm run build`, then `vercel --prod --yes`. On the phone: "vision, what's this?" at a medicine strip, a ₹ note, a sign. Try Describe and Read buttons. Turn on airplane mode → "no connection" message and narration still running. Record 5 questions + answers + ms in the journal.

- [ ] **Step 11: Journal, viva, commit**

Append to `docs/viva-prep.md`:
```markdown
## Q: Which model answers questions and why not run it on the phone?
A cloud vision-language model (Gemini 2.5 Flash, Groq Llama-4 as fallback). Such models are billions of parameters — far too large for a phone browser. They are only needed occasionally, so 2–4 s latency is acceptable there but not for walking.

## Q: How do you keep API keys safe and quota under control?
Keys live only in Vercel environment variables and are used inside the server function. The phone never sees them. The server validates every request with zod, rejects images over 1 MB, and rate-limits each IP to 10 questions per minute.

## Q: What does the system prompt do?
It fixes the model's behaviour: one or two short sentences, concrete, read text exactly, admit when the image is too dark or blurry. Short answers matter because they are spoken aloud.
```
```bash
git add -A && git commit -m "feat(qa): /api/ask with zod validation, rate limit, Gemini->Groq fallback; client ask flow"
```

---

## Task 9: Second fine-tune + ablations + model swap (Day 7 — Sep 18)

**Plain-English:** An ablation changes exactly one thing and re-measures, so you can say *why* the final choice is right rather than *that* it is. We run: image size 320 vs 640 (speed vs accuracy), nano vs small (bigger model vs browser speed), augmentation on vs off (does flipping/colour-jitter help?), and fp32 vs int8 ONNX (does quantisation cost accuracy?). We also fix problems seen in v1 (bad classes, label noise) for the final model. You run everything on Colab; the plan gives the exact cells.

**Files:**
- Create: `training/scripts/quantize.py`, `training/results/ablations.md`, `training/results/<run>/` folders
- Modify: `training/train.ipynb`, `public/models/best.onnx` (+ `labels.json` if classes changed)

- [ ] **Step 1: Decide v2 fixes from v1 results (30 min, journal it)**

Look at `training/results/n640_v1/confusion_matrix.png` and per-class AP. Rules: a class with AP50 < 0.15 and < 150 train instances → remove it from `TARGET_CLASSES` + mappings; classes constantly confused with each other (e.g. `stairs_up`/`stairs_down`) → keep but note; if a dataset is clearly mislabelled, drop it. Re-run merge → re-zip → re-upload if anything changed. **Do not look at test-set images to make these decisions — only at the val metrics and train images.**

- [ ] **Step 2: Ablation cells (append to `train.ipynb`)**

Cell 6 — a helper that trains + evaluates on test + saves json:
```python
def run(name, base='yolo11n.pt', imgsz=640, epochs=50, **extra):
    mdl = YOLO(base)
    mdl.train(data='/content/data/merged/data.yaml', epochs=epochs, imgsz=imgsz, batch=32, patience=10,
              project=RUNS, name=name, exist_ok=True, seed=0, **extra)
    best = YOLO(f'{RUNS}/{name}/weights/best.pt')
    m = best.val(data='/content/data/merged/data.yaml', split='test', imgsz=imgsz, project=RUNS, name=f'{name}_test', exist_ok=True)
    json.dump({'map50': float(m.box.map50), 'map': float(m.box.map),
               'per_class_map50_95': {classes[i]: float(a) for i, a in enumerate(m.box.maps)}, 'speed_ms': m.speed},
              open(f'{RUNS}/{name}_test.json', 'w'), indent=2)
    print(name, 'mAP50', round(float(m.box.map50), 3), 'mAP50-95', round(float(m.box.map), 3))
    return best
```
Cell 7 — the runs (each ~40–70 min on a T4; run in this order and stop when Colab time runs out — minimum three ablations per spec):
```python
final = run('n640_v2')                                  # final candidate (after data fixes)
run('n320_v2', imgsz=320)                               # ablation A: image size
run('s640_v2', base='yolo11s.pt')                       # ablation B: model size
run('n640_noaug_v2', mosaic=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, scale=0.0, translate=0.0)  # ablation C: augmentation
```
Cell 8 — export final + fp16 + evaluate ONNX files:
```python
final.export(format='onnx', opset=17, imgsz=640, simplify=True, dynamic=False)
shutil.copy(f'{RUNS}/n640_v2/weights/best.onnx', '/content/drive/MyDrive/vision/best_v2.onnx')
YOLO('/content/drive/MyDrive/vision/best_v2.onnx').val(data='/content/data/merged/data.yaml', split='test', imgsz=640)   # fp32 ONNX == pt within noise?
```

- [ ] **Step 3: `training/scripts/quantize.py` (ablation D, run on the Mac)**

```python
"""
Purpose : Make an int8 version of the ONNX model (weights stored as 8-bit ints instead of 32-bit floats,
          ~4x smaller) so we can measure accuracy loss and phone speed for the quantisation ablation.
Depends : onnxruntime (quantization tools), onnx.
Question you should be able to answer: what does quantisation trade for size/speed?
Usage   : python training/scripts/quantize.py public/models/best.onnx public/models/best_int8.onnx
"""
import sys
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.preprocess import quant_pre_process

src, dst = sys.argv[1], sys.argv[2]
quant_pre_process(src, "/tmp/pre.onnx")
quantize_dynamic("/tmp/pre.onnx", dst, weight_type=QuantType.QUInt8)
print("wrote", dst)
```
Then evaluate both on the Mac (`yolo val model=public/models/best_int8.onnx data=training/data/merged/data.yaml split=test`) and time both in the browser (Step 5). If the int8 file fails to load in ONNX Runtime Web, export fp16 on Colab instead (`final.export(format='onnx', half=True, ...)`) and compare fp32 vs fp16 — write down which one you did and why.

- [ ] **Step 4: Bring results home; write `training/results/ablations.md`**

Table columns: run · imgsz · params · mAP50 (test) · mAP50-95 · new-class mAP50 · ONNX size MB · phone fps (from Step 5). Rows: baseline COCO, n640_v1, n640_v2, n320_v2, s640_v2, n640_noaug_v2, n640_v2 int8. Copy each run's `results.csv`/`results.png` into `training/results/<run>/`.

- [ ] **Step 5: STUDENT — swap in the final model and measure phone speed**

Copy `best_v2.onnx` → `public/models/best.onnx` (and `labels.json` if classes changed). Deploy. On the phone note fps for: v2 640, v2 int8 (temporarily copy it over `best.onnx`), and — if you exported it — n320 with `Detector.load(..., 320)`. Put the fps numbers into `ablations.md`. Leave the model you will demo with as `public/models/best.onnx`.

- [ ] **Step 6: Journal, viva, commit**

Append to `docs/viva-prep.md`:
```markdown
## Q: What ablations did you run and what did they show?
(fill with your numbers) Image size 320 vs 640: ___. Nano vs small: ___. Augmentation off: ___. int8 quantisation: ___. The final choice is n/640/aug because ___.

## Q: What is data augmentation?
Random changes to training images (flip, colour shift, mosaic of 4 images, scale) so the network sees more variety and overfits less. Turning it off in my ablation changed mAP50 by ___.

## Q: What is quantisation?
Storing weights as 8-bit integers instead of 32-bit floats: ~4× smaller file, often faster on CPU, slightly lower accuracy. In my test: ___.
```
```bash
git add -A && git commit -m "feat(model): v2 fine-tune, ablations (imgsz, model size, aug, int8), final model swapped in"
```

---

## Task 10: Wearable polish — wake lock, earbud button, accessibility, blindfold test (Day 8 — Sep 19)

**Plain-English:** When the phone is in a pouch, nobody touches it. Three things make that work: the screen must not sleep (browsers stop the camera when the phone locks) — the Screen Wake Lock API asks the OS to keep it on; the wired-earbud play/pause button is a second way to ask a question — the browser exposes it through the Media Session API, which only works while some audio is "playing", so we loop a silent clip; and every control must read correctly to a screen reader.

**Files:**
- Create: `lib/mediaButton.ts`
- Modify: `lib/assistant.ts`, `app/page.tsx`, `components/*.tsx`

- [ ] **Step 1: Wake lock in `Assistant.start()`**

```ts
  private wakeLock: WakeLockSentinel | null = null;
  private async keepScreenOn() {
    try { this.wakeLock = await navigator.wakeLock?.request("screen"); } catch { /* not supported: user sets screen timeout to Never */ }
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") void this.keepScreenOn(); });
  }
```
Call `await this.keepScreenOn();` right after the camera starts; release in `stop()` with `this.wakeLock?.release()`.

- [ ] **Step 2: `lib/mediaButton.ts`**

```ts
/**
 * Purpose : Let the play/pause button on wired earbuds trigger "ask a question" (backup to the trigger word).
 *           The Media Session API only delivers button presses while the page is playing audio,
 *           so we loop a silent WAV generated in memory.
 * Depends : Media Session API (Chrome Android, Safari iOS).
 * Question you should be able to answer: why do we need the silent audio loop?
 */
function silentWavUrl(seconds = 1, rate = 8000): string {
  const n = seconds * rate, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const str = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); str(8, "WAVE"); str(12, "fmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, n * 2, true);
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}

export function startMediaButtonListener(onPress: () => void): () => void {
  if (!("mediaSession" in navigator)) return () => {};
  const audio = new Audio(silentWavUrl()); audio.loop = true; audio.volume = 0.01;
  void audio.play().catch(() => {});
  navigator.mediaSession.metadata = new MediaMetadata({ title: "Vision Assistant" });
  for (const action of ["play", "pause", "nexttrack", "previoustrack"] as MediaSessionAction[]) {
    try { navigator.mediaSession.setActionHandler(action, () => { onPress(); void audio.play().catch(() => {}); }); } catch { /* unsupported action */ }
  }
  return () => { audio.pause(); audio.src = ""; };
}
```
In `Assistant.start()`: `this.stopMediaButton = startMediaButtonListener(() => void this.askByButton());` and in `stop()`: `this.stopMediaButton?.();`. Add to `Assistant`:
```ts
  private stopMediaButton: (() => void) | null = null;

  /** Earbud button pressed: beep, listen once for up to 6 s, then ask. */
  private askByButton(): Promise<void> {
    return new Promise((resolve) => {
      beep();
      let text = "";
      const rec = createRecognizer({
        continuous: false,
        onResult: (t) => { text = t; this.set({ heard: t }); },
        onEnd: () => { clearTimeout(timer); void (text ? this.ask(text) : this.say("I didn't catch that", true)).then(resolve); },
        onError: () => { clearTimeout(timer); void this.say("I didn't catch that", true).then(resolve); },
      });
      if (!rec) { void this.say("voice not supported", true).then(resolve); return; }
      const timer = setTimeout(() => rec.stop(), 6000);
      rec.start();
    });
  }
```
(import `createRecognizer` from `./speech`).

- [ ] **Step 3: Accessibility pass**

- Every button: `aria-label` (already), minimum 48×48 px, visible focus ring (`focus-visible:ring-4 ring-yellow-400`).
- `Caption` `aria-live="polite"` for what was said; add `aria-live="assertive"` on the error banner.
- Start screen: `<h1 className="sr-only">Vision Assistant</h1>` and a spoken hint on first load is not possible before a tap — show large text "Tap anywhere to start" and make the whole screen the button.
- Test with TalkBack (Android) or VoiceOver (iOS): every control announces its purpose.

- [ ] **Step 4: STUDENT — pouch + blindfold test with a friend (30 min, the most important test of the project)**

Set phone screen timeout to Never as backup. Put the phone in the pouch, camera forward, one wired earbud. Friend guides for safety. Walk: corridor, doorway, stairs, a chair in the path, a person walking toward you. Ask 3 questions by voice, 1 by earbud button. Record in `docs/journal/2026-09-19.md`: what was announced, what was missed, what was wrong, anything annoying, battery % before/after. This feeds spec §7 items 6 and 9.

- [ ] **Step 5: Commit**

```bash
npm run lint && npm test && npm run build && vercel --prod --yes
git add -A && git commit -m "feat(app): wake lock, earbud media button, accessibility polish"
```
Viva entry: **Q: What did the blindfold test teach you?** — write your own answer.

---

## Task 11: Evaluation collection (Day 9 — Sep 20)

**Plain-English:** The report needs numbers nobody can argue with. We add a tiny debug log to the app so every spoken phrase and every Q&A round-trip is recorded with timestamps and can be downloaded as JSON; then you walk five to ten short clips and count phrases as useful / redundant / wrong; photograph 30 objects for Q&A accuracy; and read latency straight from the log.

**Files:**
- Create: `docs/eval/README.md`, `docs/eval/qa_sheet.csv`, `docs/eval/narration_clips.md`, `docs/eval/latency.md`, `docs/eval/logs/` (JSON downloads)
- Modify: `lib/assistant.ts` (event log), `app/page.tsx` (hidden debug panel)

- [ ] **Step 1: Event log in `Assistant`**

```ts
  log: Array<{ t: number; kind: "phrase" | "ask" | "answer" | "stt-restart" | "error"; text: string; ms?: number; backend?: string; fps?: number }> = [];
  private record(kind: (typeof this.log)[number]["kind"], text: string, extra: Partial<(typeof this.log)[number]> = {}) {
    this.log.push({ t: Math.round(performance.now()), kind, text, backend: this.state.backend, fps: this.state.fps, ...extra });
  }
  downloadLog(): string { return JSON.stringify({ exportedAt: new Date().toISOString(), userAgent: navigator.userAgent, log: this.log }, null, 2); }
```
Call `record("phrase", phrase.text)` in `onBoxes`, `record("ask", question)` and `record("answer", answerOrError, { ms: result.ms })` in `ask()`, `record("stt-restart", String(n))` in the listener callback. In `page.tsx` add a small "Log" button (top-right, `aria-label="Copy debug log"`) that copies `assistant.downloadLog()` to the clipboard via `navigator.clipboard.writeText` and speaks "log copied" — paste into a note on the phone, AirDrop/email to the Mac, save as `docs/eval/logs/<clip-name>.json`. (Viewers cannot download files from the browser sandbox, so clipboard it is.)

- [ ] **Step 2: Narration precision — `docs/eval/narration_clips.md`**

Walk 5–10 clips of 30–60 s each (corridor, stairs, crowded, outdoor, doorway…). For each: copy the log, then fill the table: clip · seconds · phrases spoken · useful · redundant · wrong · precision = useful / spoken. Do the whole set twice: once with default `NARRATOR_CONFIG`, once with `confirmFrames: 2, minGapMs: 1000` (change, deploy, re-walk, restore). Report both — that is spec §7.6.

- [ ] **Step 3: Q&A accuracy — `docs/eval/qa_sheet.csv`**

Header: `id,category,question,expected,answer,verdict,ms,provider`. Categories (5–6 each): currency, medicine, product label, sign/text, scene, chair/empty-seat. Verdict ∈ right / partly / wrong. 30 rows. Provider = which one answered. Add it to the route's JSON without touching `answerWithFallback`: in `app/api/ask/route.ts`
```ts
  let used = "none";
  const named = (name: string, fn: Answerer): Answerer => async (q, img) => { const t = await fn(q, img); used = name; return t; };
  const result = await answerWithFallback([named("gemini", geminiAnswerer), named("groq", groqAnswerer)], v.data.question, v.data.image);
  return NextResponse.json("answer" in result ? { ...result, provider: used } : result, { status: "answer" in result ? 200 : 503 });
```
(import `type Answerer` from `@/lib/server/askLogic`; in `lib/ask.ts` read `data.provider` and include it in `AskResult` as `provider?: string`; log it in `record("answer", …, { ms, provider })`). Average latency from the `ms` column.

- [ ] **Step 4: Latency — `docs/eval/latency.md`**

From the logs: detector ms per frame (fps badge → 1000/fps) on WebGPU and WASM; time from object confirmed to phrase spoken ≈ `confirmFrames / fps + TTS start` (measure TTS start delay with a phrase log vs stopwatch, 5 samples); Q&A round-trip average and p90 from the sheet. One table.

- [ ] **Step 5: Failure images**

Save 4–6 screenshots where the detector was wrong (missed stairs, phantom pole, wrong class) into `docs/report/figures/fail_*.png` with a one-line cause each in `docs/eval/README.md` (dark, motion blur, unusual angle, class confusion, small object).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "eval: narration precision, Q&A accuracy sheet, latency table, failure cases"
```

---

## Task 12: Report, slides, backup video (Day 10 — Sep 21)

**Plain-English:** The report follows spec §7 exactly, in that order, so the marker finds every required item. Write it in your own words from the journal; the plan gives the skeleton and where each number comes from.

**Files:**
- Create: `docs/report/report.md` (export to PDF/DOCX at the end), `docs/report/slides.md`, `docs/report/demo-video-checklist.md`

- [ ] **Step 1: `docs/report/report.md` skeleton**

```markdown
# Vision-to-Speech Assistant for Visually Impaired Users
Author · Course · Date · Live URL · Repo link

1. Problem and goal (½ page) — from spec §1
2. System overview (1 page, architecture diagram from spec §2 redrawn) — two tiers and why
3. Dataset (spec §7.1) — sources with links/licences, class list, counts table + fig_class_counts.png, 6–8 sample images, merge/dedupe/split method, test-set hygiene
4. Model and training (spec §3) — YOLO11n, fine-tuning, config, Colab, export
5. Results — 5.1 baseline vs fine-tuned table (§7.2) · 5.2 ablations table (§7.3) · 5.3 training curves (§7.4, results.png) · 5.4 failure cases (§7.5)
6. Application — 6.1 in-browser inference (§Task 6) · 6.2 narration engine rules + unit tests (§5) · 6.3 Q&A route (§6) · 6.4 wearable design
7. Application evaluation — 7.1 narration precision (§7.6) · 7.2 Q&A accuracy (§7.7) · 7.3 latency (§7.8) · 7.4 blindfold test notes
8. Limitations (§7.9) and ethics (not a safety device)
9. Conclusion and future work (wake-word model, ESP32-CAM, depth)
Appendix: file map with one line per file; how to run; viva-prep link
```
Every number cited must point to a file in `training/results/` or `docs/eval/`.

- [ ] **Step 2: Slides (`docs/report/slides.md`, 10–12 slides)** — title · problem · demo video (30 s) · architecture · dataset · training + before/after · ablations · narration engine · Q&A · evaluation · limitations · what I learned.

- [ ] **Step 3: Backup demo video (3–4 min, phone screen-recorded + a second phone filming you wearing it)** — checklist: start screen → boxes → walk past chair/person/stairs with narration audible → "vision, what's this?" on a medicine strip → Describe button → airplane-mode "no connection" → back online. Store on Drive and USB.

- [ ] **Step 4: Commit** — `git add -A && git commit -m "docs: report, slides, demo video checklist"`

---

## Task 13: Buffer and rehearsal (Day 11 — Sep 22)

- [ ] Rehearse the live demo twice on the actual phone with the actual earbud on the college Wi-Fi/hotspot; note fps and Q&A latency there.
- [ ] Re-read `docs/viva-prep.md` end to end; for every file in `lib/`, be able to say its purpose and its "question you should be able to answer" without looking.
- [ ] `npm run lint && npm test && npm run build && pytest training/tests -q` all green; `vercel --prod --yes` one last time; confirm the README URL opens on someone else's phone.
- [ ] Final commit, zip the repo (without `node_modules`, `training/data`) for submission.

---

## Self-review against the spec

| Spec section | Task |
|---|---|
| §1 two modes, worn design, not-building list | 5, 7, 8, 10 |
| §2 two tiers | 6 (tier 1), 8 (tier 2) |
| §3 classes, data source, merge rules, split, time cap, model, export | 2, 3, 4, 9 |
| §4 stack, phone/server split, folder layout, wearing, UI | 1, 5, 6, 8, 10 |
| §5 narration rules 1–4, constants at top of file | 7 |
| §6 Q&A flow, error table, presets | 5 (presets, empty), 8 (route, errors), 10 (earbud) |
| §7 evaluation items 1–9 | 4 (1, 4), 3+4 (2), 9 (3), 11 (5–8), 12 (9) |
| §8 learning: explain-first, journal, headers, viva, student runs, small files | every task |
| §9 stretch goals | not planned (only if Days 9–10 have slack) |
| §10 timeline + risks | task dates; Task 6 decision point; Task 2 time cap; Task 8 fallback provider; Task 12 backup video |
| §11 success criteria | 13 |

Type consistency checked: `Frame`/`FrameSource` (T5) used by T6/T8; `Box` (T6) used by T7 tests and narrator; `Command` (T5) used by Assistant; `Answerer`/`answerWithFallback` (T8) shared by providers and route; `AssistantState` fields added in T6 (`boxes: Box[]`, `backend`, `fps`), T8 (`lastAskMs`), T11 (log).
