# Task 9, Step 1 — what changes for v2, and what deliberately does not

*Decided 2026-09-22, before any v2 training started. Evidence: the **val** confusion matrix and
per-epoch curves from `n640_v1/`, plus `merged/counts.json`. Test images were not looked at — the
test per-class table quoted below was already published on Sep 13 and is repeated only as context.*

## Decision 1 — no class is removed

The plan's rule was: drop a class with val AP50 < 0.15 **and** < 150 training instances. Nothing
comes close to firing.

| | weakest class | value |
|---|---|---|
| lowest test AP50 | bicycle | 0.453 |
| fewest train instances | auto | 264 (AP50 0.544) |

The alarm line was 0.15; the floor is 0.453. So v2 is a *strengthening* run, not a rescue, and the
15-class target list carries over unchanged. `labels.json` does not change, which means the ONNX
output stays 1×19×8400 and `detector.ts` needs no edit when the model is swapped.

## Decision 2 — the error is misses, not confusions, so v2 must buy recall

Reading the **background column** of the normalised val confusion matrix (objects that were really
there and were predicted as nothing):

| class | missed | | class | missed |
|---|---:|---|---|---:|
| curb | 0.57 | | person | 0.37 |
| pole | 0.50 | | table | 0.31 |
| crosswalk | 0.45 | | car | 0.30 |
| bicycle | 0.41 | | chair | 0.23 |

Inter-class confusion, by contrast, is tiny. The largest single off-diagonal cell in the whole
matrix is true `auto` predicted as `bus` at 0.12; after that stairs_up↔stairs_down at 0.05 each
way, car→motorbike 0.05, person→motorbike 0.05. Nothing is systematically mistaken for anything
else.

**What follows.** Re-labelling, merging classes or splitting classes would fix a problem this model
does not have. Every intervention in v2 should aim at finding the objects at all. That also matches
`recall_by_band.md`: recall is 0.408 in the far band against 0.773 in the close band — the model
misses small, distant instances.

## Decision 3 — v2 continues from v1's weights instead of restarting

v1 ran all 50 epochs without early-stopping and was still improving when it stopped:

| epoch | 45 | 46 | 47 | 48 | 49 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| val mAP50 | 0.6788 | 0.6793 | 0.6795 | 0.6814 | 0.6858 | **0.6876** |
| val box loss | 1.160 | 1.151 | 1.150 | 1.146 | 1.148 | **1.145** |
| val cls loss | 1.156 | 1.152 | 1.150 | 1.142 | 1.134 | **1.130** |

mAP50 rising *and* both validation losses still falling is the signature of an under-trained model,
not an overfitting one — if it were overfitting, val loss would turn up while train loss kept
falling. So the cheapest recall available is simply more training. `n640_v2` = 30 further epochs
from `n640_v1/best.pt`, same data, same 640 px, batch 32, seed 0.

## Decision 4 — the targeted data collection is deferred, and this is stated rather than hidden

The Sep 15 analysis set a data priority list for v2: distant stairs, chair→person hard negatives,
floor-clutter→dog hard negatives, and curb. **None of it is in v2.** Every item needs photographs
that do not exist in any of the three source datasets — there is not a single far-band stairs
instance in the merged set, so no amount of re-training can teach distant stairs. Collecting and
labelling them is field work, and it stays on the list for Task 11 rather than being quietly
dropped. v2 is honestly described as "same images, trained longer, rare classes shown more
often" (Decision 6).

## Decision 5 — the ablation protocol

Four arms, and only one thing differs per arm:

| arm | changes | everything else |
|---|---|---|
| `n640_e12` (control) | — | yolo11n COCO start, 12 epochs, batch 16, 640 px, seed 0, default aug |
| `n320_e12` | imgsz 320 | as control |
| `n640_noaug_e12` | mosaic/flip/hsv/scale/translate/erasing = 0 | as control |
| `s640_e12` | yolo11s COCO start | as control |
| int8 | post-training dynamic quantisation of the exported ONNX | no training |

Three protocol points worth being able to defend:

1. **Batch 16, not v1's 32.** yolo11s at 640 px does not fit alongside batch 32 in 16 GB of unified
   memory (nano alone used 9.5 GB). Rather than give one arm a different batch size and confound
   it, every arm drops to 16.
2. **12 epochs, not 50.** ~6 min/epoch on this machine; 4 × 50 epochs is 20 h. The arms exist to
   rank a *factor*, and they are all measured under the same budget, so the comparison is valid
   even though every absolute number will sit below v1's 0.667. The headline model stays the full
   run.
3. **`close_mosaic` scaled to 20 % of the budget** (v1 used 10 of 50). Left at the default 10, a
   12-epoch control would have mosaic switched off for 10 of its 12 epochs — the augmentation
   ablation would then be comparing two nearly-unaugmented arms and would show nothing.

Runner: `training/scripts/run_ablations.py` (one arm per invocation), queued by
`training/scripts/run_queue.sh`.

## Decision 6 — rare classes are oversampled in v2 (added 2026-09-30)

*Revised after the Sep 22 local queue died at epoch 2 of v2 and produced nothing; v2 had not been
trained when this was decided.*

Box count alone does not predict AP here — door (408 boxes) scores 0.95 and stairs_up (326) 0.92,
while bicycle (843), pole (1006) and curb (908) sit at 0.45–0.50. So imbalance is not the main
cause of the weak classes; they are thin, small and far (Decision 2). It does plausibly hold back
the two classes that are both **rare and weak**: auto (264 boxes, 143 images, AP50 0.544) and
crosswalk (325, 0.520).

v2 therefore trains on a **repeat-factor-sampled** train list (Gupta et al., LVIS 2019;
`training/scripts/oversample.py`): class factor r_c = max(1, sqrt(t / f_c)) with t = 0.15, where
f_c is the fraction of train images containing c; each image is repeated by the largest factor of
its classes (capped at 4, stochastic rounding, seed 0). Val and test are untouched.

| | before | after |
|---|---:|---:|
| train entries | 7,228 | 10,262 (+42 %) |
| auto boxes | 264 | 747 |
| crosswalk boxes | 325 | 601 |
| chair boxes | 358 | 827 |
| person boxes | 5,463 | 6,444 |

Why oversampling and not the loss's `cls_pw` class weights (Ultralytics 8.4 has both): only one
imbalance intervention per run, so a change in auto/crosswalk AP can be credited to it. `cls_pw`
stays a candidate for a later run. The ablation arms are **not** oversampled — they still compare
image size, model size and augmentation under Decision 5's protocol.

**Hardware.** v2 and all ablation arms run on a Kaggle T4 (`training/kaggle/`), not the Mac.
Every arm shares that hardware, so the arms stay comparable with each other; v1 vs v2 is compared
on the same held-out test split, which is what matters for that comparison.
