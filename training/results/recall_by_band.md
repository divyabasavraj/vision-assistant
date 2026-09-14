# Recall in the bands the narrator actually speaks in

Run: `training/.venv/bin/python training/scripts/recall_by_band.py` · weights `n640_v1/best.pt`
· 904 test images · conf 0.35 (curb 0.5, the shipped per-class bar) · match at IoU >= 0.5.

## Why this metric exists

Flat test-set recall answers "what fraction of labelled objects does the model find?". The
application never asks that question. `bucketOf()` in `lib/narrator.ts` sorts every detection by box
height as a fraction of the frame, scaled by the class's real-world size — far, near, close — and an
object in the **far** band that is not straight ahead is *never spoken*. Measuring recall on those
instances scores the model on work the product deliberately throws away.

Splitting recall by the same bands the app uses:

| class | far (ignored) | near (spoken) | close (urgent) |
|---|---|---|---|
| person | 0.202 (331) | 0.734 (139) | **0.816** (174) |
| dog | — | 0.500 (6) | 0.785 (79) |
| car | 0.556 (450) | 0.722 (72) | **0.947** (113) |
| bicycle | 0.071 (28) | 0.167 (12) | 0.707 (58) |
| motorbike | 0.139 (36) | 0.440 (25) | 0.844 (45) |
| auto | 0.333 (24) | 0.571 (7) | 1.000 (2) |
| bus | 0.706 (119) | 0.933 (45) | 0.818 (11) |
| stairs_up | — | 1.000 (2) | 0.833 (24) |
| stairs_down | — | 0.636 (22) | 0.829 (35) |
| **curb** | 0.000 (2) | 0.462 (13) | **0.482** (85) |
| pole | 0.154 (39) | 0.455 (44) | **0.842** (19) |
| chair | 0.364 (11) | 0.621 (29) | 0.786 (28) |
| table | 0.000 (2) | 0.000 (8) | 0.583 (36) |
| door | — | 1.000 (7) | 0.913 (46) |
| crosswalk | 0.000 (1) | — | 0.571 (42) |
| **ALL** | 0.408 (1043) | 0.657 (431) | **0.773** (797) |

## What it says

**The flat numbers understate the product badly.** 1,043 of 2,271 test instances — 46% — sit in the
far band, and roughly half of those are never spoken about at all. Overall recall on things close
enough to matter is **0.773**, not the ~0.55 a flat count suggests.

**Pole is not the problem it appeared to be.** Flat recall 0.412 looked alarming for a collision
hazard. At close range it is **0.842**. Poles you are about to walk into are found; poles thirty
metres away are not, and the narrator would not mention those anyway.

**Curb is the real weakness, and it is the one that matters most.** 0.482 in the close band, on 85
instances — the model misses roughly half the curbs you are about to step off. A curb is a trip
hazard at exactly the moment it is closest, so there is no distance at which this is acceptable.
This, not pole and not crosswalk, is the class Task 9 should be built around.

**Table (0.583) and crosswalk (0.571) are next**, both on reasonable support. Table is a trip hazard
for the same reason as curb.

**Person at 0.816 close / 0.734 near is healthy.** The flat 0.483 was dominated by tiny distant
figures: recall on person instances under 0.5% of the frame is 0.086, and those are 29% of all
person labels. Crowd scenes look like a crowding failure and are really a size failure — only 3
ground-truth person pairs in the entire test set overlap enough for NMS to be implicated.

## What to change

1. **Task 9 priority order is curb, then table, then crosswalk** — by close-band recall on a safety
   argument, not by flat mAP.
2. **Report both numbers.** Flat recall is the honest comparison against other detectors; band
   recall is the honest description of the product. Quoting only one of them misleads in one
   direction or the other.
