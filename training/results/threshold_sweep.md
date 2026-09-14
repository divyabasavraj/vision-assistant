# Threshold sweep — where should SCORE_THRESHOLD sit?

Run: `training/.venv/bin/python training/scripts/threshold_sweep.py` · weights `n640_v1/best.pt`
· 904 test images · a prediction counts as correct at IoU >= 0.5 against an unclaimed ground-truth
box of the same class.

This exists because curb, pole and crosswalk cannot be demonstrated live in a classroom. The
evidence for them has to be a measured table, not a camera pointed at one example.

## Aggregate

| threshold | mean recall | mean precision | mean F1 |
|---|---|---|---|
| 0.25 | 0.641 | 0.662 | 0.651 |
| 0.35 | 0.609 | 0.722 | 0.661  **<- best F1** |
| 0.45 | 0.575 | 0.762 | 0.656 |
| 0.5 | 0.548 | 0.775 | 0.642 |
| 0.6 | 0.507 | 0.822 | 0.627 |

Mean F1 peaks at **0.35**, not at the 0.5 we shipped. And F1 weighs a miss and a false alarm
equally, which is wrong for this application — for a walking aid a missed pole is worse than an
imagined one, so the true optimum sits at or below 0.35, not above it.

## Per class, 0.35 against 0.5

| class | test inst | R@0.35 | R@0.50 | dR | P@0.35 | P@0.50 | dP | verdict |
|---|---|---|---|---|---|---|---|---|
| auto | 33 | 0.424 | 0.273 | -0.151 | 0.560 | 0.529 | -0.031 | **strictly worse at 0.5** |
| pole | 102 | 0.412 | 0.294 | -0.118 | 0.627 | 0.750 | +0.123 | bad trade — keep low |
| crosswalk | 43 | 0.558 | 0.442 | -0.116 | 0.615 | 0.655 | +0.040 | bad trade — keep low |
| stairs_up | 26 | 0.846 | 0.769 | -0.077 | 0.880 | 0.909 | +0.029 | marginal |
| car | 635 | 0.644 | 0.573 | -0.071 | 0.749 | 0.809 | +0.060 | marginal |
| bus | 175 | 0.771 | 0.709 | -0.062 | 0.794 | 0.821 | +0.027 | marginal |
| person | 644 | 0.483 | 0.430 | -0.053 | 0.691 | 0.774 | +0.083 | marginal |
| stairs_down | 57 | 0.754 | 0.702 | -0.052 | 0.811 | 0.833 | +0.022 | marginal |
| bicycle | 98 | 0.459 | 0.408 | -0.051 | 0.577 | 0.635 | +0.058 | marginal |
| motorbike | 106 | 0.509 | 0.462 | -0.047 | 0.720 | 0.790 | +0.070 | marginal |
| table | 46 | 0.457 | 0.413 | -0.044 | 0.875 | 0.905 | +0.030 | marginal |
| dog | 85 | 0.765 | 0.741 | -0.024 | 0.802 | 0.875 | +0.073 | marginal |
| door | 53 | 0.925 | 0.906 | -0.019 | 0.925 | 0.923 | -0.002 | **strictly worse at 0.5** |
| chair | 68 | 0.647 | 0.632 | -0.015 | 0.667 | 0.741 | +0.074 | **free precision — raise it** |
| curb | 100 | 0.480 | 0.470 | -0.010 | 0.539 | 0.681 | +0.142 | **free precision — raise it** |

## What this changes

**The prediction was half right.** The confusion matrix said the weak classes fail by being missed
entirely, so a higher bar could not help them and would only cost recall. That held for **pole**
(-0.118 recall) and **crosswalk** (-0.116), and the precision it bought came mostly from **person**
(+0.083) and **car** (+0.060) exactly as predicted.

**It was wrong about curb** — the class we were most worried about. Curb barely notices the change
(-0.010 recall) and gains the largest precision jump in the table (+0.142). It wants a *high*
threshold, not a low one. Intuition and the matrix both got this one backwards.

**`auto` is the one pure loss.** It gives up recall *and* precision at 0.5 (-0.151, -0.031). There is
no trade being made there, only damage. It is also the lowest-support class in training (264
instances), which is the likelier root cause.

## Caveat

`stairs_up` (26 test instances), `auto` (33), `crosswalk` (43) and `table` (46) carry wide error
bars. A handful of images moves them several points. Treat the well-supported classes — person
(644), car (635), bus (175), pole (102), curb (100) — as the load-bearing evidence.

## Recommendation

One global threshold cannot serve fifteen classes whose curves point in opposite directions. The
table argues for:

- **global default back to 0.35** — F1-optimal, and recall matters more than precision here
- **per-class override for curb at 0.5** — +0.142 precision for -0.010 recall is close to free
- leave pole, crosswalk, auto, bicycle low; they are recall-starved and a higher bar only hurts

The real fix for pole/crosswalk/auto is not a threshold at all. It is training data — they are the
thin, flat and low-support classes, which is what Task 9 is for.

