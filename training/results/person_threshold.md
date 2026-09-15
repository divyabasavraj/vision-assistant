# Person threshold, measured in the narrator's bands

Run: `training/.venv/bin/python training/scripts/person_threshold.py` · weights `n640_v1/best.pt` ·
test split (904 images, 644 person instances: far 331 / near 139 / close 174) · one model pass at
conf 0.35, every threshold evaluated offline from the same predictions · 2026-09-15.

## Why

The field test kept hearing "person" for things that were not people (a chair, once). The
confusion matrix explains it: of everything the model boxes that is really background, 32% is
called person, 27% car, every other class under 10%. Person is the largest training class (5,463
boxes, 28% of all) on top of a COCO pre-training where it dominates even more, so "upright thing I
do not recognise" resolves to person.

The flat sweep (`threshold_sweep.md`) called 0.35 -> 0.50 "marginal" for person (P +0.083,
R -0.053). `recall_by_band.md` showed person's flat recall is dragged down by tiny far figures the
narrator never speaks, which suggested the recall cost of a higher bar might land mostly in the
ignored band. That was the hypothesis. It was wrong.

## Result

| person thr | precision | TP | false persons | far R (ignored) | near R (spoken) | close R (urgent) |
|---|---|---|---|---|---|---|
| **0.35** (was) | 0.691 | 311 | 139 | 0.202 | 0.734 | **0.816** |
| **0.40** (now) | 0.730 | 305 | 113 | 0.193 | 0.712 | **0.816** |
| 0.45 | 0.746 | 297 | 101 | 0.181 | 0.705 | 0.799 |
| 0.50 | 0.774 | 277 | 81 | 0.154 | 0.676 | 0.759 |
| 0.55 | 0.786 | 254 | 69 | 0.133 | 0.612 | 0.718 |

- **0.50 is not free.** Close-band recall falls 0.816 -> 0.759: about 10 of the 174 people right in
  front of the user go unspoken, near loses 0.058 more. A walking aid cannot pay that for precision.
- **0.40 is the last step that costs nothing at close range.** Close recall identical, near -0.022
  (3 of 139), and 26 of 139 false persons (19%) gone. Shipped as `CLASS_THRESHOLD.person = 0.4`.
- Beyond 0.40 every step buys precision with close-band recall. The curve is the argument.

## What sits under a false person

| person thr | nothing labelled | bicycle | motorbike | chair |
|---|---|---|---|---|
| 0.35 | 118 | 10 | 7 | 2 |
| 0.50 | 68 | 6 | 3 | 2 |

- **The chair is not a threshold problem.** Two false persons sit on a labelled chair, and both
  survive every bar to 0.50: they are confident errors. The chair seen in the field is a data
  problem — 358 chair training boxes, and evidently not that kind of chair — so it belongs to
  Task 9 as hard negatives and extra chair examples, not to a threshold.
- bicycle/motorbike under a person box is mostly the rider: the label sets disagree on whether a
  rider is a person. Not an error the app would notice.
- 118 of 139 sit on nothing labelled. Some are hallucinations; some are people the merged datasets
  never labelled, which is common in merged sets. True precision is somewhat better than 0.69.

## Method note

Match at IoU >= 0.5, greedy by score, one prediction per ground truth, as in the other two
scripts. Bands use box height as a fraction of the frame with person's size scale of 1.0
(far < 0.25, near < 0.50, else close), the same rule as `bucketOf()` in `lib/narrator.ts`.
Raw rows in `person_threshold.json`.
