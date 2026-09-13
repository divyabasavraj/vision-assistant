# Baseline: COCO-pretrained YOLO11n on our test split (no fine-tuning)

Run 2026-09-13 on Apple M5 (MPS), Ultralytics 8.4.147, `imgsz=640`.
Test split = 904 images from `training/data/merged` (vip + sod + fukao), labels translated
to COCO ids by `make_baseline_labels.py`; classes that do not exist in COCO are dropped
from the labels, so the baseline scores 0 on them by definition.

```
yolo val model=yolo11n.pt data=training/data/baseline/data.yaml split=test imgsz=640 device=mps project=training/runs name=baseline
```

| Class         | Images | Instances | P     | R     | mAP50 | mAP50-95 |
|---------------|-------:|----------:|------:|------:|------:|---------:|
| **all**       |    904 |      1857 | 0.557 | 0.480 | **0.496** | 0.339 |
| person        |    217 |       644 | 0.420 | 0.616 | 0.455 | 0.278 |
| bicycle       |     61 |        98 | 0.566 | 0.465 | 0.481 | 0.317 |
| car           |    216 |       635 | 0.555 | 0.521 | 0.485 | 0.320 |
| motorcycle    |     56 |       106 | 0.538 | 0.434 | 0.451 | 0.271 |
| bus           |    104 |       175 | 0.709 | 0.640 | 0.729 | 0.553 |
| dog           |     66 |        85 | 0.905 | 0.675 | 0.846 | 0.668 |
| chair         |     36 |        68 | 0.174 | 0.250 | 0.113 | 0.074 |
| dining table  |     36 |        46 | 0.591 | 0.239 | 0.406 | 0.228 |

- **Overall mAP50 on the 8 overlapping classes: 0.496** (mAP50-95 0.339).
- **0.0 on all 7 new classes: stairs_up, stairs_down, curb, pole, door, crosswalk, auto** — they are not in COCO's 80 classes.
- Averaged over all 15 of our classes the baseline is therefore 8/15 × 0.496 ≈ **0.26 mAP50**. That is the number fine-tuning must beat.
- Speed: 1.3 ms inference per image on MPS.
